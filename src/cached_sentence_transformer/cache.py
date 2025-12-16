"""SentenceTransformer wrapper with PostgreSQL-backed embedding caching.

This module implements the high-level `CachedSentenceTransformer` class which
coordinates cache lookups, inference for misses, and cache writes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .hashing import stable_id
from .identifiers import sanitize_identifier
from .postgres_kv_store import PostgresKVStore
from .vectors import bytes_to_vector, vector_to_bytes


class CachedSentenceTransformer:
    """SentenceTransformer wrapper that caches embeddings in PostgreSQL.

    Embeddings are stored by a stable hash of (model_key, normalize_embeddings flag, text).

    Args:
        None.

    Returns:
        None.

    Throws:
        None.

    Side Effects:
        None.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        *,
        pg_dsn: Optional[str] = None,
        table_name: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        get_batch_limit: int = 50000,
        add_batch_limit: int = 20000,
        **kwargs: Any,
    ) -> None:
        """Compose a SentenceTransformer with a PostgreSQL-backed embedding cache.

        Args:
            model_name_or_path: Passed to `SentenceTransformer(...)`; also used to
                derive the cache table name if `table_name` is not provided.
            pg_dsn: psycopg2 DSN; if omitted, a DSN is constructed from environment
                variables (optionally loaded from a local `.env`).
            table_name: Explicit table name for cache storage; if omitted, derived
                from model name plus optional `truncate_dim`.
            truncate_dim: Passed to SentenceTransformer to truncate embedding size;
                also included in the derived table name to avoid dimension mismatch.
            get_batch_limit: Max ids per SELECT batch when reading from cache.
            add_batch_limit: Max rows per INSERT batch when writing to cache.
            **kwargs: Additional keyword args forwarded to `SentenceTransformer`.

        Returns:
            None.

        Throws:
            ValueError: If `pg_dsn` is missing and required env vars are not set.
            psycopg2.Error: If connecting to PostgreSQL or creating the table fails.

        Side Effects:
            May read environment variables and `.env`; opens a PostgreSQL connection
            and may create a table; instantiates a SentenceTransformer model (which
            may download weights and allocate CPU/GPU memory).
        """
        self._st = SentenceTransformer(model_name_or_path, truncate_dim=truncate_dim, **kwargs)

        logger = logging.getLogger(__name__)
        model_key = model_name_or_path or getattr(self._st, "name_or_path", "model")
        self._model_key = str(model_key)
        self._get_batch_limit = int(get_batch_limit)
        self._add_batch_limit = int(add_batch_limit)
        self._truncate_dim = int(truncate_dim) if truncate_dim is not None else None

        table_suffix = f"__dim{self._truncate_dim}" if self._truncate_dim is not None else ""
        if not pg_dsn:
            try:
                load_dotenv(dotenv_path=".env", override=False)
            except Exception as exc:
                logger.warning("Failed to load .env via python-dotenv: %s", exc)

            host = os.environ.get("PSQL_HOST_NAME")
            port = os.environ.get("PSQL_PORT")
            db = os.environ.get("PSQL_DBNAME")
            user = os.environ.get("PSQL_USER")
            pwd = os.environ.get("PSQL_PASSWORD")

            missing = [
                key
                for key, val in (
                    ("PSQL_HOST_NAME", host),
                    ("PSQL_PORT", port),
                    ("PSQL_DBNAME", db),
                    ("PSQL_USER", user),
                    ("PSQL_PASSWORD", pwd),
                )
                if val in (None, "")
            ]
            if not missing:
                pg_dsn = f"host={host} port={port} dbname={db} user={user} password={pwd}"
            else:
                logger.error(
                    "pg_dsn is None and required env vars are missing: %s. "
                    "Expected environment variables: PSQL_HOST_NAME, PSQL_PORT, PSQL_DBNAME, PSQL_USER, PSQL_PASSWORD. "
                    "Set them in your environment or in a .env file.",
                    ", ".join(missing),
                )
                raise ValueError(
                    "pg_dsn is required. Provide 'pg_dsn' directly or set env vars "
                    "PSQL_HOST_NAME, PSQL_PORT, PSQL_DBNAME, PSQL_USER, PSQL_PASSWORD in the environment or .env"
                )

        tbl = table_name or sanitize_identifier(f"st_cache__{self._model_key}{table_suffix}")
        self.table_name = tbl
        self._store = PostgresKVStore(pg_dsn, tbl)
        logger.info(
            "CachedSentenceTransformer: table=%s truncate_dim=%s",
            tbl,
            self._truncate_dim,
        )

    def encode(
        self,
        sentences: Union[str, List[str]],
        *,
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
        normalize_embeddings: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[List[float], np.ndarray, torch.Tensor]:
        """Encode sentences while caching embeddings in PostgreSQL by stable id.

        Args:
            sentences: A single string or a list of strings to embed.
            batch_size: Batch size used for computing missing embeddings.
            show_progress_bar: If True, display tqdm progress bars for cache I/O
                and for model inference (delegated to SentenceTransformer).
            output_value: Forwarded to SentenceTransformer `encode`; typically
                "sentence_embedding".
            convert_to_numpy: If True, return a numpy array of shape (N, D).
            convert_to_tensor: If True, return a torch tensor of shape (N, D).
            device: Optional device override passed to SentenceTransformer and used
                as the output tensor device when `convert_to_tensor=True`.
            normalize_embeddings: Forwarded to SentenceTransformer; included in the
                cache key to avoid mixing normalized and unnormalized vectors.
            **kwargs: Additional keyword args forwarded to SentenceTransformer `encode`.

        Returns:
            If `convert_to_tensor` is True, a `torch.Tensor` of embeddings.
            Else if `convert_to_numpy` is True, a `np.ndarray` of embeddings.
            Else, an empty list placeholder (API parity with SentenceTransformer).

        Throws:
            ValueError: If cached embedding blobs have invalid length or DSN/env is invalid.
            psycopg2.Error: If cache reads/writes fail.
            RuntimeError: If SentenceTransformer inference fails.

        Side Effects:
            Executes SELECT/INSERT statements against PostgreSQL; may run model
            inference and allocate CPU/GPU memory; may log cache hit/miss stats.
        """
        single_input = isinstance(sentences, str)
        texts: List[str] = [sentences] if single_input else list(sentences)
        if len(texts) == 0:
            return (
                torch.empty((0, 0), device=device)
                if convert_to_tensor
                else np.empty((0, 0), dtype=np.float32)
            )

        logger = logging.getLogger(__name__)
        ids: List[str] = [stable_id(self._model_key, t, normalize_embeddings) for t in texts]

        unique_ids: List[str] = list(dict.fromkeys(ids))
        logger.debug("PG Cache GET: %d unique ids (batch=%d)", len(unique_ids), self._get_batch_limit)
        cached_bytes: Dict[str, bytes] = self._store.fetch_many(
            unique_ids, batch_size=self._get_batch_limit, show_pbar=bool(show_progress_bar)
        )

        missing_positions: List[int] = []
        missing_ids: List[str] = []
        missing_texts: List[str] = []

        missing_id_set = set(unique_ids) - set(cached_bytes.keys())
        missing_seen: set[str] = set()

        for idx, rec_id in enumerate(ids):
            if rec_id in missing_id_set and rec_id not in missing_seen:
                missing_positions.append(idx)
                missing_ids.append(rec_id)
                missing_texts.append(texts[idx])
                missing_seen.add(rec_id)

        hits_total = len(ids) - len(missing_positions)
        misses_total = len(missing_positions)
        logger.info("PG Cache stats: hits=%d misses=%d total=%d", hits_total, misses_total, len(ids))

        if misses_total > 0:
            logger.info("Computing embeddings for %d missing ids", misses_total)
            computed = self._st.encode(
                missing_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                output_value=output_value,
                convert_to_numpy=True,
                convert_to_tensor=False,
                device=device,
                normalize_embeddings=normalize_embeddings,
                **kwargs,
            )
            if isinstance(computed, np.ndarray):
                comp = computed.astype(np.float32, copy=False)
            else:
                comp = np.asarray(computed, dtype=np.float32)
            id_to_bytes: Dict[str, bytes] = {rec_id: vector_to_bytes(comp[i]) for i, rec_id in enumerate(missing_ids)}
            self._store.insert_many(
                id_to_bytes, batch_size=self._add_batch_limit, show_pbar=bool(show_progress_bar)
            )
            cached_bytes.update(id_to_bytes)

        if convert_to_tensor or convert_to_numpy:
            vectors: List[np.ndarray] = [bytes_to_vector(cached_bytes[rec_id]) for rec_id in ids]
            arr = np.vstack(vectors)
            if convert_to_tensor:
                underlying_device = getattr(self._st, "device", None)
                out_device = device or (underlying_device if underlying_device is not None else "cpu")
                return torch.tensor(arr, device=out_device)
            return arr
        return []

    def close(self) -> None:
        """Close the backing PostgreSQL cache connection.

        Args:
            None.

        Returns:
            None.

        Throws:
            None.

        Side Effects:
            Closes the underlying PostgreSQL connection.
        """
        self._store.close()


