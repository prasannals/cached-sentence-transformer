"""Integration tests for the PostgreSQL-backed cache.

These tests are skipped by default and require a running PostgreSQL instance.
They stub out SentenceTransformer to avoid model downloads and heavy inference.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import psycopg2
import pytest

import cached_sentence_transformer.cache as cst_cache


def _should_run_integration() -> bool:
    """Decide whether integration tests should run based on environment.

    Args:
        None.

    Returns:
        True if RUN_PG_INTEGRATION is set to \"1\"; otherwise False.

    Throws:
        None.

    Side Effects:
        Reads environment variables.
    """
    return os.environ.get("RUN_PG_INTEGRATION") == "1"


def _require_pg_dsn() -> str:
    """Fetch the PostgreSQL DSN from the environment and fail fast if missing.

    Args:
        None.

    Returns:
        The DSN string from the PG_DSN environment variable.

    Throws:
        RuntimeError: If PG_DSN is not set.

    Side Effects:
        Reads environment variables.
    """
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        raise RuntimeError("PG_DSN must be set for integration tests (e.g., 'host=... port=... dbname=... user=... password=...').")
    return dsn


@dataclass
class _DummySentenceTransformer:
    """Stub replacement for SentenceTransformer for integration tests.

    Args:
        name_or_path: Stored as `name_or_path` to emulate SentenceTransformer.
        truncate_dim: Ignored for this stub; included for signature compatibility.
        **kwargs: Ignored.

    Returns:
        None.

    Throws:
        None.

    Side Effects:
        None.
    """

    name_or_path: Optional[str] = None
    truncate_dim: Optional[int] = None
    encode_calls: int = 0
    last_sentences: List[str] | None = None

    def __init__(self, model_name_or_path: Optional[str] = None, *, truncate_dim: Optional[int] = None, **kwargs: Any) -> None:
        """Create a dummy model compatible with the wrapper's initialization call.

        Args:
            model_name_or_path: Model identifier (stored only for cache table naming).
            truncate_dim: Optional truncation dim (stored for completeness).
            **kwargs: Ignored.

        Returns:
            None.

        Throws:
            None.

        Side Effects:
            None.
        """
        self.name_or_path = model_name_or_path
        self.truncate_dim = truncate_dim
        self.encode_calls = 0
        self.last_sentences = None

    @property
    def device(self) -> str:
        """Return the dummy device string.

        Args:
            None.

        Returns:
            A constant device string, \"cpu\".

        Throws:
            None.

        Side Effects:
            None.
        """
        return "cpu"

    def encode(
        self,
        sentences: List[str],
        *,
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
        normalize_embeddings: Optional[bool] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return deterministic fake embeddings for the provided sentences.

        Args:
            sentences: List of sentences to encode.
            batch_size: Ignored.
            show_progress_bar: Ignored.
            output_value: Ignored.
            convert_to_numpy: Must be True for this stub.
            convert_to_tensor: Must be False for this stub.
            device: Ignored.
            normalize_embeddings: Ignored by the stub (cache key handling is tested in wrapper).
            **kwargs: Ignored.

        Returns:
            A numpy array of shape (N, 3) with deterministic float32 values.

        Throws:
            ValueError: If conversion flags request a tensor output.

        Side Effects:
            Increments `encode_calls`.
        """
        if convert_to_tensor or not convert_to_numpy:
            raise ValueError("DummySentenceTransformer only supports convert_to_numpy=True, convert_to_tensor=False")
        self.encode_calls += 1
        self.last_sentences = list(sentences)
        out = np.zeros((len(sentences), 3), dtype=np.float32)
        for i, s in enumerate(sentences):
            out[i, 0] = float(len(s))
            out[i, 1] = float(sum(ord(ch) for ch in s) % 997)
            out[i, 2] = float(i)
        return out


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="Set RUN_PG_INTEGRATION=1 to run Postgres integration tests.")
def test_encode_uses_cache_on_second_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second call with same inputs should hit cache and avoid recomputation.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Throws:
        AssertionError: If the wrapper recomputes embeddings on a cache hit.

    Side Effects:
        Creates and drops a temporary PostgreSQL table; performs cache reads/writes.
    """
    dsn = _require_pg_dsn()
    monkeypatch.setattr(cst_cache, "SentenceTransformer", _DummySentenceTransformer)

    table_name = f"cst_test_{uuid.uuid4().hex}"
    model = cst_cache.CachedSentenceTransformer(model_name_or_path="dummy-model", pg_dsn=dsn, table_name=table_name)
    try:
        # Include a duplicate on purpose: the wrapper should only compute/store unique ids.
        texts = ["alpha", "alpha", "beta"]

        out1 = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        assert isinstance(out1, np.ndarray)
        assert model._st.encode_calls == 1
        assert model._st.last_sentences == ["alpha", "beta"]

        with psycopg2.connect(dsn) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                (row_count_after_first,) = cur.fetchone()
        assert row_count_after_first == 2

        out2 = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        assert isinstance(out2, np.ndarray)
        assert model._st.encode_calls == 1
        assert model._st.last_sentences == ["alpha", "beta"]
        np.testing.assert_allclose(out2, out1, rtol=0, atol=0)

        # Flipping normalize_embeddings should produce different cache keys and trigger recompute+insert.
        out3 = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        assert isinstance(out3, np.ndarray)
        assert model._st.encode_calls == 2
        assert model._st.last_sentences == ["alpha", "beta"]

        with psycopg2.connect(dsn) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                (row_count_after_third,) = cur.fetchone()
        assert row_count_after_third == 4
    finally:
        model.close()
        with psycopg2.connect(dsn) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')


