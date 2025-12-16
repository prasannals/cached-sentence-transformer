"""PostgreSQL key-value store for embedding blobs.

This module implements a minimal schema and efficient batch read/write helpers
for storing embeddings as BYTEA by TEXT ids.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from tqdm import tqdm


class PostgresKVStore:
    """Minimal PostgreSQL key-value store for embedding blobs.

    Keys are stored as TEXT primary keys, and values are stored as BYTEA.

    Args:
        None.

    Returns:
        None.

    Throws:
        None.

    Side Effects:
        None.
    """

    def __init__(self, dsn: str, table_name: str) -> None:
        """Create a PostgreSQL-backed key-value store with schema auto-creation.

        Args:
            dsn: psycopg2 DSN string used to connect to PostgreSQL.
            table_name: Table to use for storage; will be created if missing.

        Returns:
            None.

        Throws:
            psycopg2.Error: If connection fails or schema creation fails.

        Side Effects:
            Opens a PostgreSQL connection and may create a table.
        """
        self._logger = logging.getLogger(__name__)
        self._dsn = dsn
        self._table = table_name
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure the backing table exists with the expected columns.

        Args:
            None.

        Returns:
            None.

        Throws:
            psycopg2.Error: If the CREATE TABLE statement fails.

        Side Effects:
            Executes DDL against PostgreSQL and may create a table.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        vec BYTEA NOT NULL
                    );
                    """
                ).format(table=sql.Identifier(self._table))
            )

    def fetch_many(self, ids: List[str], *, batch_size: int = 50000, show_pbar: bool = False) -> Dict[str, bytes]:
        """Fetch cached values for many ids from PostgreSQL in batches.

        Args:
            ids: List of cache ids to retrieve.
            batch_size: Max number of ids to fetch per query batch.
            show_pbar: If True, display a tqdm progress bar.

        Returns:
            A mapping from id -> raw BYTEA value for all ids found in the table.

        Throws:
            psycopg2.Error: If a SELECT query fails.
            ValueError: If `batch_size` is not a positive integer.

        Side Effects:
            Executes SELECT statements against PostgreSQL; may create a progress bar.
        """
        result: Dict[str, bytes] = {}
        if not ids:
            return result
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        pbar = tqdm(total=len(ids), desc="pg_get", unit="ids", leave=False) if show_pbar else None
        for start in range(0, len(ids), batch_size):
            chunk = ids[start : start + batch_size]
            with self._conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT id, vec FROM {table} WHERE id = ANY(%s)").format(
                        table=sql.Identifier(self._table)
                    ),
                    (chunk,),
                )
                for rec_id, vec in cur.fetchall():
                    result[str(rec_id)] = bytes(vec)
            if pbar is not None:
                pbar.update(len(chunk))
        if pbar is not None:
            pbar.close()
        return result

    def insert_many(self, id_to_vec: Dict[str, bytes], *, batch_size: int = 20000, show_pbar: bool = False) -> None:
        """Insert many (id, vec) records into PostgreSQL using batched upserts.

        Args:
            id_to_vec: Mapping from id -> raw float32 bytes (BYTEA) to store.
            batch_size: Max number of rows to insert per batch.
            show_pbar: If True, display a tqdm progress bar.

        Returns:
            None.

        Throws:
            psycopg2.Error: If an INSERT fails.
            ValueError: If `batch_size` is not a positive integer.

        Side Effects:
            Executes INSERT statements against PostgreSQL; may create a progress bar.
        """
        if not id_to_vec:
            return
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        items = list(id_to_vec.items())
        pbar = tqdm(total=len(items), desc="pg_add", unit="ids", leave=False) if show_pbar else None
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            values = [(k, psycopg2.Binary(v)) for k, v in chunk]
            with self._conn.cursor() as cur:
                execute_values(
                    cur,
                    sql.SQL(
                        """
                        INSERT INTO {table} (id, vec)
                        VALUES %s
                        ON CONFLICT (id) DO NOTHING
                        """
                    ).format(table=sql.Identifier(self._table)).as_string(cur),
                    values,
                    page_size=min(batch_size, 10000),
                )
            if pbar is not None:
                pbar.update(len(chunk))
        if pbar is not None:
            pbar.close()

    def close(self) -> None:
        """Close the underlying PostgreSQL connection.

        Args:
            None.

        Returns:
            None.

        Throws:
            None (exceptions during close are swallowed intentionally).

        Side Effects:
            Closes the PostgreSQL connection.
        """
        try:
            self._conn.close()
        except Exception:
            pass


