"""Shared pytest fixtures for cached-sentence-transformer unit tests.

These fixtures provide an in-memory fake Postgres layer so we can exercise the
cache logic without a running database.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest


class FakeSQL:
    """Minimal stand-in for psycopg2.sql used for unit tests.

    Args:
        None.

    Returns:
        None.

    Throws:
        None.

    Side Effects:
        None.
    """

    class _SQLStr:
        """Simple SQL object supporting `.format(...).as_string(cur)` chaining.

        Args:
            s: Underlying SQL template string.

        Returns:
            None.

        Throws:
            None.

        Side Effects:
            None.
        """

        def __init__(self, s: str) -> None:
            self._s = s

        def format(self, **kwargs: Any) -> FakeSQL._SQLStr:
            out = self._s
            for k, v in kwargs.items():
                out = out.replace("{" + k + "}", str(v))
            return FakeSQL._SQLStr(out)

        def as_string(self, cur: Any) -> str:  # noqa: ARG002
            return self._s

        def __str__(self) -> str:
            return self._s

    class Identifier:
        """Identifier wrapper used by FakeSQL formatting.

        Args:
            name: Identifier string.

        Returns:
            None.

        Throws:
            None.

        Side Effects:
            None.
        """

        def __init__(self, name: str) -> None:
            self._name = name

        def __str__(self) -> str:
            return f'"{self._name}"'

    @staticmethod
    def SQL(s: str) -> FakeSQL._SQLStr:
        """Create a fake SQL object.

        Args:
            s: SQL template string.

        Returns:
            A fake SQL object supporting `.format()` and `.as_string()`.

        Throws:
            None.

        Side Effects:
            None.
        """
        return FakeSQL._SQLStr(s)


class FakeCursor:
    """In-memory cursor supporting the subset used by PostgresKVStore.

    Args:
        conn: Parent connection.

    Returns:
        None.

    Throws:
        None.

    Side Effects:
        Stores the last executed query/params for debugging.
    """

    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn
        self._results: list[tuple[str, bytes]] = []
        self.last_query: str | None = None
        self.last_params: tuple[Any, ...] | None = None

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # noqa: ANN401
        return None

    def execute(self, query: Any, params: tuple[Any, ...] | None = None) -> None:
        q = str(query)
        self.last_query = q
        self.last_params = params

        if q.strip().upper().startswith("SELECT"):
            assert params is not None
            chunk = params[0]
            assert isinstance(chunk, list)
            out: list[tuple[str, bytes]] = []
            for k in chunk:
                if k in self._conn.kv:
                    out.append((k, self._conn.kv[k]))
            self._results = out
            return

        self._results = []

    def fetchall(self) -> list[tuple[str, bytes]]:
        return list(self._results)


class FakeConn:
    """In-memory connection that stores key-value bytes in a dict.

    Args:
        kv: Backing mapping used as the \"table\".

    Returns:
        None.

    Throws:
        None.

    Side Effects:
        None.
    """

    def __init__(self, kv: dict[str, bytes]) -> None:
        self.kv = kv
        self.autocommit = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.closed = True


class DummySentenceTransformer:
    """Tiny SentenceTransformer stub producing deterministic float32 embeddings.

    Args:
        model_name_or_path: Stored as `name_or_path` for table naming.
        truncate_dim: Ignored by the stub.
        **kwargs: Ignored.

    Returns:
        None.

    Throws:
        None.

    Side Effects:
        None.
    """

    def __init__(self, model_name_or_path: str | None = None, *, truncate_dim: int | None = None, **kwargs: Any) -> None:
        self.name_or_path = model_name_or_path
        self._device = "cpu"
        self.encode_calls = 0
        self.last_sentences: list[str] | None = None

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
        return self._device

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:  # noqa: ANN401
        """Return deterministic fake embeddings for the provided sentences.

        Args:
            sentences: List of sentences to encode.
            **kwargs: May include normalize_embeddings; used to vary output deterministically.

        Returns:
            A numpy array of shape (N, 4) with deterministic float32 values.

        Throws:
            None.

        Side Effects:
            Increments `encode_calls` and stores `last_sentences`.
        """
        self.encode_calls += 1
        self.last_sentences = list(sentences)
        out = np.zeros((len(sentences), 4), dtype=np.float32)
        for i, s in enumerate(sentences):
            out[i, 0] = float(len(s))
            out[i, 1] = float(sum(ord(ch) for ch in s) % 997)
            out[i, 2] = float(i)
            out[i, 3] = 1.0 if kwargs.get("normalize_embeddings") else 0.0
        return out


@pytest.fixture()
def fake_kv(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Patch pg_kv_store to use an in-memory Postgres implementation and return its store.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        A shared mapping used as the fake backing store.

    Throws:
        None.

    Side Effects:
        Monkeypatches `cached_sentence_transformer.postgres_kv_store` to use fakes.
    """
    import cached_sentence_transformer.postgres_kv_store as kv_mod

    kv: dict[str, bytes] = {}

    monkeypatch.setattr(kv_mod, "sql", FakeSQL)
    monkeypatch.setattr(kv_mod.psycopg2, "connect", lambda dsn: FakeConn(kv))  # noqa: ARG005
    monkeypatch.setattr(kv_mod.psycopg2, "Binary", lambda b: b)

    def _fake_execute_values(cur: FakeCursor, query: str, values: Iterable[tuple[str, bytes]], page_size: int) -> None:  # noqa: ARG001
        for k, v in values:
            if k not in kv:
                kv[k] = bytes(v)

    monkeypatch.setattr(kv_mod, "execute_values", _fake_execute_values)
    return kv


@pytest.fixture()
def patch_dummy_st(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch CachedSentenceTransformer to use the dummy SentenceTransformer stub.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Throws:
        None.

    Side Effects:
        Monkeypatches `cached_sentence_transformer.cache.SentenceTransformer`.
    """
    import cached_sentence_transformer.cache as cached_mod

    monkeypatch.setattr(cached_mod, "SentenceTransformer", DummySentenceTransformer)


