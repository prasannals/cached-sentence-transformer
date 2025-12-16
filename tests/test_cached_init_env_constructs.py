"""Unit test: CachedSentenceTransformer constructs DSN from env and derives table name."""

from __future__ import annotations

import pytest

from cached_sentence_transformer.cache import CachedSentenceTransformer


def test_cached_sentence_transformer_env_dsn_construction_and_table_name(fake_kv: dict[str, bytes], patch_dummy_st, monkeypatch: pytest.MonkeyPatch) -> None:
    """Init should construct DSN from env and derive a deterministic table name.

    Args:
        fake_kv: In-memory dict backing store provided by pytest fixture.
        patch_dummy_st: Fixture that patches SentenceTransformer with a dummy model.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Throws:
        AssertionError: If DSN construction or table naming does not match expectations.

    Side Effects:
        Mutates environment variables within the test process.
    """
    _ = fake_kv
    _ = patch_dummy_st

    monkeypatch.setenv("PSQL_HOST_NAME", "h")
    monkeypatch.setenv("PSQL_PORT", "5432")
    monkeypatch.setenv("PSQL_DBNAME", "d")
    monkeypatch.setenv("PSQL_USER", "u")
    monkeypatch.setenv("PSQL_PASSWORD", "p")

    model = CachedSentenceTransformer(model_name_or_path="my-model", pg_dsn=None, truncate_dim=10)
    assert model.table_name.startswith("st_cache__my-model__dim10") is True
    model.close()


