"""Unit test: CachedSentenceTransformer init fails fast when env DSN is incomplete."""

from __future__ import annotations

import pytest

from cached_sentence_transformer.cache import CachedSentenceTransformer


def test_cached_sentence_transformer_fails_fast_when_env_dsn_missing(fake_kv: dict[str, bytes], patch_dummy_st, monkeypatch: pytest.MonkeyPatch) -> None:
    """Init should raise ValueError if pg_dsn is None and required env vars are missing.

    Args:
        fake_kv: In-memory dict backing store provided by pytest fixture.
        patch_dummy_st: Fixture that patches SentenceTransformer with a dummy model.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Throws:
        ValueError: Expected when required env vars are missing.

    Side Effects:
        Mutates environment variables within the test process.
    """
    _ = fake_kv
    _ = patch_dummy_st

    for k in ["PSQL_HOST_NAME", "PSQL_PORT", "PSQL_DBNAME", "PSQL_USER", "PSQL_PASSWORD"]:
        monkeypatch.delenv(k, raising=False)

    with pytest.raises(ValueError):
        CachedSentenceTransformer(model_name_or_path="m", pg_dsn=None)


