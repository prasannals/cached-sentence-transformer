"""Unit test: PostgresKVStore using a fake Postgres layer."""

from __future__ import annotations

from cached_sentence_transformer.postgres_kv_store import PostgresKVStore


def test_postgres_kv_store_insert_and_fetch_and_batch_validation(fake_kv: dict[str, bytes]) -> None:
    """PostgresKVStore should store/retrieve bytes and fail fast on invalid batch sizes.

    Args:
        fake_kv: In-memory dict backing store provided by pytest fixture.

    Returns:
        None.

    Throws:
        AssertionError: If inserted values cannot be fetched back correctly.
        ValueError: Expected for invalid batch sizes.

    Side Effects:
        Populates the in-memory store and opens/closes a fake connection.
    """
    store = PostgresKVStore("dsn", "tbl")

    store.insert_many({"a": b"\x00\x00\x80?", "b": b"\x00\x00\x00@"}, batch_size=10)
    got = store.fetch_many(["a", "b", "c"], batch_size=2)
    assert got["a"] == b"\x00\x00\x80?"
    assert got["b"] == b"\x00\x00\x00@"
    assert "c" not in got

    try:
        store.fetch_many(["a"], batch_size=0)
        raise AssertionError("Expected ValueError for batch_size=0")
    except ValueError:
        pass

    try:
        store.insert_many({"a": b"x"}, batch_size=-1)
        raise AssertionError("Expected ValueError for batch_size=-1")
    except ValueError:
        pass

    store.close()
    assert store._conn.closed is True
    assert len(fake_kv) == 2


