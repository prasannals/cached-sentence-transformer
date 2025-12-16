"""Unit test: CachedSentenceTransformer fails fast on corrupt cached blobs."""

from __future__ import annotations

import pytest

from cached_sentence_transformer.cache import CachedSentenceTransformer
from cached_sentence_transformer.hashing import stable_id


def test_encode_fails_fast_on_corrupt_cached_blob(fake_kv: dict[str, bytes], patch_dummy_st) -> None:
    """A corrupt cached blob should raise ValueError when decoding is attempted.

    Args:
        fake_kv: In-memory dict backing store provided by pytest fixture.
        patch_dummy_st: Fixture that patches SentenceTransformer with a dummy model.

    Returns:
        None.

    Throws:
        ValueError: Expected when a cached blob has invalid length for float32 decoding.

    Side Effects:
        Inserts a corrupt blob into the in-memory store.
    """
    _ = patch_dummy_st

    model = CachedSentenceTransformer(model_name_or_path="m", pg_dsn="dsn")
    bad_id = stable_id("m", "boom", True)
    fake_kv[bad_id] = b"\x00"

    with pytest.raises(ValueError):
        model.encode(["boom"], normalize_embeddings=True, convert_to_numpy=True, convert_to_tensor=False)
    model.close()


