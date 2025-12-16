"""Unit test: CachedSentenceTransformer encode empty inputs and no-output mode."""

from __future__ import annotations

import numpy as np

from cached_sentence_transformer.cache import CachedSentenceTransformer


def test_encode_empty_and_no_output_requested(fake_kv: dict[str, bytes], patch_dummy_st) -> None:
    """encode() should handle empty inputs and the no-output configuration.

    Args:
        fake_kv: In-memory dict backing store provided by pytest fixture.
        patch_dummy_st: Fixture that patches SentenceTransformer with a dummy model.

    Returns:
        None.

    Throws:
        AssertionError: If empty handling or no-output handling differs from contract.

    Side Effects:
        Populates the in-memory store via PostgresKVStore.
    """
    _ = patch_dummy_st

    model = CachedSentenceTransformer(model_name_or_path="m", pg_dsn="dsn")

    out_empty = model.encode([], convert_to_numpy=True, convert_to_tensor=False)
    assert isinstance(out_empty, np.ndarray)
    assert out_empty.shape == (0, 0)

    out_none = model.encode(["x"], convert_to_numpy=False, convert_to_tensor=False)
    assert out_none == []
    assert len(fake_kv) == 1
    model.close()


