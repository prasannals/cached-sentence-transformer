"""Unit test: CachedSentenceTransformer encode miss->hit caching and tensor output."""

from __future__ import annotations

import numpy as np
import torch

from cached_sentence_transformer.cache import CachedSentenceTransformer


def test_encode_miss_then_hit_and_tensor_output(fake_kv: dict[str, bytes], patch_dummy_st) -> None:
    """encode() should compute on miss, store bytes, and reuse cache on hit.

    Args:
        fake_kv: In-memory dict backing store provided by pytest fixture.
        patch_dummy_st: Fixture that patches SentenceTransformer with a dummy model.

    Returns:
        None.

    Throws:
        AssertionError: If caching behavior or tensor output behavior is incorrect.

    Side Effects:
        Populates the in-memory store via PostgresKVStore.
    """
    _ = fake_kv
    _ = patch_dummy_st

    model = CachedSentenceTransformer(model_name_or_path="m", pg_dsn="dsn")
    texts = ["alpha", "alpha", "beta"]

    arr1 = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, convert_to_tensor=False)
    assert isinstance(arr1, np.ndarray)
    assert model._st.encode_calls == 1
    assert model._st.last_sentences == ["alpha", "beta"]

    arr2 = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, convert_to_tensor=False)
    assert model._st.encode_calls == 1
    np.testing.assert_allclose(arr2, arr1, rtol=0, atol=0)

    ten = model.encode(texts, normalize_embeddings=True, convert_to_numpy=False, convert_to_tensor=True, device="cpu")
    assert isinstance(ten, torch.Tensor)
    assert ten.device.type == "cpu"
    assert ten.shape == torch.Size([3, 4])

    model.close()


