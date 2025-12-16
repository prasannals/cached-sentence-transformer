"""Unit test: vectors bytes round-trip and fail-fast validation."""

from __future__ import annotations

import numpy as np
import pytest

from cached_sentence_transformer.vectors import bytes_to_vector, vector_to_bytes


def test_vector_bytes_round_trip_and_invalid_length_fails_fast() -> None:
    """Round-trip vector serialization should preserve float32 content and validate blob size.

    Args:
        None.

    Returns:
        None.

    Throws:
        AssertionError: If round-trip values differ from expected float32 representation.
        ValueError: Expected for invalid blob lengths that are not a multiple of 4 bytes.

    Side Effects:
        Allocates small numpy arrays.
    """
    vec = np.asarray([1.0, 2.0, 3.5], dtype=np.float32)
    blob = vector_to_bytes(vec)
    out = bytes_to_vector(blob)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, vec, rtol=0, atol=0)

    with pytest.raises(ValueError):
        bytes_to_vector(b"\x00")


