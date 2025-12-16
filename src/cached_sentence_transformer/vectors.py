"""Vector serialization helpers for PostgreSQL storage.

This module provides functions to encode float32 embeddings as bytes and decode
them back into numpy arrays.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np


def vector_to_bytes(vec: Union[np.ndarray, Sequence[float]]) -> bytes:
    """Convert an embedding vector to a compact float32 byte representation.

    Args:
        vec: A 1D embedding vector as a numpy array or a float sequence.

    Returns:
        Raw bytes representing the vector as contiguous float32 values.

    Throws:
        ValueError: If `vec` cannot be converted to a numeric float32 array.

    Side Effects:
        Allocates a numpy array view/copy during conversion.
    """
    if isinstance(vec, np.ndarray):
        arr = vec.astype(np.float32, copy=False)
    else:
        arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes(order="C")


def bytes_to_vector(blob: bytes) -> np.ndarray:
    """Decode float32 bytes from PostgreSQL into a 1D numpy array view.

    Args:
        blob: Raw bytes previously produced by `vector_to_bytes`.

    Returns:
        A 1D `np.ndarray` view over `blob` with dtype float32.

    Throws:
        ValueError: If `blob` length is not a multiple of 4 bytes (float32 size).

    Side Effects:
        The returned array shares memory with `blob` (no copy).
    """
    if (len(blob) % 4) != 0:
        raise ValueError(
            f"Invalid embedding blob length: {len(blob)} bytes; expected a multiple of 4 (float32)."
        )
    return np.frombuffer(blob, dtype=np.float32)


