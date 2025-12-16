"""Stable hashing utilities for cache keys.

This module centralizes the cache key computation to keep it consistent across
call sites.
"""

from __future__ import annotations

import hashlib
from typing import Optional


def stable_id(model_key: str, text: str, normalize: Optional[bool]) -> str:
    """Compute a stable cache key for (model, normalize flag, text).

    Args:
        model_key: Stable identifier for the embedding model (e.g., name or path).
        text: Input text to embed.
        normalize: Whether embeddings are normalized; included in the cache key.

    Returns:
        A SHA1 hex digest string used as the cache primary key.

    Throws:
        UnicodeEncodeError: If UTF-8 encoding of the payload fails (unexpected).

    Side Effects:
        None.
    """
    hasher = hashlib.sha1()
    if normalize is True:
        norm_flag = "T"
    elif normalize is False:
        norm_flag = "F"
    else:
        norm_flag = "N"
    payload = f"model={model_key}||norm={norm_flag}||text={text}".encode("utf-8")
    hasher.update(payload)
    return hasher.hexdigest()


