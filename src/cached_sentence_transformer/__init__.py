"""Public API for the cached_sentence_transformer package.

This package provides a PostgreSQL-backed embedding cache wrapper for
SentenceTransformers models.
"""

from __future__ import annotations

__all__ = ["CachedSentenceTransformer"]

__version__ = "0.1.0"

# Public exports are defined in submodules to keep this namespace small.
from .cache import CachedSentenceTransformer  # noqa: E402


