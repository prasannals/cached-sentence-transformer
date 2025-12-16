"""PostgreSQL identifier utilities.

This module contains small helpers for preparing safe PostgreSQL identifiers.
"""

from __future__ import annotations


def sanitize_identifier(name: str) -> str:
    """Sanitize a string for safe use as a PostgreSQL identifier (table name).

    Args:
        name: Raw identifier candidate, potentially containing unsupported characters.

    Returns:
        A PostgreSQL-safe identifier (max 63 chars) where unsupported characters are
        replaced with underscores.

    Throws:
        None.

    Side Effects:
        None.
    """
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return ("".join(out))[:63]  # PostgreSQL identifier length limit


