"""Unit test: identifiers.sanitize_identifier."""

from __future__ import annotations

from cached_sentence_transformer.identifiers import sanitize_identifier


def test_sanitize_identifier_replaces_invalid_chars_and_truncates() -> None:
    """Validate identifier sanitization and PostgreSQL length truncation.

    Args:
        None.

    Returns:
        None.

    Throws:
        AssertionError: If sanitization does not replace unsupported characters or does not truncate to 63 chars.

    Side Effects:
        None.
    """
    assert sanitize_identifier("a b.c") == "a_b_c"
    assert len(sanitize_identifier("x" * 100)) == 63


