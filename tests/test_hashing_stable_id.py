"""Unit test: hashing.stable_id."""

from __future__ import annotations

from cached_sentence_transformer.hashing import stable_id


def test_stable_id_is_deterministic_and_sensitive_to_normalize_flag() -> None:
    """Ensure stable ids are deterministic and include the normalize flag in the key.

    Args:
        None.

    Returns:
        None.

    Throws:
        AssertionError: If ids are not deterministic or do not vary with normalize flag changes.

    Side Effects:
        None.
    """
    a1 = stable_id("m", "hello", True)
    a2 = stable_id("m", "hello", True)
    b = stable_id("m", "hello", False)
    assert a1 == a2
    assert a1 != b


