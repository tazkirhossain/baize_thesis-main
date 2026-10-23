"""Shared source taxonomy and strict binary-label validation."""
import numpy as np

SOURCE_COLUMNS = ["id", "url", "label"]
POSITIVE_CATEGORY = "adult"
KNOWN_CATEGORIES = frozenset({
    "adult", "arts", "business", "computers", "games", "health", "home",
    "kids", "news", "recreation", "reference", "science", "shopping",
    "society", "sports",
})


def binary_label(value) -> int:
    """Validate before narrowing to int8; never truncate fractions or overflow."""
    if isinstance(value, str):
        if value.strip() not in ("0", "1"):
            raise ValueError("label must be exactly 0 or 1")
        return int(value.strip())
    if not np.isscalar(value) or value not in (0, 1):
        raise ValueError("label must be exactly 0 or 1")
    return int(value)
