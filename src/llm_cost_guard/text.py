"""Text normalization shared by the layers."""

from __future__ import annotations

import re
import unicodedata

_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Casefold, strip accents and collapse whitespace.

    >>> normalize("  ¿Cuánto   CUESTA? ")
    '¿cuanto cuesta?'
    """
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _SPACES.sub(" ", without_marks).strip().casefold()


def word_count(text: str) -> int:
    """Count words (runs of letters or digits)."""
    return len(re.findall(r"\w+", text))
