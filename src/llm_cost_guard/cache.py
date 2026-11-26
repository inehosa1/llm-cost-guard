"""A small in-memory semantic cache: equivalent questions reuse a previous answer."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Embed = Callable[[str], Sequence[float]]
"""Turns a question into a vector (a local embedding model in production)."""


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 when either vector is all zeros."""
    if len(a) != len(b):
        raise ValueError(f"vectors of different length: {len(a)} and {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass(frozen=True, slots=True)
class _Entry:
    vector: tuple[float, ...]
    answer: str
    stored_at: float


class SemanticCache:
    """Stores answers by question embedding and returns the closest one above a threshold.

    Linear scan on purpose: it is meant for thousands of entries per project. Past that,
    use a vector index (e.g. pgvector) behind the same two methods.
    """

    def __init__(
        self,
        embed: Embed,
        *,
        threshold: float = 0.92,
        max_entries: int = 1000,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._embed = embed
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: deque[_Entry] = deque(maxlen=max_entries)

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, question: str) -> str | None:
        """Return the answer of the most similar live entry above the threshold, if any."""
        if not self._entries:
            return None
        vector = tuple(self._embed(question))
        now = self._clock()
        best: tuple[float, str] | None = None
        for entry in self._entries:
            if self._ttl is not None and now - entry.stored_at > self._ttl:
                continue
            score = cosine(vector, entry.vector)
            if score >= self._threshold and (best is None or score > best[0]):
                best = (score, entry.answer)
        return best[1] if best else None

    def store(self, question: str, answer: str) -> None:
        """Remember an answer. The oldest entry is evicted when the cache is full."""
        self._entries.append(_Entry(tuple(self._embed(question)), answer, self._clock()))

    def clear(self) -> None:
        """Forget everything (e.g. after the source documents change)."""
        self._entries.clear()
