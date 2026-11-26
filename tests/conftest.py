"""Shared test doubles: a deterministic embedder and a fake paid model."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from llm_cost_guard import Completion
from llm_cost_guard.text import normalize

DIMENSIONS = 64


def bag_of_words(text: str) -> Sequence[float]:
    """Hash each word into a fixed-size vector: same words, same vector, order ignored."""
    vector = [0.0] * DIMENSIONS
    for word in re.findall(r"\w+", normalize(text)):
        bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIMENSIONS
        vector[bucket] += 1.0
    return vector


@dataclass
class FakeModel:
    """Records every call; each answer costs ``price``."""

    price: float = 0.01
    calls: list[str] = field(default_factory=list)

    def __call__(self, question: str) -> Completion:
        self.calls.append(question)
        return Completion(text=f"model answer to: {question}", cost_usd=self.price)


@pytest.fixture
def model() -> FakeModel:
    return FakeModel()
