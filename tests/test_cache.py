from __future__ import annotations

import pytest

from llm_cost_guard import SemanticCache, cosine

from .conftest import bag_of_words


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_cosine() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError, match="different length"):
        cosine([1], [1, 2])


def test_returns_the_answer_of_an_equivalent_question() -> None:
    cache = SemanticCache(bag_of_words, threshold=0.9)
    cache.store("What documents do I need for an appraisal?", "ID and the deed.")
    assert cache.lookup("what documents do I need for an appraisal") == "ID and the deed."
    assert cache.lookup("How long does an appraisal take?") is None


def test_picks_the_most_similar_entry() -> None:
    cache = SemanticCache(bag_of_words, threshold=0.5)
    cache.store("appraisal price madrid", "Madrid price")
    cache.store("appraisal price madrid centre flat", "Centre flat price")
    assert cache.lookup("appraisal price madrid centre flat") == "Centre flat price"
    assert cache.lookup("appraisal price madrid") == "Madrid price"


def test_empty_cache_does_not_embed() -> None:
    def embed(_text: str) -> list[float]:
        raise AssertionError("should not be called")

    assert SemanticCache(embed).lookup("anything") is None


def test_entries_expire() -> None:
    clock = Clock()
    cache = SemanticCache(bag_of_words, ttl_seconds=60, clock=clock)
    cache.store("opening hours", "9 to 18")
    clock.now = 30
    assert cache.lookup("opening hours") == "9 to 18"
    clock.now = 61
    assert cache.lookup("opening hours") is None


def test_evicts_the_oldest_entry_when_full() -> None:
    cache = SemanticCache(bag_of_words, max_entries=2)
    for i in range(3):
        cache.store(f"question number {i}", f"answer {i}")
    assert len(cache) == 2
    assert cache.lookup("question number 0") is None
    assert cache.lookup("question number 2") == "answer 2"


def test_clear() -> None:
    cache = SemanticCache(bag_of_words)
    cache.store("a question", "an answer")
    cache.clear()
    assert len(cache) == 0


@pytest.mark.parametrize(
    ("options", "message"),
    [({"threshold": 0.0}, "threshold"), ({"max_entries": 0}, "max_entries")],
)
def test_validates_options(options: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SemanticCache(bag_of_words, **options)  # type: ignore[arg-type]
