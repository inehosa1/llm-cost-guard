"""The guard: runs the layers in order and only calls the paid model when it has to."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from .cache import SemanticCache
from .types import Allow, Answer, Decision, Generate, Layer, Route

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GuardStats:
    """Running totals, to see how much the guard saves."""

    routes: Counter[Route] = field(default_factory=Counter)
    model_cost_usd: float = 0.0

    @property
    def total(self) -> int:
        """Questions answered."""
        return sum(self.routes.values())

    @property
    def free_ratio(self) -> float:
        """Share of questions answered without the paid model (0.0 when there are none)."""
        return 1 - self.routes[Route.MODEL] / self.total if self.total else 0.0


class Guard:
    """Answers a question through the first layer that can, the cache, or the model.

    Order matters: put the cheapest and most certain layers first (prompt injection,
    rules, small talk) and the classifier last.
    """

    def __init__(self, layers: Sequence[Layer], *, cache: SemanticCache | None = None) -> None:
        names = [layer.name for layer in layers]
        if len(names) != len(set(names)):
            raise ValueError(f"layer names must be unique: {names}")
        self._layers = tuple(layers)
        self._cache = cache
        self.stats = GuardStats()

    def ask(self, question: str, generate: Generate) -> Answer:
        """Answer ``question``. ``generate`` is only called when nothing cheaper answered."""
        if not question.strip():
            raise ValueError("question must not be empty")

        allow_reason = ""
        for layer in self._layers:
            verdict = layer.check(question)
            if isinstance(verdict, Decision):
                return self._record(
                    Answer(verdict.answer, verdict.route, layer.name, verdict.reason)
                )
            if isinstance(verdict, Allow):
                allow_reason = verdict.reason
                break

        if self._cache is not None:
            cached = self._cache.lookup(question)
            if cached is not None:
                return self._record(Answer(cached, Route.CACHE, "cache", "similar question"))

        completion = generate(question)
        if self._cache is not None:
            self._cache.store(question, completion.text)
        return self._record(
            Answer(completion.text, Route.MODEL, "model", allow_reason, completion.cost_usd)
        )

    def _record(self, answer: Answer) -> Answer:
        self.stats.routes[answer.route] += 1
        self.stats.model_cost_usd += answer.cost_usd
        logger.debug("answered via %s (%s): %s", answer.route, answer.layer, answer.reason)
        return answer
