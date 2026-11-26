"""Value types shared by the guard, its layers and the cache."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol


class Route(StrEnum):
    """Where an answer came from. Every route except ``MODEL`` costs nothing."""

    BLOCKED = "blocked"
    """A deny rule or a prompt-injection attempt."""
    RULE = "rule"
    """A canned answer defined by a rule."""
    SMALL_TALK = "small_talk"
    """A greeting, thanks or goodbye."""
    OFF_TOPIC = "off_topic"
    """The classifier placed the question outside the allowed topics."""
    CACHE = "cache"
    """A semantically equivalent question was answered before."""
    MODEL = "model"
    """The paid model generated the answer."""


@dataclass(frozen=True, slots=True)
class Decision:
    """A layer's verdict: answer now, through ``route``, without calling the model."""

    route: Route
    answer: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Allow:
    """A layer's verdict: skip the remaining layers and go to the cache and the model."""

    reason: str = ""


ALLOW: Final = Allow()
"""Default :class:`Allow` verdict."""

Verdict = Decision | Allow | None
"""What a layer returns. ``None`` means "no opinion, ask the next layer"."""


class Layer(Protocol):
    """A cheap check that runs before the model. Layers must not raise for normal input."""

    @property
    def name(self) -> str:
        """Short identifier used in stats and logs."""
        ...

    def check(self, question: str) -> Verdict:
        """Return a verdict for ``question``."""
        ...


@dataclass(frozen=True, slots=True)
class Completion:
    """What the paid model returned, with what it cost."""

    text: str
    cost_usd: float = 0.0


Generate = Callable[[str], Completion]
"""Calls the paid model. The guard only invokes it when no layer or cache entry answered."""


@dataclass(frozen=True, slots=True)
class Answer:
    """The guard's result for one question."""

    text: str
    route: Route
    layer: str
    """Name of the layer that answered, ``"cache"`` or ``"model"``."""
    reason: str = ""
    cost_usd: float = 0.0

    @property
    def free(self) -> bool:
        """True when no paid model was called."""
        return self.route is not Route.MODEL
