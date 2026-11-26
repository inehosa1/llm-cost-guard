"""Built-in layers. Each one is cheap, local and answers a question or lets it through."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from .text import normalize, word_count
from .types import Allow, Decision, Route, Verdict

logger = logging.getLogger(__name__)


class Action(StrEnum):
    """What a :class:`Rule` does when it matches."""

    DENY = "deny"
    """Block the question with the rule's answer."""
    CANNED = "canned"
    """Answer with a fixed text (an FAQ entry) at no cost."""
    ALLOW = "allow"
    """Skip the remaining layers (e.g. the topic gate) and go to the model."""


@dataclass(frozen=True, slots=True)
class Rule:
    """A keyword or regex rule.

    Both run against the normalized question (casefolded, without accents), so a keyword
    rule for "Cancelación" matches "cancelacion". Keywords match whole words; write regex
    patterns in lowercase without accents.
    """

    pattern: str
    action: Action
    answer: str = ""
    regex: bool = False
    priority: int = 0
    """Higher priorities are evaluated first; ties keep declaration order."""
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.pattern.strip():
            raise ValueError("a rule needs a non-empty pattern")
        if self.action is not Action.ALLOW and not self.answer:
            raise ValueError(f"a {self.action} rule needs an answer")
        keyword = re.escape(normalize(self.pattern))
        source = self.pattern if self.regex else rf"(?<!\w){keyword}(?!\w)"
        object.__setattr__(self, "_compiled", re.compile(source))

    def matches(self, normalized_question: str) -> bool:
        """Whether the rule applies to an already normalized question."""
        return self._compiled.search(normalized_question) is not None


class RuleSet:
    """Project rules evaluated by priority. The first matching rule decides."""

    name = "rules"

    def __init__(self, rules: Iterable[Rule]) -> None:
        indexed = list(enumerate(rules))
        self._rules = [r for _, r in sorted(indexed, key=lambda ir: (-ir[1].priority, ir[0]))]

    def check(self, question: str) -> Verdict:
        """Apply the first matching rule, or return ``None`` if none matches."""
        q = normalize(question)
        for rule in self._rules:
            if not rule.matches(q):
                continue
            if rule.action is Action.ALLOW:
                return Allow(reason=f"allowed by rule {rule.pattern!r}")
            route = Route.BLOCKED if rule.action is Action.DENY else Route.RULE
            return Decision(route, rule.answer, reason=f"rule {rule.pattern!r}")
        return None


_SMALL_TALK: Mapping[str, str] = {
    r"^(hi|hello|hey|hola|buenas|buenos dias|buenas tardes|buenas noches)\b": "greeting",
    r"^(thanks|thank you|thx|gracias|muchas gracias|mil gracias)\b": "thanks",
    r"^(bye|goodbye|see you|adios|hasta luego|chao)\b": "goodbye",
}


class SmallTalk:
    """Answers greetings, thanks and goodbyes for free.

    Only short messages qualify, so "hello, what is the VAT rate?" still reaches the model.
    """

    name = "small_talk"

    def __init__(self, answers: Mapping[str, str], *, max_words: int = 4) -> None:
        """``answers`` maps a kind (``greeting``, ``thanks``, ``goodbye``) to its reply."""
        unknown = set(answers) - set(_SMALL_TALK.values())
        if unknown:
            raise ValueError(f"unknown small-talk kinds: {sorted(unknown)}")
        self._answers = dict(answers)
        self._max_words = max_words
        self._patterns = [(re.compile(p), kind) for p, kind in _SMALL_TALK.items()]

    def check(self, question: str) -> Verdict:
        """Reply to a short greeting, thanks or goodbye that has a configured answer."""
        if word_count(question) > self._max_words:
            return None
        q = normalize(question)
        for pattern, kind in self._patterns:
            if kind in self._answers and pattern.search(q):
                return Decision(Route.SMALL_TALK, self._answers[kind], reason=kind)
        return None


DEFAULT_INJECTION_PATTERNS: tuple[str, ...] = (
    r"\b(ignore|disregard|forget)\b.{0,20}\b(previous|prior|above|your|all)\b"
    r".{0,20}\b(instructions|rules|prompt)\b",
    r"\b(ignora|olvida)\b.{0,20}\b(tus|las|todas)\b.{0,20}\b(instrucciones|reglas)\b",
    r"\b(show|reveal|print|repeat)\b.{0,20}\b(system|hidden|initial)\s+prompt\b",
    # Spanish verbs often take an enclitic pronoun: "muéstrame", "revélame".
    r"\b(muestra|muestrame|revela|revelame|repite|repiteme)\b.{0,20}"
    r"\b(prompt|instrucciones)\s+(de sistema|del sistema|iniciales|ocultas)\b",
    r"\byou are now\b.{0,30}\b(unrestricted|jailbroken|dan)\b",
)


class PromptInjection:
    """Blocks obvious prompt-injection attempts before they reach the model.

    The patterns are deliberately narrow: a false positive blocks a real question, so this
    layer is a first filter, not a security boundary. Keep untrusted documents fenced in
    the prompt as well.
    """

    name = "prompt_injection"

    def __init__(self, answer: str, patterns: Iterable[str] = DEFAULT_INJECTION_PATTERNS) -> None:
        self._answer = answer
        self._patterns = [re.compile(p) for p in patterns]

    def check(self, question: str) -> Verdict:
        """Block the question if it matches an injection pattern."""
        q = normalize(question)
        for pattern in self._patterns:
            if pattern.search(q):
                return Decision(Route.BLOCKED, self._answer, reason="prompt injection")
        return None


@dataclass(frozen=True, slots=True)
class Classification:
    """A classifier's guess about the question's topic."""

    topic: str
    confidence: float


Classify = Callable[[str], Classification]
"""A cheap classifier: a small model, embeddings against topic examples, or heuristics."""


class TopicGate:
    """Turns away questions that a classifier confidently places outside the allowed topics."""

    name = "topic_gate"

    def __init__(
        self,
        classify: Classify,
        allowed: Collection[str],
        *,
        answer: str,
        min_confidence: float = 0.7,
        fail_open: bool = True,
    ) -> None:
        """``fail_open``: if the classifier errors, let the question through (True) or block it."""
        if not allowed:
            raise ValueError("a topic gate needs at least one allowed topic")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self._classify = classify
        self._allowed = frozenset(allowed)
        self._answer = answer
        self._min_confidence = min_confidence
        self._fail_open = fail_open

    def check(self, question: str) -> Verdict:
        """Turn the question away if it is confidently off topic."""
        try:
            result = self._classify(question)
        except Exception:
            logger.exception("topic classifier failed")
            if self._fail_open:
                return None
            return Decision(Route.OFF_TOPIC, self._answer, reason="classifier unavailable")
        if result.topic in self._allowed or result.confidence < self._min_confidence:
            return None
        return Decision(
            Route.OFF_TOPIC,
            self._answer,
            reason=f"topic {result.topic!r} ({result.confidence:.2f})",
        )


__all__ = [
    "Action",
    "Classification",
    "Classify",
    "PromptInjection",
    "Rule",
    "RuleSet",
    "SmallTalk",
    "TopicGate",
]
