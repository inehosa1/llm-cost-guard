"""Answer what you can for free before calling a paid LLM.

A :class:`Guard` runs cheap, local layers (rules, small talk, prompt-injection checks, a
topic classifier) and a semantic cache before it calls the model, and tells you which
route every answer took.
"""

from .cache import SemanticCache, cosine
from .guard import Guard, GuardStats
from .layers import (
    Action,
    Classification,
    PromptInjection,
    Rule,
    RuleSet,
    SmallTalk,
    TopicGate,
)
from .types import ALLOW, Allow, Answer, Completion, Decision, Layer, Route

__all__ = [
    "ALLOW",
    "Action",
    "Allow",
    "Answer",
    "Classification",
    "Completion",
    "Decision",
    "Guard",
    "GuardStats",
    "Layer",
    "PromptInjection",
    "Route",
    "Rule",
    "RuleSet",
    "SemanticCache",
    "SmallTalk",
    "TopicGate",
    "cosine",
]
__version__ = "0.1.0"
