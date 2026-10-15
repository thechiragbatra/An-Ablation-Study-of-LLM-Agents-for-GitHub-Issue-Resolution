"""Detect a repository's stance on AI-generated contributions. Deliberately conservative:
any mention of AI blocks publishing unless a human has reviewed the policy, and an
explicit prohibition always blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

POLICY_PATHS = (
    "CONTRIBUTING.md",
    "CONTRIBUTING.rst",
    "CONTRIBUTING",
    ".github/CONTRIBUTING.md",
    "docs/CONTRIBUTING.md",
    "AI_POLICY.md",
    ".github/AI_POLICY.md",
    "CODE_OF_CONDUCT.md",
)

_AI_UPPER = re.compile(r"\bA\.?I\.?(?=\b|[-\s])")  # "AI", "A.I." (case-sensitive)
_AI_WORDS = re.compile(
    r"artificial intelligence|\bLLMs?\b|large language models?|machine[- ]generated|"
    r"\bgenerative\b|chat-?gpt|\bgpt-?\d|copilot|\bclaude\b|\bgemini\b|code[- ]generation tools?|"
    r"auto(?:matically)?[- ]generated (?:code|patch|pull|contribution)",
    re.IGNORECASE,
)
_PROHIBIT = re.compile(
    r"\b(?:not (?:be )?accept(?:ed)?|do(?:n't| not) accept|won't accept|will not (?:be )?accept|"
    r"not (?:be )?(?:allowed|permitted|welcome)|disallow|prohibit|forbid|forbidden|banned?|"
    r"reject|refuse|must not|may not|do(?:n't| not) (?:submit|use|open|send)|"
    r"will be closed|close (?:them|it|such)|no\b)",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n\s*\n|\n[-*] ")

Verdict = Literal["none", "mentioned", "prohibited"]


@dataclass(frozen=True)
class PolicyAssessment:
    verdict: Verdict
    evidence: str | None = None
    source: str | None = None


def _mentions_ai(sentence: str) -> bool:
    return bool(_AI_UPPER.search(sentence) or _AI_WORDS.search(sentence))


def assess_text(text: str, source: str = "") -> PolicyAssessment:
    mention: str | None = None
    for raw in _SENTENCE.split(text):
        sentence = " ".join(raw.split())
        if not sentence or not _mentions_ai(sentence):
            continue
        if _PROHIBIT.search(sentence):
            return PolicyAssessment("prohibited", sentence[:240], source)
        mention = mention or sentence[:240]
    if mention is not None:
        return PolicyAssessment("mentioned", mention, source)
    return PolicyAssessment("none")


def assess(documents: dict[str, str]) -> PolicyAssessment:
    """Most restrictive verdict across all policy documents found."""
    found = PolicyAssessment("none")
    for source, text in documents.items():
        a = assess_text(text, source)
        if a.verdict == "prohibited":
            return a
        if a.verdict == "mentioned" and found.verdict == "none":
            found = a
    return found
