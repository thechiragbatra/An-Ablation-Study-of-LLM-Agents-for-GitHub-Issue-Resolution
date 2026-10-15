"""What happens after a failed validation. The single source of truth for both the
validator (which sets ``running`` vs ``unverified``) and the graph router.

Termination: every graph cycle passes through ``architect``, which strictly increments
``attempt_count``; every back-edge requires ``attempt_count < config.max_rounds()``.
"""

from __future__ import annotations

from typing import Literal

from agent.config import ExperimentConfig

FailureAction = Literal["retry", "reflect", "stop"]


def next_after_failure(config: ExperimentConfig, attempt_count: int) -> FailureAction:
    if attempt_count >= config.max_rounds():
        return "stop"
    if config.variant == "A1":
        return "retry"
    if config.variant == "A3":
        return "reflect" if config.critic.enabled else "retry"
    return "stop"
