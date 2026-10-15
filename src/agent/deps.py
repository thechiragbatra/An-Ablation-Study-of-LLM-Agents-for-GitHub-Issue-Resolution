"""Run-scoped dependencies handed to every node. Not part of AgentState (not serializable)."""

from __future__ import annotations

from dataclasses import dataclass

from agent.config import ExperimentConfig
from agent.instance import Instance
from agent.llm import AccountedLLM
from agent.sandbox.base import Sandbox
from agent.tracing import Tracer


@dataclass(frozen=True)
class Deps:
    config: ExperimentConfig
    llm: AccountedLLM
    sandbox: Sandbox
    tracer: Tracer
    instance: Instance | None = None  # needed by the publisher (pilot only)
