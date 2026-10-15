"""Termination is provable for every configuration in the ablation grid.

1. Every config in the grid either validates or is rejected for a stated reason.
2. Every cycle in every compiled graph passes through ``architect`` (structural check on
   LangGraph's own graph description).
3. With a validator that always fails, the router halts after exactly ``max_rounds()``
   architect rounds, within the recursion limit.
Together with "architect strictly increments attempt_count" (asserted in the routing
tests) and the budget guard, every run halts.
"""

from __future__ import annotations

import contextlib
import itertools
from pathlib import Path
from typing import Any

import pytest
from langgraph.graph import END
from pydantic import ValidationError

from agent.budget import BudgetLedger
from agent.config import ExperimentConfig
from agent.deps import Deps
from agent.graph import _after_validator, build_graph, failure_targets
from agent.llm import AccountedLLM, PricingTable
from agent.policy import next_after_failure
from agent.sandbox import E2BSandbox
from agent.tracing import Tracer
from tests.conftest import CONFIGS
from tests.fake_llm import FakeLLM
from tests.scenarios import router_state

KNOWN_REJECTIONS = (
    "K must be 1",
    "D must be 0",
    "N must be 1",
    "no critic",
    "requires critic.enabled",
)

GRID = list(
    itertools.product(
        ["A0", "A1", "A2", "A3"],
        [False, True],  # critic.enabled
        [False, True],  # critic.uses_sandbox
        [1, 3, 5],  # K
        [0, 1, 2],  # D
    )
)


def _config(variant: str, critic: bool, sandbox: bool, k: int, d: int) -> ExperimentConfig:
    base = ExperimentConfig.from_yaml(CONFIGS / "a0.yaml").model_dump()
    data: dict[str, Any] = {
        **base,
        "variant": variant,
        "K": k,
        "D": d,
        "N": 3 if variant == "A1" else 1,
        "critic": {"enabled": critic, "uses_sandbox": sandbox},
    }
    return ExperimentConfig.model_validate(data)


def _valid_grid() -> list[ExperimentConfig]:
    out = []
    for combo in GRID:
        with contextlib.suppress(ValidationError):
            out.append(_config(*combo))
    return out


@pytest.mark.parametrize("combo", GRID, ids=lambda c: "-".join(map(str, c)))
def test_grid_config_valid_or_rejected_for_a_stated_reason(combo: tuple[Any, ...]) -> None:
    try:
        _config(*combo)
    except ValidationError as exc:
        message = str(exc)
    else:
        return
    assert any(reason in message for reason in KNOWN_REJECTIONS), message


def test_grid_covers_every_ablation_switch() -> None:
    valid = _valid_grid()
    assert {c.variant for c in valid} == {"A0", "A1", "A2", "A3"}
    a2a3 = [c for c in valid if c.variant in ("A2", "A3")]
    assert {c.critic.enabled for c in a2a3} == {True, False}
    assert {c.critic.uses_sandbox for c in a2a3} == {True, False}
    assert {c.K for c in a2a3} == {1, 3, 5}
    assert {c.D for c in valid if c.variant == "A3"} == {0, 1, 2}


def _deps(config: ExperimentConfig, tmp_path: Path) -> Deps:
    tracer = Tracer(tmp_path, "t", "i", config)
    llm = AccountedLLM(FakeLLM({}), PricingTable.load(), BudgetLedger(1, "run"), tracer)
    return Deps(config=config, llm=llm, sandbox=E2BSandbox("t"), tracer=tracer)


def _cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    """All elementary cycles of a small directed graph (DFS from each node, canonical
    start = smallest node name in the cycle)."""
    found: list[list[str]] = []

    def dfs(start: str, node: str, path: list[str]) -> None:
        for nxt in edges.get(node, ()):
            if nxt == start:
                found.append([*path])
            elif nxt not in path and nxt > start:
                dfs(start, nxt, [*path, nxt])

    for n in sorted(edges):
        dfs(n, n, [n])
    return found


@pytest.mark.parametrize("config", _valid_grid(), ids=lambda c: c.config_hash()[:8])
def test_every_cycle_passes_through_architect(config: ExperimentConfig, tmp_path: Path) -> None:
    graph = build_graph(config, _deps(config, tmp_path)).get_graph()
    edges: dict[str, set[str]] = {}
    for e in graph.edges:
        edges.setdefault(e.source, set()).add(e.target)
    cycles = _cycles(edges)
    for cycle in cycles:
        assert "architect" in cycle, cycle
    if config.max_rounds() == 1:
        assert cycles == [], "single-round configs must be acyclic"
    else:
        assert cycles, "multi-round configs must loop back"
    back = {e.source for e in graph.edges if e.target == "architect"} - {"__start__", "localizer"}
    assert back <= {"validator", "critic_reflect"}
    assert set(failure_targets(config)) <= {t for t in edges.get("validator", set())}


@pytest.mark.parametrize("config", _valid_grid(), ids=lambda c: c.config_hash()[:8])
def test_always_failing_run_halts_after_max_rounds(config: ExperimentConfig) -> None:
    route = _after_validator(config)
    attempts = 0
    executed = 1  # localizer
    for _ in range(100):  # far beyond any legal bound
        attempts += 1  # architect
        executed += 4  # architect, patcher, select, validator
        stop = next_after_failure(config, attempts) == "stop"
        nxt = route(router_state(status="unverified" if stop else "running",
                                 attempt_count=attempts))  # fmt: skip
        if nxt == END:
            break
        if nxt == "critic_reflect":
            executed += 1
    else:
        pytest.fail("router never reached END")
    assert attempts == config.max_rounds()
    assert executed < config.recursion_limit()
