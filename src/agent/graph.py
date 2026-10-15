"""Graph wiring. Variants differ ONLY here, selected by ``config.variant``.

Every variant: localizer -> architect -> patcher -> select -> validator -> ...
  select = critic (LLM rubric) if critic.enabled else selector (trivial / seeded random)
After a failed validation (see agent.policy):
  A0, A2: END
  A1:     validator -> architect                     while attempt_count < N
  A3:     validator -> critic_reflect -> architect   while attempt_count < D+1
          (validator -> architect when the critic is disabled)

Termination: every cycle passes through ``architect``, which strictly increments
``attempt_count``; every back-edge requires ``attempt_count < config.max_rounds()``. The
budget guard (any over-budget call becomes a terminal status) and LangGraph's recursion
limit are independent backstops. tests/test_termination.py checks this structurally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.budget import BudgetExceeded
from agent.config import ExperimentConfig
from agent.deps import Deps
from agent.nodes import (
    NodeFn,
    architect,
    critic,
    critic_reflect,
    localizer,
    patcher,
    publisher,
    selector,
    validator,
)
from agent.policy import next_after_failure
from agent.state import ABORT_STATUSES, AgentState, StateUpdate
from agent.tracing import Tracer

Graph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def traced_call(
    tracer: Tracer,
    name: str,
    state: AgentState,
    fn: Callable[[], StateUpdate],
    ledger_spent: Callable[[], float],
) -> StateUpdate:
    """Run ``fn`` inside a trace scope. Exceptions become terminal statuses, never escapes,
    so every run (including aborted ones) ends with a complete trace."""
    scope = tracer.begin_node(name, cast(dict[str, Any], state))
    error: str | None = None
    try:
        update: StateUpdate = dict(fn())  # type: ignore[assignment]
    except BudgetExceeded as exc:
        error = str(exc)
        update = {"status": "budget_exceeded", "error": error}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        update = {"status": "error", "error": error}
    update["cost_so_far"] = ledger_spent()
    status_after = update.get("status", state["status"])
    tracer.end_node(scope, cast(dict[str, Any], update), status_after, error)
    return update


def _traced(name: str, fn: NodeFn, deps: Deps) -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        return traced_call(
            deps.tracer, name, state, lambda: fn(state, deps), lambda: deps.llm.ledger.spent
        )

    node.__name__ = name
    return node


def _continue_unless_aborted(next_node: str) -> Callable[[AgentState], str]:
    def route(state: AgentState) -> str:
        return END if state["status"] in ABORT_STATUSES else next_node

    return route


_ON_FAILURE = {"retry": "architect", "reflect": "critic_reflect", "stop": END}


def _after_validator(config: ExperimentConfig) -> Callable[[AgentState], str]:
    """The only back-edges. ``policy`` guards them with ``attempt_count < max_rounds``;
    the explicit attempt check here makes the bound local to the router as well."""

    def route(state: AgentState) -> str:
        if state["status"] == "sandbox_verified" and config.publisher.enabled:
            return "publisher"  # terminal leaf: packages locally, never loops
        if state["status"] != "running" or state["attempt_count"] >= config.max_rounds():
            return END
        return _ON_FAILURE[next_after_failure(config, state["attempt_count"])]

    return route


def select_node(config: ExperimentConfig) -> tuple[str, NodeFn]:
    return ("critic", critic) if config.critic.enabled else ("selector", selector)


def failure_targets(config: ExperimentConfig) -> list[str]:
    """Nodes the validator can loop back to under this config (for explicit path maps)."""
    if config.max_rounds() <= 1:
        return []
    if config.variant == "A3" and config.critic.enabled:
        return ["critic_reflect"]
    return ["architect"]


def build_graph(config: ExperimentConfig, deps: Deps) -> Graph:
    select_name, select_fn = select_node(config)
    back = failure_targets(config)
    nodes: list[tuple[str, NodeFn]] = [
        ("localizer", localizer),
        ("architect", architect),
        ("patcher", patcher),
        (select_name, select_fn),
        ("validator", validator),
    ]
    if "critic_reflect" in back:
        nodes.append(("critic_reflect", critic_reflect))
    if config.publisher.enabled:
        nodes.append(("publisher", publisher))

    g: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    for name, fn in nodes:
        g.add_node(name, _traced(name, fn, deps))  # type: ignore[call-overload]

    def forward(src: str, dst: str) -> None:
        g.add_conditional_edges(src, _continue_unless_aborted(dst), [dst, END])

    g.add_edge(START, "localizer")
    forward("localizer", "architect")
    forward("architect", "patcher")
    forward("patcher", select_name)
    forward(select_name, "validator")
    leaf = ["publisher"] if config.publisher.enabled else []
    g.add_conditional_edges("validator", _after_validator(config), [*back, *leaf, END])
    if config.publisher.enabled:
        g.add_edge("publisher", END)
    if "critic_reflect" in back:
        forward("critic_reflect", "architect")
    return g.compile()


def run_graph(graph: Graph, state: AgentState, config: ExperimentConfig) -> AgentState:
    final = graph.invoke(state, config={"recursion_limit": config.recursion_limit()})
    return cast(AgentState, final)
