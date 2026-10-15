"""Shared scripted scenarios for graph-routing tests (FakeLLM + real Docker sandbox)."""

from __future__ import annotations

from typing import Any, cast

from agent.budget import BudgetLedger
from agent.config import ExperimentConfig
from agent.instance import Instance
from agent.nodes.architect import ArchitectOutput, CandidateModel, EditModel
from agent.nodes.critic import CandidateScore, CriticReflection, CriticSelection
from agent.nodes.localizer import LocalizationOutput
from agent.run import RunOutcome, run_instance
from agent.state import AgentState, initial_state, upsert_candidates
from agent.tracing import NodeRecord, read_trace
from tests.fake_llm import FakeLLM

BUGGY = "    return sum(values) / len(values)\n"
FIXED = "    if not values:\n        return 0.0\n    return sum(values) / len(values)\n"
FIXED_ALT = "    return sum(values) / len(values) if values else 0.0\n"
WRONG = "    return sum(values) / len(values) if values else None\n"


def loc(*files: str) -> LocalizationOutput:
    return LocalizationOutput(files=list(files) or ["calc/stats.py"], reasoning="mean lives here")


def cand(search: str, replace: str, path: str = "calc/stats.py", why: str = "c") -> CandidateModel:
    return CandidateModel(
        rationale=why, edits=[EditModel(path=path, search=search, replace=replace)]
    )


C_CORRECT = cand(BUGGY, FIXED, why="guard empty input")
C_CORRECT_ALT = cand(BUGGY, FIXED_ALT, why="conditional expression")
C_WRONG = cand(BUGGY, WRONG, why="return None for empty")
C_UNAPPLIABLE = cand("this text is not in the file\n", "x\n", why="hallucinated context")


def arch(*cands: CandidateModel) -> ArchitectOutput:
    return ArchitectOutput(candidates=list(cands))


CORRECT = arch(C_CORRECT)
WRONG_FIX = arch(C_WRONG)
UNAPPLIABLE = arch(C_UNAPPLIABLE)


def score(idx: int, correctness: int = 3, risk: int = 2) -> CandidateScore:
    return CandidateScore(
        candidate_index=idx,
        correctness_likelihood=correctness,
        minimality=4,
        test_impact=3,
        risk=risk,
        justification=f"candidate {idx} justification",
    )


def select(selected: int, *scores: CandidateScore) -> CriticSelection:
    return CriticSelection(
        scores=list(scores), selected_index=selected, selection_reason=f"pick {selected}"
    )


def reflect(analysis: str = "REFLECTION: empty input still returns None") -> CriticReflection:
    return CriticReflection(failure_analysis=analysis, revision_guidance=["return 0.0 instead"])


def run(
    config: ExperimentConfig,
    instance: Instance,
    fake: FakeLLM,
    experiment_ledger: BudgetLedger | None = None,
) -> RunOutcome:
    return run_instance(config, instance, provider=fake, experiment_ledger=experiment_ledger)


def trace_nodes(outcome: RunOutcome) -> list[NodeRecord]:
    return [r for r in read_trace(outcome.trace_path) if isinstance(r, NodeRecord)]


def graph_nodes(outcome: RunOutcome) -> list[str]:
    return [n.node for n in trace_nodes(outcome) if not n.node.startswith("_")]


def router_state(**kw: Any) -> AgentState:
    s = initial_state(
        Instance(instance_id="i", repo="r", base_commit="c", issue_text="t", test_command="true"),
        "t.jsonl",
    )
    return cast(AgentState, {**s, **kw})


def merge(state: AgentState, update: dict[str, Any]) -> AgentState:
    """Apply an update the way LangGraph's reducers would."""
    merged: dict[str, Any] = dict(state)
    for key, value in update.items():
        if key == "candidates":
            merged[key] = upsert_candidates(state["candidates"], value)
        elif key in ("validation_results", "critic_notes"):
            merged[key] = [*merged[key], *value]
        else:
            merged[key] = value
    return cast(AgentState, merged)
