"""AgentState: the single source of truth shared by every graph variant (A0-A3).

History fields have reducers, so a node returns only the *new* items: ``critic_notes``
and ``validation_results`` are append-only (``operator.add``); ``candidates`` is
upsert-by-index (append new, or replace one to record its diff). Every other field is
replace-on-write.
Nodes must return a partial ``StateUpdate`` and never mutate the state they receive.

Gold evaluation tests (SWE-bench FAIL_TO_PASS / PASS_TO_PASS) are intentionally *not*
part of this state: anything in here may reach a prompt, so eval-only data stays out.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from agent.instance import Instance

Status = Literal["running", "sandbox_verified", "unverified", "budget_exceeded", "error"]
TERMINAL_STATUSES: frozenset[Status] = frozenset(
    {"sandbox_verified", "unverified", "budget_exceeded", "error"}
)
ABORT_STATUSES: frozenset[Status] = frozenset({"budget_exceeded", "error"})
SelectedBy = Literal["only", "critic", "critic_fallback", "random"]
ValidationStage = Literal["critic", "final"]
CriticNoteKind = Literal["selection", "reflection"]

OUTPUT_TAIL_CHARS = 4096


class FileEdit(TypedDict):
    """One exact search/replace edit. ``search`` must match exactly once in ``path``."""

    path: str
    search: str
    replace: str


class Candidate(TypedDict):
    index: int
    edits: list[FileEdit]
    rationale: str
    diff: str | None
    attempt: int  # architect round that produced it (1-based)
    reflection_depth: int
    apply_error: str | None
    duplicate_of: int | None  # earlier candidate in the same round with an identical diff
    selected_by: SelectedBy | None


def is_applicable(cand: Candidate) -> bool:
    return cand["apply_error"] is None and cand["diff"] is not None


def round_candidates(candidates: list[Candidate], attempt: int) -> list[Candidate]:
    return [c for c in candidates if c["attempt"] == attempt]


def upsert_candidates(existing: list[Candidate], new: list[Candidate]) -> list[Candidate]:
    """Reducer for ``candidates``: an item whose ``index`` already exists replaces it (the
    patcher filling in ``diff``/``apply_error``); ``index == len`` appends. Returns a new
    list; inputs are never mutated."""
    out = list(existing)
    for cand in new:
        i = cand["index"]
        if i < len(out):
            out[i] = cand
        elif i == len(out):
            out.append(cand)
        else:
            raise ValueError(f"candidate index {i} skips ahead of history length {len(out)}")
    return out


class CriticNote(TypedDict):
    kind: CriticNoteKind
    attempt: int
    reflection_depth: int
    candidate_index: int
    scores: dict[str, float]  # rubric dimension -> 1..5; empty for reflections
    selected: bool
    critique: str


class ValidationResult(TypedDict):
    stage: ValidationStage  # "critic" runs inform selection; only "final" is the verdict
    candidate_index: int
    attempt: int
    command: str
    applied: bool
    exit_code: int | None
    passed: bool
    timed_out: bool
    duration_s: float
    stdout_tail: str
    stderr_tail: str


class AgentState(TypedDict):
    instance_id: str
    repo: str
    base_commit: str
    issue_text: str
    test_command: str
    localized_files: list[str]
    candidates: Annotated[list[Candidate], upsert_candidates]
    selected_index: int | None
    critic_notes: Annotated[list[CriticNote], operator.add]
    validation_results: Annotated[list[ValidationResult], operator.add]
    attempt_count: int
    reflection_depth: int
    cost_so_far: float
    trace_path: str
    status: Status
    error: str | None


class StateUpdate(TypedDict, total=False):
    """Partial update returned by a node. Mirrors ``AgentState`` with every key optional."""

    instance_id: str
    repo: str
    base_commit: str
    issue_text: str
    test_command: str
    localized_files: list[str]
    candidates: list[Candidate]
    selected_index: int | None
    critic_notes: list[CriticNote]
    validation_results: list[ValidationResult]
    attempt_count: int
    reflection_depth: int
    cost_so_far: float
    trace_path: str
    status: Status
    error: str | None


def initial_state(instance: Instance, trace_path: str) -> AgentState:
    return AgentState(
        instance_id=instance.instance_id,
        repo=instance.repo,
        base_commit=instance.base_commit,
        issue_text=instance.issue_text,
        test_command=instance.test_command,
        localized_files=[],
        candidates=[],
        selected_index=None,
        critic_notes=[],
        validation_results=[],
        attempt_count=0,
        reflection_depth=0,
        cost_so_far=0.0,
        trace_path=trace_path,
        status="running",
        error=None,
    )


def tail(text: str, limit: int = OUTPUT_TAIL_CHARS) -> str:
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]
