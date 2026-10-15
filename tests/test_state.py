from __future__ import annotations

import json
import operator
import typing

import pytest

from agent.instance import Instance
from agent.state import (
    AgentState,
    Candidate,
    StateUpdate,
    initial_state,
    tail,
    upsert_candidates,
)

SPEC_FIELDS = {
    # Fields named in the project brief.
    "instance_id", "repo", "base_commit", "issue_text", "localized_files", "candidates",
    "selected_index", "critic_notes", "validation_results", "attempt_count",
    "reflection_depth", "cost_so_far", "trace_path", "status",
    # Additions approved in the Phase 1 plan.
    "test_command", "error",
}  # fmt: skip


def _cand(i: int, diff: str | None = None) -> Candidate:
    return Candidate(
        index=i, edits=[], rationale=f"c{i}", diff=diff, attempt=1, reflection_depth=0,
        apply_error=None, duplicate_of=None, selected_by=None,
    )  # fmt: skip


def test_state_fields_match_spec() -> None:
    assert set(AgentState.__annotations__) == SPEC_FIELDS


def test_state_update_mirrors_state() -> None:
    assert set(StateUpdate.__annotations__) == set(AgentState.__annotations__)
    assert StateUpdate.__total__ is False


def test_gold_tests_not_in_state() -> None:
    lowered = {f.lower() for f in AgentState.__annotations__}
    assert not any("fail_to_pass" in f or "gold" in f for f in lowered)


def test_history_fields_have_reducers() -> None:
    hints = typing.get_type_hints(AgentState, include_extras=True)
    assert hints["candidates"].__metadata__ == (upsert_candidates,)
    assert hints["critic_notes"].__metadata__ == (operator.add,)
    assert hints["validation_results"].__metadata__ == (operator.add,)


def test_upsert_appends_and_replaces_without_mutation() -> None:
    existing = [_cand(0), _cand(1)]
    snapshot = json.dumps(existing)
    out = upsert_candidates(existing, [_cand(1, diff="d"), _cand(2)])
    assert [c["index"] for c in out] == [0, 1, 2]
    assert out[1]["diff"] == "d"
    assert json.dumps(existing) == snapshot  # inputs untouched


def test_upsert_rejects_gaps() -> None:
    with pytest.raises(ValueError, match="skips ahead"):
        upsert_candidates([_cand(0)], [_cand(5)])


def test_initial_state(instance: Instance) -> None:
    s = initial_state(instance, "/tmp/t.jsonl")
    assert set(s) == SPEC_FIELDS
    assert s["status"] == "running"
    assert s["attempt_count"] == 0
    assert s["candidates"] == []
    assert s["selected_index"] is None
    assert json.loads(json.dumps(s)) == s  # JSON round-trip (traces hash it)


def test_tail_truncates() -> None:
    assert tail("abc", 10) == "abc"
    t = tail("x" * 100, 10)
    assert t.endswith("x" * 10)
    assert "truncated" in t
