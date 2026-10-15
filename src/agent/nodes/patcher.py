"""Materializes every candidate of the current round into a diff. No LLM.

Each candidate is applied to a clean checkout, its diff recorded, and the tree reset, so
the critic can compare diffs and the validator can re-apply the selected one with
``git apply``. An edit that cannot be applied exactly is recorded as ``apply_error`` (a
measurable failure category) rather than raised.
"""

from __future__ import annotations

import shlex

from agent.deps import Deps
from agent.sandbox.base import Sandbox, SandboxError, safe_relpath
from agent.state import AgentState, Candidate, FileEdit, StateUpdate, round_candidates


class EditApplyError(Exception):
    pass


def _apply_edit(sandbox: Sandbox, edit: FileEdit) -> None:
    try:
        path = safe_relpath(edit["path"])
    except SandboxError as exc:
        raise EditApplyError(str(exc)) from None
    exists = sandbox.run(f"test -f {shlex.quote(path)}", 30).ok
    if edit["search"] == "":
        if exists:
            raise EditApplyError(f"{path}: empty search means create, but the file exists")
        sandbox.write(path, edit["replace"])
        return
    if not exists:
        raise EditApplyError(f"{path}: file does not exist")
    content = sandbox.read_file(path)
    count = content.count(edit["search"])
    if count != 1:
        raise EditApplyError(f"{path}: search text matched {count} times (must be exactly 1)")
    sandbox.write(path, content.replace(edit["search"], edit["replace"], 1))


def _materialize(sandbox: Sandbox, cand: Candidate) -> tuple[str | None, str | None]:
    """Returns (diff, apply_error) for one candidate, leaving the tree clean."""
    sandbox.reset_to_base()
    try:
        for edit in cand["edits"]:
            _apply_edit(sandbox, edit)
        diff = sandbox.read_diff()
    except EditApplyError as exc:
        return None, str(exc)
    finally:
        sandbox.reset_to_base()
    if diff == "":
        return None, "edits produced no change"
    return diff, None


def patcher(state: AgentState, deps: Deps) -> StateUpdate:
    pending = [
        c
        for c in round_candidates(state["candidates"], state["attempt_count"])
        if c["diff"] is None and c["apply_error"] is None
    ]
    if not pending:
        raise ValueError("patcher reached with no pending candidates")

    first_with_diff: dict[str, int] = {}
    updated: list[Candidate] = []
    for cand in pending:
        diff, apply_error = _materialize(deps.sandbox, cand)
        duplicate_of = None
        if diff is not None:
            duplicate_of = first_with_diff.setdefault(diff, cand["index"])
            if duplicate_of == cand["index"]:
                duplicate_of = None
        updated.append(
            Candidate(
                **{**cand, "diff": diff, "apply_error": apply_error, "duplicate_of": duplicate_of}
            )
        )
    return {"candidates": updated}
