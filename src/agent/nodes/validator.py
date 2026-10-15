"""Final validation of the selected candidate. No LLM. Identical across variants.

Resets to the base commit, ``git apply``s the candidate's stored diff (the same mechanism
SWE-bench scoring uses) and runs the agent-visible test command.
"""

from __future__ import annotations

from agent.deps import Deps
from agent.policy import next_after_failure
from agent.state import AgentState, StateUpdate, Status, ValidationResult, is_applicable, tail


def validator(state: AgentState, deps: Deps) -> StateUpdate:
    idx = state["selected_index"]
    if idx is None:
        raise ValueError("validator reached with no selected candidate")
    cand = state["candidates"][idx]
    cmd = state["test_command"]
    sb = deps.sandbox

    applied = False
    exit_code: int | None = None
    passed = timed_out = False
    duration, out, err = 0.0, "", ""
    if is_applicable(cand):
        assert cand["diff"] is not None
        sb.reset_to_base()
        ap = sb.apply_diff(cand["diff"])
        applied = ap.ok
        if applied:
            res = sb.run(cmd, deps.config.sandbox.timeout_s)
            exit_code, passed, timed_out = res.exit_code, res.ok, res.timed_out
            duration, out, err = res.duration_s, res.stdout, res.stderr
        else:
            err = ap.stderr

    result = ValidationResult(
        stage="final",
        candidate_index=idx,
        attempt=cand["attempt"],
        command=cmd,
        applied=applied,
        exit_code=exit_code,
        passed=passed,
        timed_out=timed_out,
        duration_s=duration,
        stdout_tail=tail(out),
        stderr_tail=tail(err),
    )
    status: Status
    if passed:
        status = "sandbox_verified"
    elif next_after_failure(deps.config, state["attempt_count"]) == "stop":
        status = "unverified"
    else:
        status = "running"
    return {"validation_results": [result], "status": status}
