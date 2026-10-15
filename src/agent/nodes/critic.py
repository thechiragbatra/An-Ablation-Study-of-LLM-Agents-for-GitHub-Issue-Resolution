"""Critic: rubric-based selection among a round's candidates, and reflection on a failed
validation. All outputs are pydantic-validated JSON; routing never reads free text.
"""

from __future__ import annotations

from pydantic import BaseModel

from agent.deps import Deps
from agent.llm import LLMRequest, Message
from agent.nodes.selector import forced_choice, mark_selected, selection_pool
from agent.prompts import CRITIC_REFLECT_SYSTEM, CRITIC_SELECT_SYSTEM, format_files
from agent.state import (
    AgentState,
    Candidate,
    CriticNote,
    StateUpdate,
    ValidationResult,
    tail,
)


class CandidateScore(BaseModel):
    candidate_index: int
    correctness_likelihood: int
    minimality: int
    test_impact: int
    risk: int  # higher = riskier
    justification: str


class CriticSelection(BaseModel):
    scores: list[CandidateScore]
    selected_index: int
    selection_reason: str


class CriticReflection(BaseModel):
    failure_analysis: str
    revision_guidance: list[str]


def _clamp(x: int) -> float:
    return float(min(5, max(1, x)))


def _rubric(s: CandidateScore) -> dict[str, float]:
    return {
        "correctness_likelihood": _clamp(s.correctness_likelihood),
        "minimality": _clamp(s.minimality),
        "test_impact": _clamp(s.test_impact),
        "risk": _clamp(s.risk),
    }


def _fallback_pick(pool: list[Candidate], scores: dict[int, dict[str, float]]) -> Candidate:
    """Deterministic: highest correctness, then lowest risk, then lowest index.
    Unscored candidates rank last."""

    def key(c: Candidate) -> tuple[float, float, int]:
        s = scores.get(c["index"])
        if s is None:
            return (1.0, 0.0, c["index"])
        return (-s["correctness_likelihood"], s["risk"], c["index"])

    return min(pool, key=key)


def _sandbox_check(state: AgentState, deps: Deps, pool: list[Candidate]) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    sb = deps.sandbox
    for cand in pool:
        assert cand["diff"] is not None
        sb.reset_to_base()
        applied = sb.apply_diff(cand["diff"])
        if applied.ok:
            res = sb.run(state["test_command"], deps.config.sandbox.timeout_s)
            exit_code, passed, timed_out = res.exit_code, res.ok, res.timed_out
            duration, out, err = res.duration_s, res.stdout, res.stderr
        else:
            exit_code, passed, timed_out = None, False, False
            duration, out, err = 0.0, "", applied.stderr
        results.append(
            ValidationResult(
                stage="critic",
                candidate_index=cand["index"],
                attempt=cand["attempt"],
                command=state["test_command"],
                applied=applied.ok,
                exit_code=exit_code,
                passed=passed,
                timed_out=timed_out,
                duration_s=duration,
                stdout_tail=tail(out),
                stderr_tail=tail(err),
            )
        )
    sb.reset_to_base()
    return results


def _describe(cand: Candidate, check: ValidationResult | None) -> str:
    if cand["apply_error"] is not None:
        body = f"<not_applicable>{cand['apply_error']}</not_applicable>"
    elif cand["duplicate_of"] is not None:
        body = f"<duplicate_of>{cand['duplicate_of']}</duplicate_of>"
    else:
        body = f"<diff>\n{cand['diff']}\n</diff>"
    if check is not None:
        verdict = "PASSED" if check["passed"] else f"FAILED (exit code {check['exit_code']})"
        body += (
            f"\n<sandbox_test_result>{verdict}\n<stdout>\n{check['stdout_tail']}\n</stdout>"
            f"\n</sandbox_test_result>"
        )
    return (
        f"<candidate index={cand['index']}>\n<rationale>{cand['rationale']}</rationale>\n"
        f"{body}\n</candidate>"
    )


def critic(state: AgentState, deps: Deps) -> StateUpdate:
    """Select mode: score the round's candidates against the rubric and pick one."""
    cfg = deps.config
    rnd, pool = selection_pool(state)
    forced = forced_choice(rnd, pool)
    if forced is not None:  # nothing to judge between: no LLM call
        return mark_selected(forced, "only")

    checks = _sandbox_check(state, deps, pool) if cfg.critic.uses_sandbox else []
    by_idx = {v["candidate_index"]: v for v in checks}
    files = {p: deps.sandbox.read_base_file(p) for p in state["localized_files"]}
    req = LLMRequest(
        purpose="critic_select",
        system=CRITIC_SELECT_SYSTEM,
        messages=(
            Message(
                "user",
                f"<issue>\n{state['issue_text']}\n</issue>\n\n{format_files(files)}\n\n"
                + "\n\n".join(_describe(c, by_idx.get(c["index"])) for c in rnd)
                + "\n\nSelectable indices: "
                + ", ".join(str(c["index"]) for c in pool),
            ),
        ),
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        effort=cfg.effort,
    )
    out = deps.llm.complete(req, CriticSelection).parsed
    assert isinstance(out, CriticSelection)

    round_idx = {c["index"] for c in rnd}
    scores = {s.candidate_index: _rubric(s) for s in out.scores if s.candidate_index in round_idx}
    justifications = {s.candidate_index: s.justification for s in out.scores}
    pool_idx = {c["index"] for c in pool}
    if out.selected_index in pool_idx:
        chosen = next(c for c in pool if c["index"] == out.selected_index)
        update = mark_selected(chosen, "critic")
    else:
        chosen = _fallback_pick(pool, scores)
        update = mark_selected(chosen, "critic_fallback")

    notes: list[CriticNote] = []
    for idx in sorted(scores.keys() | {chosen["index"]}):
        critique = justifications.get(idx, "")
        if idx == chosen["index"]:
            critique = f"{critique}\n[selection] {out.selection_reason}".strip()
            if update["candidates"][0]["selected_by"] == "critic_fallback":
                critique += f"\n[fallback] critic picked invalid index {out.selected_index}"
        notes.append(
            CriticNote(
                kind="selection",
                attempt=state["attempt_count"],
                reflection_depth=state["reflection_depth"],
                candidate_index=idx,
                scores=scores.get(idx, {}),
                selected=idx == chosen["index"],
                critique=critique,
            )
        )
    return {**update, "critic_notes": notes, "validation_results": checks}


def critic_reflect(state: AgentState, deps: Deps) -> StateUpdate:
    """Reflection mode: diagnose a failed validation to guide the next revision."""
    cfg = deps.config
    idx = state["selected_index"]
    if idx is None:
        raise ValueError("critic_reflect reached with no selected candidate")
    cand = state["candidates"][idx]
    final = next(
        v
        for v in reversed(state["validation_results"])
        if v["stage"] == "final" and v["candidate_index"] == idx
    )
    earlier = [n["critique"] for n in state["critic_notes"] if n["kind"] == "reflection"]
    files = {p: deps.sandbox.read_base_file(p) for p in state["localized_files"]}
    if cand["apply_error"] is not None:
        outcome = f"The edits could not be applied: {cand['apply_error']}"
    else:
        outcome = (
            f"Exit code {final['exit_code']}, timed out: {final['timed_out']}\n"
            f"<stdout>\n{final['stdout_tail']}\n</stdout>\n<stderr>\n{final['stderr_tail']}\n"
            "</stderr>"
        )
    history = (
        "\n\n<earlier_reviews>\n" + "\n---\n".join(earlier) + "\n</earlier_reviews>"
        if earlier
        else ""
    )
    req = LLMRequest(
        purpose="critic_reflect",
        system=CRITIC_REFLECT_SYSTEM,
        messages=(
            Message(
                "user",
                f"<issue>\n{state['issue_text']}\n</issue>\n\n{format_files(files)}\n\n"
                f"<failed_patch>\n<rationale>{cand['rationale']}</rationale>\n"
                f"<diff>\n{cand['diff'] or '(not applied)'}\n</diff>\n</failed_patch>\n\n"
                f"<validation command={state['test_command']!r}>\n{outcome}\n</validation>"
                f"{history}",
            ),
        ),
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        effort=cfg.effort,
    )
    out = deps.llm.complete(req, CriticReflection).parsed
    assert isinstance(out, CriticReflection)
    guidance = "\n".join(f"- {g}" for g in out.revision_guidance)
    note = CriticNote(
        kind="reflection",
        attempt=state["attempt_count"],
        reflection_depth=state["reflection_depth"],
        candidate_index=idx,
        scores={},
        selected=False,
        critique=f"{out.failure_analysis}\n{guidance}".strip(),
    )
    return {"critic_notes": [note]}
