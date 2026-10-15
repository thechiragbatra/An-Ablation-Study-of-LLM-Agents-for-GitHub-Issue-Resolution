from __future__ import annotations

from pydantic import BaseModel, Field

from agent.deps import Deps
from agent.llm import LLMRequest, Message
from agent.prompts import ARCHITECT_DIVERSITY, ARCHITECT_SYSTEM, format_files
from agent.state import AgentState, Candidate, FileEdit, StateUpdate

_FEEDBACK_ROUNDS = 3  # most recent failed rounds shown back to the model


class EditModel(BaseModel):
    path: str
    search: str
    replace: str


class CandidateModel(BaseModel):
    rationale: str
    edits: list[EditModel] = Field(min_length=1)


class ArchitectOutput(BaseModel):
    candidates: list[CandidateModel] = Field(min_length=1)


def _feedback(state: AgentState) -> str:
    """Earlier failed rounds: the validated (selected) candidate, its final test outcome,
    and the critic's reflection on it if there was one. Empty on the first round."""
    finals = {v["candidate_index"]: v for v in state["validation_results"] if v["stage"] == "final"}
    reflections = {
        n["attempt"]: n["critique"] for n in state["critic_notes"] if n["kind"] == "reflection"
    }
    blocks: list[str] = []
    validated = [state["candidates"][i] for i in finals]
    for cand in validated[-_FEEDBACK_ROUNDS:]:
        vr = finals[cand["index"]]
        if vr["passed"]:
            continue
        if cand["apply_error"]:
            outcome = f"The edits could not be applied: {cand['apply_error']}"
        elif not vr["applied"]:
            outcome = "The patch could not be applied for validation."
        elif vr["timed_out"]:
            outcome = "The test command timed out."
        else:
            outcome = (
                f"Tests failed (exit code {vr['exit_code']}).\n"
                f"<stdout>\n{vr['stdout_tail']}\n</stdout>\n<stderr>\n{vr['stderr_tail']}\n</stderr>"
            )
        critique = reflections.get(cand["attempt"])
        review = f"\n<review>\n{critique}\n</review>" if critique else ""
        blocks.append(
            f"<attempt number={cand['attempt']}>\n<rationale>{cand['rationale']}</rationale>\n"
            f"<diff>\n{cand['diff'] or '(not applied)'}\n</diff>\n{outcome}{review}\n</attempt>"
        )
    if not blocks:
        return ""
    return (
        "\n\nEarlier attempts failed. Each was applied to the ORIGINAL files shown above, "
        "not on top of each other. Learn from them:\n" + "\n".join(blocks)
    )


def architect(state: AgentState, deps: Deps) -> StateUpdate:
    cfg = deps.config
    files = {p: deps.sandbox.read_base_file(p) for p in state["localized_files"]}
    k = cfg.K
    ask = f"Return exactly {k} candidate fix{'es' if k > 1 else ''}."
    if k > 1:
        ask += "\n" + ARCHITECT_DIVERSITY
    req = LLMRequest(
        purpose="architect",
        system=ARCHITECT_SYSTEM,
        messages=(
            Message(
                "user",
                f"<issue>\n{state['issue_text']}\n</issue>\n\n{format_files(files)}"
                f"{_feedback(state)}\n\n"
                f"Validation command that will be run: `{state['test_command']}`\n{ask}",
            ),
        ),
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        effort=cfg.effort,
    )
    out = deps.llm.complete(req, ArchitectOutput).parsed
    assert isinstance(out, ArchitectOutput)

    attempt = state["attempt_count"] + 1
    depth = attempt - 1 if cfg.variant == "A3" else 0
    start = len(state["candidates"])
    # Fewer than K is accepted (the shortfall is visible as n_new vs K in the trace);
    # extras beyond K are dropped so K stays a controlled variable.
    new: list[Candidate] = [
        Candidate(
            index=start + i,
            edits=[FileEdit(path=e.path, search=e.search, replace=e.replace) for e in c.edits],
            rationale=c.rationale,
            diff=None,
            attempt=attempt,
            reflection_depth=depth,
            apply_error=None,
            duplicate_of=None,
            selected_by=None,
        )
        for i, c in enumerate(out.candidates[:k])
    ]
    return {
        "candidates": new,
        "attempt_count": attempt,
        "reflection_depth": depth,
        "selected_index": None,
    }
