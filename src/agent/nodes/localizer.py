from __future__ import annotations

from pydantic import BaseModel, Field

from agent.deps import Deps
from agent.llm import LLMRequest, Message
from agent.prompts import LOCALIZER_SYSTEM
from agent.state import AgentState, StateUpdate

# SWE-bench Lite is Python-only; listing only *.py keeps large repos (django: ~2.7k .py
# of ~6.6k files) fully visible instead of silently truncated.
_MAX_LISTED_FILES = 10_000


class LocalizationOutput(BaseModel):
    files: list[str] = Field(description="Repo-relative paths, most relevant first")
    reasoning: str


class LocalizationError(Exception):
    pass


def localizer(state: AgentState, deps: Deps) -> StateUpdate:
    listing = deps.sandbox.run("git ls-files -- '*.py'", 60)
    if not listing.ok:
        raise LocalizationError(f"git ls-files failed: {listing.stderr[:200]}")
    all_files = listing.stdout.splitlines()
    shown = all_files[:_MAX_LISTED_FILES]
    limit = deps.config.max_localized_files

    req = LLMRequest(
        purpose="localize",
        system=LOCALIZER_SYSTEM,
        messages=(
            Message(
                "user",
                f"<issue>\n{state['issue_text']}\n</issue>\n\n"
                f"<files count={len(all_files)} shown={len(shown)}>\n"
                + "\n".join(shown)
                + f"\n</files>\n\nChoose at most {limit} files.",
            ),
        ),
        model=deps.config.model,
        max_tokens=deps.config.max_tokens,
        effort=deps.config.effort,
    )
    out = deps.llm.complete(req, LocalizationOutput).parsed
    assert isinstance(out, LocalizationOutput)

    known = set(all_files)
    chosen = list(dict.fromkeys(f for f in out.files if f in known))[:limit]
    if not chosen:
        raise LocalizationError(f"no valid paths among localizer output {out.files[:10]}")
    return {"localized_files": chosen}
