"""Pending PR packages. Pure and local: nothing here talks to GitHub.

The publisher node turns a sandbox-verified run into a package under
``<pending_dir>/<package_id>/``; only ``pilot.publish --publish`` (with a matching human
approval) can turn a package into a draft PR. ``content_sha256`` binds an approval to
the exact title, body, diff and file contents: any later edit invalidates it.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agent.config import ExperimentConfig
from agent.instance import Instance
from agent.state import AgentState, CriticNote, ValidationResult

TITLE_PREFIX = "[AI-Generated Patch] "
DISCLOSURE_MARKER = "<!-- ai-generated-disclosure -->"
REVIEWER_PLACEHOLDER = "@@REVIEWER@@"
HEAD_BRANCH_PREFIX = "ai-patch/"
_MAX_TITLE = 120
_MAX_OUTPUT = 1500
_DIFF_HEADER = re.compile(r"^diff --git a/\S+ b/(\S+)$")


class PackageError(Exception):
    pass


def diff_files(diff: str) -> tuple[set[str], set[str], set[str]]:
    """(touched, created, deleted) paths of a git diff, from its ``diff --git`` headers
    (so empty new files, which have no ---/+++ lines, are detected too)."""
    touched: set[str] = set()
    created: set[str] = set()
    deleted: set[str] = set()
    current: str | None = None
    for line in diff.splitlines():
        if m := _DIFF_HEADER.match(line):
            current = m.group(1)
            touched.add(current)
        elif current is not None and line.startswith("new file mode"):
            created.add(current)
        elif current is not None and line.startswith("deleted file mode"):
            deleted.add(current)
    return touched, created, deleted


class PendingPackage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    package_id: str
    created: str
    kind: Literal["issue", "code_health"]
    repo_slug: str
    issue_number: int | None
    issue_url: str | None
    default_branch: str
    base_sha: str
    head_branch: str
    title: str
    body: str
    diff: str
    files: dict[str, str]  # full new contents of every changed (non-deleted) file
    deleted: list[str]
    run_id: str
    trace_path: str
    variant: str
    model: str
    cost_usd: float

    def content_sha256(self) -> str:
        bound = {
            k: getattr(self, k)
            for k in (
                "kind", "repo_slug", "issue_number", "default_branch", "base_sha",
                "head_branch", "title", "body", "diff", "files", "deleted",
            )
        }  # fmt: skip
        return hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()

    def final_body(self, reviewer: str) -> str:
        return self.body.replace(REVIEWER_PLACEHOLDER, reviewer)


def package_invariants(pkg: PendingPackage) -> list[str]:
    """Structural rules every package must satisfy (also re-checked at publish time)."""
    problems = []
    if not pkg.title.startswith(TITLE_PREFIX):
        problems.append(f"title must start with {TITLE_PREFIX!r}")
    if not pkg.body.startswith(DISCLOSURE_MARKER):
        problems.append("body must open with the AI-generated disclosure")
    if pkg.body.count(REVIEWER_PLACEHOLDER) != 1:
        problems.append("body must contain the reviewer placeholder exactly once")
    if not pkg.diff.strip():
        problems.append("empty diff")
    if not pkg.head_branch.startswith(HEAD_BRANCH_PREFIX):
        problems.append(f"head branch must start with {HEAD_BRANCH_PREFIX!r}")
    if pkg.head_branch == pkg.default_branch:
        problems.append("head branch must never be the default branch")
    touched, _, deleted = diff_files(pkg.diff)
    if set(pkg.files) | set(pkg.deleted) != touched or set(pkg.deleted) != deleted:
        problems.append("file contents do not match the diff")
    return problems


# --------------------------------------------------------------------------- rendering


def _clip(text: str, limit: int = _MAX_OUTPUT) -> str:
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]


def _diffstat(diff: str) -> list[str]:
    stats: dict[str, list[int]] = {}
    current = None
    for line in diff.splitlines():
        if m := _DIFF_HEADER.match(line):
            current = m.group(1)
            stats[current] = [0, 0]
        elif current and line.startswith("+") and not line.startswith("+++"):
            stats[current][0] += 1
        elif current and line.startswith("-") and not line.startswith("---"):
            stats[current][1] += 1
    return [f"- `{p}` (+{a} / -{d})" for p, (a, d) in stats.items()]


def _validation_line(v: ValidationResult) -> str:
    verdict = "passed" if v.get("passed") else "failed"
    detail = f"exit code {v['exit_code']}" if v["applied"] else "patch did not apply"
    return f"- **{v['stage']}** run on candidate #{v['candidate_index']}: {verdict} ({detail})"


def render_body(state: AgentState, config: ExperimentConfig, instance: Instance) -> str:
    origin = instance.origin
    if origin is None:
        raise PackageError("instance has no origin")
    idx = state["selected_index"]
    if idx is None:
        raise PackageError("no selected candidate")
    cand = state["candidates"][idx]
    notes: list[CriticNote] = state["critic_notes"]
    selection = next(
        (n for n in notes if n["kind"] == "selection" and n["candidate_index"] == idx), None
    )
    reflections = [n for n in notes if n["kind"] == "reflection"]
    final = [v for v in state["validation_results"] if v["stage"] == "final"]
    critic_runs = [v for v in state["validation_results"] if v["stage"] == "critic"]
    same_round = [c for c in state["candidates"] if c["attempt"] == cand["attempt"]]

    source = (
        f"Resolves #{origin.issue_number} ({origin.issue_url})"
        if origin.kind == "issue"
        else "No linked issue: this change was found by static analysis (code-health scan)."
    )
    lines = [
        DISCLOSURE_MARKER,
        "> [!NOTE]",
        "> **This is an AI-generated patch**, opened as a draft by an automated research "
        "system (a final-year project studying LLM agents for GitHub issue resolution). "
        "A human reviewed it before submission. Please treat it like any outside "
        "contribution: you are welcome to close it, and no reply is expected.",
        "",
        source,
        "",
        "## Summary (automated architectural write-up)",
        "",
        f"**Approach:** {cand['rationale'].strip()}",
        "",
        "**Files examined:** " + ", ".join(f"`{f}`" for f in state["localized_files"]),
        "",
        "**Files changed:**",
        *_diffstat(cand["diff"] or ""),
        "",
        "## How this candidate was chosen",
        "",
        f"- Pipeline variant **{config.variant}**: {len(same_round)} candidate(s) in the "
        f"final round (K={config.K}); {state['attempt_count']} round(s) in total "
        f"(reflection depth {state['reflection_depth']}).",
        f"- Selected candidate **#{idx}** (selected by: `{cand['selected_by']}`).",
    ]
    if selection is not None:
        scores = ", ".join(f"{k} {v:g}/5" for k, v in selection["scores"].items())
        if scores:
            lines.append(f"- Critic rubric scores: {scores} (risk: higher is riskier).")
        lines.append(f"- Critic's reasoning: {selection['critique'].strip()}")
    for r in reflections:
        lines.append(f"- Reflection after round {r['attempt']}: {_clip(r['critique'], 400)}")
    lines += [
        "",
        "## Validation",
        "",
        "Only the repository's **existing** tests were run, in an isolated sandbox "
        "without network access. No new tests were written.",
        "",
        *[_validation_line(v) for v in critic_runs + final],
        "",
        f"Command: `{final[-1]['command'] if final else instance.test_command}`",
        "",
        "## Disclosure",
        "",
        f"- Generated by `{config.model}` (effort `{config.effort}`) via an "
        f"Architect/Critic pipeline; total model cost ${state['cost_so_far']:.2f}.",
        f"- Human reviewer who approved submission: {REVIEWER_PLACEHOLDER}.",
        "- If this repository does not want AI-generated contributions, please close this PR; "
        "it will not be resubmitted.",
    ]
    return "\n".join(lines) + "\n"


def make_package(
    state: AgentState,
    config: ExperimentConfig,
    instance: Instance,
    *,
    files: dict[str, str],
    deleted: list[str],
    run_id: str,
    trace_path: str,
) -> PendingPackage:
    origin = instance.origin
    if origin is None:
        raise PackageError("publisher requires an instance with an origin (repo/issue)")
    idx = state["selected_index"]
    if idx is None:
        raise PackageError("no selected candidate")
    diff = state["candidates"][idx]["diff"] or ""
    short = run_id.rsplit("-", 1)[-1]
    if origin.kind == "issue":
        head = f"{HEAD_BRANCH_PREFIX}issue-{origin.issue_number}-{short}"
        package_id = f"{origin.repo_slug.replace('/', '__')}-issue{origin.issue_number}-{short}"
    else:
        head = f"{HEAD_BRANCH_PREFIX}code-health-{short}"
        package_id = f"{origin.repo_slug.replace('/', '__')}-codehealth-{short}"
    issue_title = state["issue_text"].strip().splitlines()[0] if state["issue_text"] else ""
    title = TITLE_PREFIX + issue_title[: _MAX_TITLE - len(TITLE_PREFIX)]
    pkg = PendingPackage(
        package_id=package_id,
        created=datetime.now(UTC).isoformat(),
        kind=origin.kind,
        repo_slug=origin.repo_slug,
        issue_number=origin.issue_number,
        issue_url=origin.issue_url,
        default_branch=origin.default_branch,
        base_sha=origin.base_sha,
        head_branch=head,
        title=title,
        body=render_body(state, config, instance),
        diff=diff,
        files=files,
        deleted=deleted,
        run_id=run_id,
        trace_path=trace_path,
        variant=config.variant,
        model=config.model,
        cost_usd=state["cost_so_far"],
    )
    problems = package_invariants(pkg)
    if problems:
        raise PackageError("invalid package: " + "; ".join(problems))
    return pkg


def write_package(pkg: PendingPackage, pending_dir: Path) -> Path:
    d = pending_dir / pkg.package_id
    d.mkdir(parents=True, exist_ok=False)  # never overwrite: approvals bind to contents
    (d / "package.json").write_text(pkg.model_dump_json(indent=2))
    (d / "title.txt").write_text(pkg.title + "\n")
    (d / "body.md").write_text(pkg.body)
    (d / "patch.diff").write_text(pkg.diff)
    (d / "SHA256").write_text(pkg.content_sha256() + "\n")
    return d


def load_package(path: Path) -> PendingPackage:
    return PendingPackage.model_validate_json((path / "package.json").read_text())


def summarize_for_trace(pkg: PendingPackage, path: Path) -> dict[str, Any]:
    return {"package_id": pkg.package_id, "path": str(path), "sha256": pkg.content_sha256()}
