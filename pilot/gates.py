"""Safety gates for the live pilot. Each returns a GateResult; publishing requires ALL to
pass. They run before an issue is attempted and again, fresh, at publish time."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent.publishing import HEAD_BRANCH_PREFIX, PendingPackage, package_invariants
from pilot.config import PilotConfig
from pilot.github import GitHubClient, GitHubError
from pilot.ledger import PublishRecord, published_on, seconds_since_last
from pilot.policy import POLICY_PATHS, assess

APPROVAL_FILE = "APPROVED.json"


@dataclass(frozen=True)
class GateResult:
    name: str
    ok: bool
    reason: str


def gate_allowlist(gh: GitHubClient, cfg: PilotConfig, slug: str) -> GateResult:
    try:
        repo = gh.get_repo(slug)
    except GitHubError as err:
        return GateResult("allowlist", False, f"cannot read repo: {err}")
    if repo.get("archived"):
        return GateResult("allowlist", False, "repository is archived")
    if cfg.entry(slug) is not None:
        return GateResult("allowlist", True, "explicitly allowlisted")
    if not cfg.allow_own_repos:
        return GateResult("allowlist", False, "not in the explicit allowlist")
    login = gh.user_login()
    if login is None:
        return GateResult("allowlist", False, "cannot verify ownership without GITHUB_TOKEN")
    owner = str(repo.get("owner", {}).get("login", ""))
    if owner.lower() == login.lower():
        return GateResult("allowlist", True, "own fork" if repo.get("fork") else "own repo")
    return GateResult("allowlist", False, f"not allowlisted and not owned by {login}")


def gate_ai_policy(gh: GitHubClient, cfg: PilotConfig, slug: str) -> GateResult:
    docs: dict[str, str] = {}
    try:
        for path in POLICY_PATHS:
            text = gh.get_file(slug, path)
            if text is not None:
                docs[path] = text
    except GitHubError as err:
        return GateResult("ai_policy", False, f"cannot read contribution policy: {err}")
    a = assess(docs)
    if a.verdict == "prohibited":
        return GateResult("ai_policy", False, f"{a.source} prohibits AI contributions: "
                          f"{a.evidence!r}")  # fmt: skip
    if a.verdict == "mentioned":
        entry = cfg.entry(slug)
        if entry is None or not entry.ai_policy_reviewed:
            return GateResult(
                "ai_policy", False,
                f"{a.source} mentions AI ({a.evidence!r}); a human must review it and set "
                "ai_policy_reviewed: true in the allowlist",
            )  # fmt: skip
        return GateResult("ai_policy", True, f"AI mentioned in {a.source}; reviewed by a human")
    if not docs:
        return GateResult("ai_policy", True, "no contribution/AI policy documents found")
    return GateResult("ai_policy", True, f"no AI mention in {', '.join(sorted(docs))}")


def gate_caps(
    cfg: PilotConfig, log: list[PublishRecord], now: datetime, published_this_run: int
) -> GateResult:
    if published_this_run >= cfg.max_prs_per_run:
        return GateResult("caps", False, f"max_prs_per_run={cfg.max_prs_per_run} reached")
    today = published_on(log, now)
    if today >= cfg.max_prs_per_day:
        return GateResult("caps", False, f"max_prs_per_day={cfg.max_prs_per_day} reached")
    return GateResult("caps", True, f"{today}/{cfg.max_prs_per_day} today")


def gate_rate(cfg: PilotConfig, log: list[PublishRecord], now: datetime) -> GateResult:
    since = seconds_since_last(log, now)
    if since is not None and since < cfg.min_seconds_between_prs:
        return GateResult(
            "rate", False, f"last PR {since:.0f}s ago (< {cfg.min_seconds_between_prs}s)"
        )
    return GateResult("rate", True, "ok")


def gate_issue(gh: GitHubClient, slug: str, issue_number: int | None) -> GateResult:
    if issue_number is None:
        return GateResult("issue", True, "no linked issue (code-health scan)")
    try:
        issue = gh.get_issue(slug, issue_number)
        pulls = gh.list_open_pulls(slug)
    except GitHubError as err:
        return GateResult("issue", False, f"cannot read issue: {err}")
    if "pull_request" in issue:
        return GateResult("issue", False, f"#{issue_number} is a pull request, not an issue")
    if issue.get("state") != "open":
        return GateResult("issue", False, f"issue #{issue_number} is {issue.get('state')}")
    if issue.get("locked"):
        return GateResult("issue", False, f"issue #{issue_number} is locked")
    ours = f"{HEAD_BRANCH_PREFIX}issue-{issue_number}-"
    for pr in pulls:
        if str(pr.get("head", {}).get("ref", "")).startswith(ours):
            return GateResult(
                "issue", False, f"an AI patch PR already exists: {pr.get('html_url')}"
            )
    return GateResult("issue", True, f"issue #{issue_number} is open")


def gate_package(pkg: PendingPackage) -> GateResult:
    problems = package_invariants(pkg)
    return GateResult("package", not problems, "; ".join(problems) or "invariants hold")


def read_approval(pkg_dir: Path) -> dict[str, str] | None:
    path = pkg_dir / APPROVAL_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {k: str(v) for k, v in data.items()}


def gate_approval(pkg_dir: Path, pkg: PendingPackage) -> GateResult:
    approval = read_approval(pkg_dir)
    if approval is None:
        return GateResult("approval", False, f"no {APPROVAL_FILE}: run `python -m pilot.approve`")
    if approval.get("package_sha256") != pkg.content_sha256():
        return GateResult("approval", False, "package changed after approval; re-approve it")
    if not approval.get("approved_by", "").strip():
        return GateResult("approval", False, "approval has no reviewer name")
    return GateResult("approval", True, f"approved by {approval['approved_by']}")


def pre_run_gates(
    gh: GitHubClient, cfg: PilotConfig, slug: str, issue: int | None
) -> list[GateResult]:
    """Fail fast before spending money on an issue we could never publish."""
    return [
        gate_allowlist(gh, cfg, slug),
        gate_ai_policy(gh, cfg, slug),
        gate_issue(gh, slug, issue),
    ]


def publish_gates(
    gh: GitHubClient,
    cfg: PilotConfig,
    pkg_dir: Path,
    pkg: PendingPackage,
    *,
    log: list[PublishRecord],
    now: datetime,
    published_this_run: int,
) -> list[GateResult]:
    return [
        gate_package(pkg),
        gate_approval(pkg_dir, pkg),
        gate_allowlist(gh, cfg, pkg.repo_slug),
        gate_ai_policy(gh, cfg, pkg.repo_slug),
        gate_issue(gh, pkg.repo_slug, pkg.issue_number),
        gate_caps(cfg, log, now, published_this_run),
        gate_rate(cfg, log, now),
    ]
