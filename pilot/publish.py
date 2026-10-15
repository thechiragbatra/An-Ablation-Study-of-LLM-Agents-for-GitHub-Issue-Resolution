"""Turn ONE approved pending package into a DRAFT pull request.

    uv run python -m pilot.publish out/pending/<package_id>             # dry run (default)
    uv run python -m pilot.publish out/pending/<package_id> --publish   # real, gated

Dry run (default): evaluates every gate and prints the exact GitHub calls it would make;
it performs no GitHub writes. ``--publish`` additionally requires a human approval file
matching the package contents, GITHUB_TOKEN, and every gate passing. Commits go to a fresh
``ai-patch/...`` branch on your fork (or your own repo), never to a default branch; the
pull request is always a draft against the upstream default branch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.publishing import PendingPackage, load_package
from pilot.config import PilotConfig
from pilot.gates import GateResult, publish_gates, read_approval
from pilot.github import GitHubClient, GitHubError, RestGitHub
from pilot.ledger import PublishRecord, append_log, load_log

PUBLISHED_FILE = "PUBLISHED.json"


class PublishRefused(Exception):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__("refusing to publish:\n  - " + "\n  - ".join(reasons))
        self.reasons = reasons


@dataclass
class PublishResult:
    dry_run: bool
    gates: list[GateResult]
    planned: list[str]
    pr: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def would_pass(self) -> bool:
        return all(g.ok for g in self.gates)


def planned_calls(pkg: PendingPackage, own_repo: bool) -> list[str]:
    target = pkg.repo_slug if own_repo else "<your fork>"
    calls = [] if own_repo else [f"POST /repos/{pkg.repo_slug}/forks (reuse if it exists)"]
    calls += [
        f"POST /repos/{target}/git/blobs  x{len(pkg.files)}",
        f"POST /repos/{target}/git/trees  (base {pkg.base_sha[:12]}, "
        f"{len(pkg.files)} changed, {len(pkg.deleted)} deleted)",
        f"POST /repos/{target}/git/commits  (parent {pkg.base_sha[:12]})",
        f"POST /repos/{target}/git/refs  refs/heads/{pkg.head_branch}",
        f"POST /repos/{pkg.repo_slug}/pulls  draft=true base={pkg.default_branch} "
        f"title={pkg.title!r}",
    ]
    return calls


def publish_package(
    pkg_dir: Path,
    *,
    gh: GitHubClient,
    cfg: PilotConfig,
    publish: bool,
    published_this_run: int = 0,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
    fork_ready_attempts: int = 10,
) -> PublishResult:
    pkg = load_package(pkg_dir)
    if (pkg_dir / PUBLISHED_FILE).exists():
        raise PublishRefused([f"already published: {(pkg_dir / PUBLISHED_FILE).read_text()}"])
    log = load_log(cfg.log_path)
    gates = publish_gates(
        gh, cfg, pkg_dir, pkg, log=log, now=now(), published_this_run=published_this_run
    )
    login = gh.user_login()
    upstream = gh.get_repo(pkg.repo_slug)
    own = login is not None and str(upstream["owner"]["login"]).lower() == login.lower()
    result = PublishResult(not publish, gates, planned_calls(pkg, own))
    if not publish:
        return result
    failed = [f"{g.name}: {g.reason}" for g in gates if not g.ok]
    if login is None:
        failed.append("GITHUB_TOKEN is required to publish")
    if failed:
        raise PublishRefused(failed)
    approval = read_approval(pkg_dir)
    assert approval is not None  # guaranteed by the approval gate
    assert login is not None
    target, head = _resolve_target(
        gh, pkg, upstream, login, sleep=sleep, attempts=fork_ready_attempts
    )
    _push_branch(gh, pkg, target)
    pr = gh.create_draft_pull(
        pkg.repo_slug, title=pkg.title, body=pkg.final_body(approval["approved_by"]),
        head=head, base=pkg.default_branch,
    )  # fmt: skip

    record = PublishRecord(
        package_id=pkg.package_id, package_sha256=pkg.content_sha256(),
        repo_slug=pkg.repo_slug, issue_number=pkg.issue_number, pr_number=int(pr["number"]),
        pr_url=str(pr["html_url"]), head_branch=pkg.head_branch,
        reviewer=approval["approved_by"], published_at=now().isoformat(),
    )  # fmt: skip
    append_log(cfg.log_path, record)
    (pkg_dir / PUBLISHED_FILE).write_text(record.model_dump_json(indent=2))
    result.pr = pr
    return result


def _resolve_target(
    gh: GitHubClient,
    pkg: PendingPackage,
    upstream: dict[str, Any],
    login: str,
    *,
    sleep: Callable[[float], None],
    attempts: int,
) -> tuple[str, str]:
    """(repo to push the branch to, PR head spec). Your own repo, or your fork."""
    if str(upstream["owner"]["login"]).lower() == login.lower():
        target, head, target_default = (
            pkg.repo_slug,
            pkg.head_branch,
            str(upstream["default_branch"]),
        )
    else:
        target = str(gh.create_fork(pkg.repo_slug)["full_name"])  # reuses an existing fork
        for _ in range(attempts):  # forks are created asynchronously
            try:
                fork = gh.get_repo(target)
                break
            except GitHubError:
                sleep(3)
        else:
            raise PublishRefused([f"fork {target} did not become ready"])
        if str(fork["owner"]["login"]).lower() != login.lower():
            raise PublishRefused([f"fork {target} is not owned by {login}"])
        head, target_default = f"{login}:{pkg.head_branch}", str(fork["default_branch"])
    if pkg.head_branch in (pkg.default_branch, target_default, str(upstream["default_branch"])):
        raise PublishRefused(["head branch would be a default branch"])
    if not gh.commit_exists(target, pkg.base_sha):
        gh.sync_fork(target, target_default)
        if not gh.commit_exists(target, pkg.base_sha):
            raise PublishRefused([f"base commit {pkg.base_sha[:12]} not found in {target}"])
    return target, head


def _push_branch(gh: GitHubClient, pkg: PendingPackage, target: str) -> None:
    """One commit (parent = base sha) on a NEW ai-patch/ branch; existing refs are never
    updated (creating an existing ref fails, which aborts the publish)."""
    base_tree = gh.get_commit_tree(target, pkg.base_sha)
    modes = gh.get_tree_modes(target, base_tree)
    entries: list[dict[str, Any]] = [
        {"path": p, "mode": modes.get(p, "100644"), "type": "blob",
         "sha": gh.create_blob(target, content)}
        for p, content in sorted(pkg.files.items())
    ]  # fmt: skip
    entries += [{"path": p, "mode": modes.get(p, "100644"), "type": "blob", "sha": None}
                for p in pkg.deleted]  # fmt: skip
    tree = gh.create_tree(target, base_tree, entries)
    message = f"{pkg.title}\n\nAI-generated patch; see the draft PR description for disclosure."
    commit = gh.create_commit(target, message, tree, pkg.base_sha)
    gh.create_ref(target, f"refs/heads/{pkg.head_branch}", commit)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("package_dir", type=Path)
    ap.add_argument("--publish", action="store_true", help="actually open the draft PR")
    ap.add_argument("--pilot-config", type=Path, default=Path("configs/pilot/pilot.yaml"))
    args = ap.parse_args(argv)
    cfg = PilotConfig.from_yaml(args.pilot_config)
    gh = RestGitHub.from_env()
    try:
        result = publish_package(args.package_dir, gh=gh, cfg=cfg, publish=args.publish)
    except PublishRefused as exc:
        print(exc, file=sys.stderr)
        return 3
    except GitHubError as exc:
        print(f"GitHub error (stopped, nothing retried): {exc}", file=sys.stderr)
        return 4
    print("DRY RUN: no GitHub writes" if result.dry_run else "PUBLISHED")
    for g in result.gates:
        print(f"  [{'pass' if g.ok else 'FAIL'}] {g.name}: {g.reason}")
    print("Planned GitHub calls:" if result.dry_run else "GitHub calls made:")
    for c in result.planned:
        print(f"  {c}")
    if result.dry_run:
        verdict = "would publish" if result.would_pass else "would be REFUSED"
        print(f"With --publish this package {verdict}.")
    elif result.pr is not None:
        print(json.dumps({"pr": result.pr.get("html_url"), "draft": result.pr.get("draft")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
