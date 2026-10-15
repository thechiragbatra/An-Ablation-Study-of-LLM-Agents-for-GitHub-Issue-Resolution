"""Attempt ONE real GitHub issue and, if the patch is sandbox-verified, write a LOCAL
pending PR package. Never publishes (see pilot.approve and pilot.publish).

    uv run python -m pilot.run_issue --repo owner/name --issue 123 \\
        --config configs/pilot/a3.yaml

The repository is shallow-cloned on the host; the agent's sandbox has no network.
The issue text is untrusted input to the LLM: the sandbox isolation and the mandatory
human approval before publishing are the mitigations.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from agent.config import ExperimentConfig
from agent.instance import Instance, IssueOrigin
from agent.run import run_instance
from pilot.config import PilotConfig
from pilot.gates import pre_run_gates
from pilot.github import GitHubClient, GitHubError, RestGitHub


class PilotError(Exception):
    pass


def clone(slug: str, branch: str, dest_root: Path) -> tuple[Path, str]:
    dest = dest_root / f"{slug.replace('/', '__')}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", "--branch", branch,
         f"https://github.com/{slug}.git", str(dest)],
        check=True, timeout=600,
    )  # fmt: skip
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True, check=True
    ).stdout.strip()
    return dest, sha


def build_instance(
    gh: GitHubClient,
    cfg: PilotConfig,
    slug: str,
    number: int,
    *,
    repo_dir: Path,
    base_sha: str,
    default_branch: str,
) -> Instance:
    issue = gh.get_issue(slug, number)
    entry = cfg.entry(slug)
    image = entry.image if entry else cfg.own_repo_defaults.image
    test_command = entry.test_command if entry else cfg.own_repo_defaults.test_command
    return Instance(
        instance_id=f"{slug.replace('/', '__')}-{number}",
        repo=str(repo_dir),
        base_commit=base_sha,
        issue_text=f"{issue.get('title', '').strip()}\n\n{(issue.get('body') or '').strip()}",
        test_command=test_command,
        image=image,
        env_setup=entry.env_setup if entry else None,
        origin=IssueOrigin(
            kind="issue", repo_slug=slug, issue_number=number,
            issue_url=str(issue.get("html_url", "")), default_branch=default_branch,
            base_sha=base_sha,
        ),
    )  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--issue", type=int, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/pilot/a3.yaml"))
    ap.add_argument("--pilot-config", type=Path, default=Path("configs/pilot/pilot.yaml"))
    args = ap.parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    if not config.publisher.enabled:
        ap.error(f"{args.config} must enable the publisher for pilot runs")
    cfg = PilotConfig.from_yaml(args.pilot_config)
    gh = RestGitHub.from_env()
    try:
        failed = [g for g in pre_run_gates(gh, cfg, args.repo, args.issue) if not g.ok]
        if failed:
            for g in failed:
                print(f"[FAIL] {g.name}: {g.reason}", file=sys.stderr)
            return 3
        default_branch = str(gh.get_repo(args.repo)["default_branch"])
        repo_dir, sha = clone(args.repo, default_branch, cfg.clone_dir)
        inst = build_instance(
            gh, cfg, args.repo, args.issue, repo_dir=repo_dir, base_sha=sha,
            default_branch=default_branch,
        )  # fmt: skip
    except (GitHubError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    outcome = run_instance(config, inst)
    status = outcome.final_state["status"]
    print(f"status={status} cost=${outcome.final_state['cost_so_far']:.2f} "
          f"trace={outcome.trace_path} valid={not outcome.trace_problems}")  # fmt: skip
    pending = sorted(config.publisher.pending_dir.glob(f"*{outcome.run_id.rsplit('-', 1)[-1]}"))
    if status == "sandbox_verified" and pending:
        print(f"pending package: {pending[0]}\nnext: python -m pilot.approve {pending[0]} "
              "--reviewer NAME, then python -m pilot.publish (dry run first)")  # fmt: skip
    else:
        print("no package written (only sandbox-verified patches are packaged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
