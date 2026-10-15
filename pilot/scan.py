"""OPTIONAL code-health mode, OFF by default. Requires BOTH ``code_health_scan: true`` in
the pilot config AND the ``--enable-code-health-scan`` flag.

Runs a static analyser limited to likely logical bugs (pyflakes F*, bugbear B*, syntax
E9*) inside the offline sandbox, turns the first finding group into a synthetic task and
runs the agent on it; a verified patch becomes a pending package like any other. There
is no ground truth for these tasks, so they are NOT part of the benchmark: eval/ only
reads the frozen SWE-bench files and pilot results are reported separately.

    uv run python -m pilot.scan --repo owner/name --enable-code-health-scan
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.config import ExperimentConfig
from agent.instance import Instance, IssueOrigin
from agent.run import run_instance
from agent.sandbox import DockerSandbox
from pilot.config import PilotConfig
from pilot.gates import gate_ai_policy, gate_allowlist
from pilot.github import GitHubError, RestGitHub
from pilot.run_issue import clone

RUFF_CMD = "ruff check --select F,B,E9 --output-format json --exit-zero ."


class ScanDisabled(Exception):
    pass


def require_enabled(cfg: PilotConfig, flag: bool) -> None:
    if not (cfg.code_health_scan and flag):
        raise ScanDisabled(
            "code-health scan is off: it needs code_health_scan: true in the pilot config "
            "AND --enable-code-health-scan"
        )


def group_findings(findings: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in findings:
        by_file[str(f["filename"])].append(f)
    return sorted(by_file.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def task_text(path: str, findings: list[dict[str, Any]]) -> str:
    lines = [f"Code health: likely bugs reported by static analysis in `{path}`", ""]
    for f in findings[:10]:
        loc = f.get("location", {})
        lines.append(f"- {f['code']} line {loc.get('row')}: {f['message']}")
    lines += ["", "Fix the underlying problems without changing intended behaviour."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/pilot/a3.yaml"))
    ap.add_argument("--pilot-config", type=Path, default=Path("configs/pilot/pilot.yaml"))
    ap.add_argument("--enable-code-health-scan", action="store_true")
    args = ap.parse_args(argv)
    cfg = PilotConfig.from_yaml(args.pilot_config)
    try:
        require_enabled(cfg, args.enable_code_health_scan)
    except ScanDisabled as exc:
        print(exc, file=sys.stderr)
        return 3
    config = ExperimentConfig.from_yaml(args.config)
    gh = RestGitHub.from_env()
    try:
        gates = (gate_allowlist(gh, cfg, args.repo), gate_ai_policy(gh, cfg, args.repo))
        failed = [g for g in gates if not g.ok]
        if failed:
            for g in failed:
                print(f"[FAIL] {g.name}: {g.reason}", file=sys.stderr)
            return 3
        default_branch = str(gh.get_repo(args.repo)["default_branch"])
    except GitHubError as exc:
        print(f"GitHub error: {exc}", file=sys.stderr)
        return 4
    repo_dir, sha = clone(args.repo, default_branch, cfg.clone_dir)
    entry = cfg.entry(args.repo)
    image = entry.image if entry else cfg.own_repo_defaults.image
    with DockerSandbox(image) as sb:
        sb.create(str(repo_dir), sha)
        res = sb.run(RUFF_CMD, 300)
    if res.exit_code != 0:
        print(f"static analysis failed (is ruff installed in {image}?): {res.stderr[:300]}")
        return 4
    groups = group_findings(json.loads(res.stdout or "[]"))
    if not groups:
        print("no findings")
        return 0
    path, findings = groups[0]
    rel = str(Path(path).relative_to("/workspace")) if path.startswith("/") else path
    inst = Instance(
        instance_id=f"{args.repo.replace('/', '__')}-codehealth",
        repo=str(repo_dir), base_commit=sha, issue_text=task_text(rel, findings),
        test_command=entry.test_command if entry else cfg.own_repo_defaults.test_command,
        image=image,
        origin=IssueOrigin(kind="code_health", repo_slug=args.repo,
                           default_branch=default_branch, base_sha=sha),
    )  # fmt: skip
    outcome = run_instance(config, inst)
    print(f"status={outcome.final_state['status']} trace={outcome.trace_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
