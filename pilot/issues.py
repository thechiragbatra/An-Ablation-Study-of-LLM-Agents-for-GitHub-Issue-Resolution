"""Issue-driven input: open, unassigned, unlocked issues labelled e.g. "good first issue" or
"help wanted" in allowlisted repositories.

    uv run python -m pilot.issues [--repo owner/name ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pilot.config import PilotConfig
from pilot.github import GitHubClient, GitHubError, RestGitHub


def candidate_issues(gh: GitHubClient, cfg: PilotConfig, slugs: list[str]) -> list[dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for slug in slugs:
        for label in cfg.labels:
            for issue in gh.list_issues(slug, label):
                if "pull_request" in issue or issue.get("locked") or issue.get("assignees"):
                    continue
                out[(slug, int(issue["number"]))] = {
                    "repo": slug,
                    "number": int(issue["number"]),
                    "title": issue.get("title", ""),
                    "url": issue.get("html_url", ""),
                    "labels": sorted(lb["name"] for lb in issue.get("labels", [])),
                    "comments": int(issue.get("comments", 0)),
                }
    return sorted(out.values(), key=lambda i: (i["repo"], i["number"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--repo", action="append", default=[], help="own repo/fork to include")
    ap.add_argument("--pilot-config", type=Path, default=Path("configs/pilot/pilot.yaml"))
    args = ap.parse_args(argv)
    cfg = PilotConfig.from_yaml(args.pilot_config)
    slugs = sorted({e.repo for e in cfg.allowlist} | set(args.repo))
    if not slugs:
        print("no repositories: add some to the allowlist or pass --repo", file=sys.stderr)
        return 2
    try:
        issues = candidate_issues(RestGitHub.from_env(), cfg, slugs)
    except GitHubError as exc:
        print(f"GitHub error: {exc}", file=sys.stderr)
        return 4
    for i in issues:
        print(f"{i['repo']}#{i['number']}  [{', '.join(i['labels'])}]  {i['title']}  {i['url']}")
    print(f"{len(issues)} candidate issue(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
