"""Track what happened to published PRs, for the report's results table.

    uv run python -m pilot.outcomes [--out out/pilot/pr_outcomes.csv]

Outcome per PR: merged | closed (unmerged) | open-active (human activity) |
open-waiting (no activity yet) | ignored (open, no human activity after N days).
Read-only: this command only GETs from GitHub.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pilot.config import PilotConfig
from pilot.github import GitHubClient, GitHubError, RestGitHub
from pilot.ledger import PublishRecord, load_log

FIELDS = (
    "repo", "pr_number", "pr_url", "issue_number", "published_at", "outcome", "days_open",
    "reviews", "review_comments", "issue_comments", "first_response_hours", "review_states",
)  # fmt: skip


def _ts(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _human(items: list[dict[str, Any]], own_login: str | None) -> list[dict[str, Any]]:
    def keep(i: dict[str, Any]) -> bool:
        user = i.get("user") or {}
        if user.get("type") == "Bot":
            return False
        return own_login is None or str(user.get("login", "")).lower() != own_login.lower()

    return [i for i in items if keep(i)]


def classify(
    rec: PublishRecord, pr: dict[str, Any], reviews: list[dict[str, Any]],
    review_comments: list[dict[str, Any]], issue_comments: list[dict[str, Any]],
    *, own_login: str | None, now: datetime, ignored_after_days: int,
) -> dict[str, Any]:  # fmt: skip
    published = datetime.fromisoformat(rec.published_at)
    reviews = _human(reviews, own_login)
    review_comments = _human(review_comments, own_login)
    issue_comments = _human(issue_comments, own_login)
    stamps = [
        _ts(x.get("submitted_at") or x.get("created_at"))
        for x in reviews + review_comments + issue_comments
        if x.get("submitted_at") or x.get("created_at")
    ]
    first = min(stamps) if stamps else None
    end = pr.get("merged_at") or pr.get("closed_at")
    end_dt = _ts(end) if end else now
    days_open = (end_dt - published).total_seconds() / 86400
    activity = bool(stamps)
    if pr.get("merged_at"):
        outcome = "merged"
    elif pr.get("state") == "closed":
        outcome = "closed"
    elif activity:
        outcome = "open-active"
    elif days_open >= ignored_after_days:
        outcome = "ignored"
    else:
        outcome = "open-waiting"
    return {
        "repo": rec.repo_slug, "pr_number": rec.pr_number, "pr_url": rec.pr_url,
        "issue_number": rec.issue_number, "published_at": rec.published_at,
        "outcome": outcome, "days_open": round(days_open, 1), "reviews": len(reviews),
        "review_comments": len(review_comments), "issue_comments": len(issue_comments),
        "first_response_hours": (
            round((first - published).total_seconds() / 3600, 1) if first else ""
        ),
        "review_states": ",".join(sorted({str(r.get("state")) for r in reviews})),
    }  # fmt: skip


def sync(gh: GitHubClient, cfg: PilotConfig, now: datetime) -> list[dict[str, Any]]:
    login = gh.user_login()
    rows = []
    for rec in load_log(cfg.log_path):
        n, slug = rec.pr_number, rec.repo_slug
        rows.append(classify(
            rec, gh.get_pull(slug, n), gh.list_reviews(slug, n), gh.list_review_comments(slug, n),
            gh.list_issue_comments(slug, n), own_login=login, now=now,
            ignored_after_days=cfg.ignored_after_days,
        ))  # fmt: skip
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    counts = Counter(r["outcome"] for r in rows)
    lines = ["| Outcome | PRs |", "|---|---|"]
    for k in ("merged", "closed", "open-active", "open-waiting", "ignored"):
        lines.append(f"| {k} | {counts.get(k, 0)} |")
    lines.append(f"| **total** | {len(rows)} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--pilot-config", type=Path, default=Path("configs/pilot/pilot.yaml"))
    ap.add_argument("--out", type=Path, default=Path("out/pilot/pr_outcomes.csv"))
    args = ap.parse_args(argv)
    cfg = PilotConfig.from_yaml(args.pilot_config)
    try:
        rows = sync(RestGitHub.from_env(), cfg, datetime.now(UTC))
    except GitHubError as exc:
        print(f"GitHub error: {exc}", file=sys.stderr)
        return 4
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    md = markdown(rows)
    args.out.with_suffix(".md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
