"""Append-only log of published PRs (the source for caps, rate limits and outcomes)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PublishRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str
    package_sha256: str
    repo_slug: str
    issue_number: int | None
    pr_number: int
    pr_url: str
    head_branch: str
    reviewer: str
    published_at: str  # ISO 8601, UTC


def load_log(path: Path) -> list[PublishRecord]:
    if not path.exists():
        return []
    return [
        PublishRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_log(path: Path, record: PublishRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


def published_on(records: list[PublishRecord], day: datetime) -> int:
    d = day.astimezone(UTC).date()
    return sum(datetime.fromisoformat(r.published_at).astimezone(UTC).date() == d for r in records)


def seconds_since_last(records: list[PublishRecord], now: datetime) -> float | None:
    if not records:
        return None
    last = max(datetime.fromisoformat(r.published_at) for r in records)
    return (now - last).total_seconds()
