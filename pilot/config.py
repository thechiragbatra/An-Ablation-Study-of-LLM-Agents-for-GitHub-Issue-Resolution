"""Live-pilot settings (configs/pilot/pilot.yaml). Conservative by default."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RepoEntry(_Strict):
    """An explicitly allowlisted repository and how to run its tests offline."""

    repo: str  # owner/name
    image: str  # Docker image with the repo's dependencies preinstalled
    test_command: str
    env_setup: str | None = None
    # Set to true only after a human has read the repo's contribution / AI policy.
    ai_policy_reviewed: bool = False

    @field_validator("repo")
    @classmethod
    def _slug(cls, v: str) -> str:
        if v.count("/") != 1 or not all(v.split("/")):
            raise ValueError("repo must be 'owner/name'")
        return v


class RunDefaults(_Strict):
    """Execution settings for your own repos/forks that are not listed explicitly."""

    image: str = "ablation-fixture:py3.12"
    test_command: str = "python -m pytest -q -p no:cacheprovider"


class PilotConfig(_Strict):
    allowlist: list[RepoEntry] = []
    allow_own_repos: bool = True
    own_repo_defaults: RunDefaults = RunDefaults()
    labels: list[str] = ["good first issue", "help wanted"]
    max_prs_per_run: int = Field(default=1, ge=1, le=5)
    max_prs_per_day: int = Field(default=3, ge=1, le=10)
    min_seconds_between_prs: int = Field(default=120, ge=30)
    ignored_after_days: int = Field(default=14, ge=1)
    code_health_scan: bool = False
    pending_dir: Path = Path("out/pending")
    log_path: Path = Path("out/pilot/publish_log.jsonl")
    clone_dir: Path = Path("out/pilot/repos")

    def entry(self, slug: str) -> RepoEntry | None:
        return next((e for e in self.allowlist if e.repo.lower() == slug.lower()), None)

    @classmethod
    def from_yaml(cls, path: Path) -> PilotConfig:
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping")
        return cls.model_validate(data)
