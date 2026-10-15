"""A task instance: one issue against one repo at one commit."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class IssueOrigin(BaseModel):
    """Where a live-pilot task came from. Package metadata only: never shown to the LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["issue", "code_health"] = "issue"
    repo_slug: str  # owner/name on GitHub
    issue_number: int | None = None
    issue_url: str | None = None
    default_branch: str
    base_sha: str


class Instance(BaseModel):
    """Agent-visible task description.

    ``repo`` is a local directory (fixtures, copied into the sandbox) or ``owner/name``
    (SWE-bench: the repo is already at ``workdir`` inside ``image``).
    ``test_command`` is the validation the agent may run; gold tests live elsewhere.
    ``env_setup`` is prepended to every sandbox command (e.g. conda activation);
    ``setup_script`` runs once when the sandbox is created (e.g. locale generation).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    repo: str
    base_commit: str
    issue_text: str
    test_command: str
    image: str | None = None
    workdir: str = "/workspace"
    env_setup: str | None = None
    setup_script: str | None = None
    origin: IssueOrigin | None = None

    @classmethod
    def from_json(cls, path: Path) -> Instance:
        inst = cls.model_validate_json(path.read_text())
        repo_path = Path(inst.repo)
        if not repo_path.is_absolute() and (path.parent / repo_path).exists():
            # Fixture repos are referenced relative to the instance file.
            return inst.model_copy(update={"repo": str((path.parent / repo_path).resolve())})
        return inst
