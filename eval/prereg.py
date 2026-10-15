"""Preregistration gate. The harness refuses to run the FROZEN subset unless
docs/PREREGISTRATION.md exists, is committed (tracked and unmodified), has every
``TODO(prereg)`` placeholder filled in, and the whole working tree is clean, so every
frozen run is reproducible from a recorded commit.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

PREREG_RELPATH = Path("docs/PREREGISTRATION.md")
PLACEHOLDER = "TODO(prereg)"


class PreregistrationError(Exception):
    pass


@dataclass(frozen=True)
class PreregStatus:
    ok: bool
    problems: list[str] = field(default_factory=list)
    commit: str | None = None  # last commit that touched the preregistration
    sha256: str | None = None  # of the committed file contents


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False, timeout=30
    )


def check_preregistered(repo_root: Path, relpath: Path = PREREG_RELPATH) -> PreregStatus:
    path = repo_root / relpath
    problems: list[str] = []
    if not path.is_file():
        return PreregStatus(False, [f"{relpath} does not exist"])
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()

    if _git(repo_root, "rev-parse", "--verify", "HEAD").returncode != 0:
        problems.append("repository has no commits")
    if _git(repo_root, "ls-files", "--error-unmatch", str(relpath)).returncode != 0:
        problems.append(f"{relpath} is not committed (untracked)")
    elif _git(repo_root, "status", "--porcelain", "--", str(relpath)).stdout.strip():
        problems.append(f"{relpath} has uncommitted changes")
    n_todo = text.count(PLACEHOLDER)
    if n_todo:
        problems.append(f"{relpath} still has {n_todo} unfilled {PLACEHOLDER} placeholder(s)")
    dirty = _git(repo_root, "status", "--porcelain").stdout.strip()
    if dirty:
        n = len(dirty.splitlines())
        problems.append(f"working tree is not clean ({n} changed/untracked path(s))")

    commit = None
    if not problems:
        commit = _git(repo_root, "log", "-n1", "--format=%H", "--", str(relpath)).stdout.strip()
    return PreregStatus(not problems, problems, commit or None, digest)


def require_preregistered(repo_root: Path) -> PreregStatus:
    status = check_preregistered(repo_root)
    if not status.ok:
        raise PreregistrationError(
            "refusing to run the frozen subset:\n  - " + "\n  - ".join(status.problems)
        )
    return status
