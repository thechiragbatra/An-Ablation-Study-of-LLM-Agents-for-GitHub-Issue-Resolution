"""Abstract sandbox: an isolated working copy of a repo at a commit, plus command execution.

Every ``run`` is reported to an optional observer (the tracer) so sandbox activity is
always part of the trace.
"""

from __future__ import annotations

import posixpath
import shlex
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Self

_PATCH_PATH = ".git/agent_patch.diff"

ExecObserver = Callable[[str, int | None, float, bool], None]
"""(cmd, exit_code, duration_ms, timed_out) -> None"""


class SandboxError(Exception):
    pass


@dataclass(frozen=True)
class ExecResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def safe_relpath(path: str) -> str:
    """Normalize a repo-relative path and reject anything escaping the workspace."""
    if not path or path.startswith("/") or "\x00" in path:
        raise SandboxError(f"invalid repo-relative path: {path!r}")
    norm = posixpath.normpath(path)
    if norm == "." or norm.startswith("../") or norm == "..":
        raise SandboxError(f"path escapes workspace: {path!r}")
    return norm


class Sandbox(ABC):
    def __init__(self, observer: ExecObserver | None = None) -> None:
        self.observer = observer
        self.base_commit: str | None = None

    @abstractmethod
    def create(self, repo: str, base_commit: str) -> str:
        """Provision the sandbox with ``repo`` checked out at ``base_commit``.

        Returns the resolved full commit sha."""

    @abstractmethod
    def write(self, path: str, content: str) -> None:
        """Write a UTF-8 file at a repo-relative path (parents created)."""

    @abstractmethod
    def run(self, cmd: str, timeout_s: int) -> ExecResult:
        """Run a shell command in the repo root with a hard wall-clock timeout."""

    @abstractmethod
    def read_diff(self) -> str:
        """Unified diff of the working tree (including new files) against the base commit."""

    @abstractmethod
    def destroy(self) -> None:
        """Release all resources. Must be idempotent."""

    # Convenience built on the primitives above (so it is traced like any other run).
    def read_file(self, path: str, timeout_s: int = 30) -> str:
        res = self.run(f"cat -- {shlex.quote(safe_relpath(path))}", timeout_s)
        if not res.ok:
            raise SandboxError(f"cannot read {path!r}: {res.stderr.strip()[:200]}")
        return res.stdout

    def read_base_file(self, path: str, timeout_s: int = 30) -> str:
        """Read a file as of the base commit, independent of any applied patch."""
        if self.base_commit is None:
            raise SandboxError("sandbox not created")
        spec = shlex.quote(f"{self.base_commit}:{safe_relpath(path)}")
        res = self.run(f"git show {spec}", timeout_s)
        if not res.ok:
            raise SandboxError(f"cannot read {path!r} at base: {res.stderr.strip()[:200]}")
        return res.stdout

    def apply_diff(self, diff: str, timeout_s: int = 60) -> ExecResult:
        """``git apply`` a unified diff to the working tree (as SWE-bench scoring does).

        The patch file lives under ``.git/`` so it is never part of the diff or cleaned."""
        self.write(_PATCH_PATH, diff)
        return self.run(f"git apply --whitespace=nowarn {_PATCH_PATH}", timeout_s)

    def reset_to_base(self, timeout_s: int = 60) -> None:
        if self.base_commit is None:
            raise SandboxError("sandbox not created")
        res = self.run(
            # -fd, not -fdx: ignored files include build artifacts and editable-install
            # metadata that the environment needs (e.g. compiled extensions).
            f"git reset -q --hard {shlex.quote(self.base_commit)} && git clean -q -fd",
            timeout_s,
        )
        if not res.ok:
            raise SandboxError(f"reset failed: {res.stderr.strip()[:200]}")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.destroy()
