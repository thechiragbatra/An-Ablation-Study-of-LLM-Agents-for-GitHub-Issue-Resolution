"""Docker-backed sandbox. No network by default, bounded memory/CPU/PIDs, hard timeouts."""

from __future__ import annotations

import contextlib
import io
import shlex
import tarfile
import time
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, NotFound

from agent.sandbox.base import ExecObserver, ExecResult, Sandbox, SandboxError, safe_relpath

LABEL = "ablation.sandbox"
WORKDIR = "/workspace"
_KILL_EXIT_CODES = (124, 137)  # coreutils timeout: TERM -> 124, KILL -> 128+9
_GIT_ID = "-c user.email=sandbox@localhost -c user.name=sandbox -c commit.gpgsign=false"
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv"})


def _skip_junk(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(p in _SKIP_DIRS for p in parts) or info.name.endswith(".pyc"):
        return None
    return info


def _tar_bytes(files: dict[str, bytes | Path], mode: int = 0o644) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for arcname, src in files.items():
            if isinstance(src, Path):
                tar.add(src, arcname=arcname, filter=_skip_junk)
            else:
                info = tarfile.TarInfo(arcname)
                info.size = len(src)
                info.mode = mode
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(src))
    return buf.getvalue()


class DockerSandbox(Sandbox):
    def __init__(
        self,
        image: str,
        *,
        mem_limit: str = "2g",
        cpus: float = 2.0,
        network: bool = False,
        observer: ExecObserver | None = None,
        run_label: str = "adhoc",
        client: Any | None = None,
        workdir: str = WORKDIR,
        env_setup: str | None = None,
        setup_script: str | None = None,
    ) -> None:
        super().__init__(observer)
        self.image = image
        self.workdir = workdir
        self.env_setup = env_setup
        self.setup_script = setup_script
        self.mem_limit = mem_limit
        self.cpus = cpus
        self.network = network
        self.run_label = run_label
        self._client: Any = client or docker.from_env()
        self._container: Any | None = None

    # -- lifecycle ---------------------------------------------------------

    def create(self, repo: str, base_commit: str) -> str:
        if self._container is not None:
            raise SandboxError("sandbox already created")
        src = Path(repo)
        prebuilt = not src.is_dir()  # SWE-bench: repo already lives at workdir in the image

        self._container = self._client.containers.run(
            self.image,
            command=["sleep", "infinity"],
            detach=True,
            working_dir=self.workdir,
            network_mode="bridge" if self.network else "none",
            mem_limit=self.mem_limit,
            memswap_limit=self.mem_limit,
            nano_cpus=int(self.cpus * 1e9),
            pids_limit=512,
            labels={LABEL: "1", f"{LABEL}.run": self.run_label},
        )
        try:
            self._exec_checked("git config --global --add safe.directory '*'", 30)
            if prebuilt:
                if not self.run("test -d .git", 30).ok:
                    raise SandboxError(f"image {self.image} has no git repo at {self.workdir}")
            else:
                self._exec_checked(f"mkdir -p {shlex.quote(self.workdir)}", 30)
                self._container.put_archive(self.workdir, _tar_bytes({".": src}))
                if not self.run("test -d .git", 30).ok:
                    self._exec_checked(
                        f"git init -q && git add -A && git {_GIT_ID} commit -q -m base", 60
                    )
            self._exec_checked(f"git checkout -q --detach {shlex.quote(base_commit)}", 60)
            # Untracked files shipped with the environment (build outputs not covered by
            # .gitignore) must never show up in candidate diffs: exclude them locally.
            self._exec_checked("git ls-files --others --exclude-standard >> .git/info/exclude", 60)
            if self.setup_script:
                self._exec_checked(self.setup_script, 600)
            sha = self._exec_checked("git rev-parse HEAD", 30).stdout.strip()
        except BaseException:
            self.destroy()
            raise
        self.base_commit = sha
        return sha

    def destroy(self) -> None:
        container, self._container = self._container, None
        if container is None:
            return
        with contextlib.suppress(NotFound):
            container.remove(force=True)

    # -- primitives --------------------------------------------------------

    def write(self, path: str, content: str) -> None:
        rel = safe_relpath(path)
        parent = str(Path(rel).parent)
        if parent != ".":
            self._exec_checked(f"mkdir -p -- {shlex.quote(parent)}", 30)
        ok = self._require().put_archive(self.workdir, _tar_bytes({rel: content.encode("utf-8")}))
        if not ok:
            raise SandboxError(f"failed to write {path!r}")

    def run(self, cmd: str, timeout_s: int) -> ExecResult:
        container = self._require()
        script = f"{self.env_setup}\n{cmd}" if self.env_setup else cmd
        wrapped = ["timeout", "-s", "KILL", str(int(timeout_s)), "bash", "-c", script]
        t0 = time.perf_counter()
        try:
            exit_code, output = container.exec_run(wrapped, workdir=self.workdir, demux=True)
        except APIError as exc:
            duration = time.perf_counter() - t0
            self._observe(cmd, None, duration, timed_out=False)
            raise SandboxError(f"docker exec failed: {exc}") from exc
        duration = time.perf_counter() - t0
        stdout_b, stderr_b = output if output else (None, None)
        # Exit 137 can also mean OOM-kill; only call it a timeout if the clock agrees.
        timed_out = exit_code in _KILL_EXIT_CODES and duration >= timeout_s - 0.5
        self._observe(cmd, exit_code, duration, timed_out)
        return ExecResult(
            exit_code=exit_code,
            stdout=(stdout_b or b"").decode("utf-8", "replace"),
            stderr=(stderr_b or b"").decode("utf-8", "replace"),
            duration_s=duration,
            timed_out=timed_out,
        )

    def read_diff(self) -> str:
        if self.base_commit is None:
            raise SandboxError("sandbox not created")
        # intent-to-add makes untracked files show up in the diff without staging content
        res = self.run(
            f"git add -A -N && git -c core.quotepath=off diff --no-color --no-ext-diff "
            f"{shlex.quote(self.base_commit)}",
            60,
        )
        if not res.ok:
            raise SandboxError(f"diff failed: {res.stderr.strip()[:200]}")
        return res.stdout

    # -- helpers -----------------------------------------------------------

    @classmethod
    def cleanup_orphans(cls, run_label: str | None = None, client: Any | None = None) -> int:
        cli: Any = client or docker.from_env()
        label = f"{LABEL}.run={run_label}" if run_label else LABEL
        removed = 0
        for c in cli.containers.list(all=True, filters={"label": label}):
            c.remove(force=True)
            removed += 1
        return removed

    def _require(self) -> Any:
        if self._container is None:
            raise SandboxError("sandbox not created (or already destroyed)")
        return self._container

    def _exec_checked(self, cmd: str, timeout_s: int) -> ExecResult:
        res = self.run(cmd, timeout_s)
        if not res.ok:
            raise SandboxError(
                f"command failed ({res.exit_code}): {cmd[:120]}: {res.stderr.strip()[:300]}"
            )
        return res

    def _observe(self, cmd: str, exit_code: int | None, duration_s: float, timed_out: bool) -> None:
        if self.observer is not None:
            self.observer(cmd, exit_code, duration_s * 1000, timed_out)


def build_image(
    context_dir: Path, tag: str, client: Any | None = None, dockerfile: str | None = None
) -> str:
    """Build (or reuse the cache for) a sandbox image from a local Dockerfile."""
    cli: Any = client or docker.from_env()
    image, _logs = cli.images.build(path=str(context_dir), tag=tag, rm=True, dockerfile=dockerfile)
    return str(image.id)
