"""JSONL tracing. One file per run; one ``node`` record per node invocation.

A run is valid for analysis only if ``validate_trace`` returns no problems: it must open
with ``run_start``, close with ``run_end``, have contiguous ``seq`` numbers, and the
``run_end`` totals must equal the sum over node records (detects dropped lines).

Full LLM prompts/responses are written as side artifacts under ``<run_id>/llm/`` and
referenced from node records, so the trace itself stays compact.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent.config import REPO_ROOT, ExperimentConfig

SCHEMA_VERSION = 1
_TRACKED_PACKAGES = ("anthropic", "langgraph", "docker", "pydantic")
FINAL_STATE_FILE = "final_state.json"
_SUMMARY_STR_CHARS = 200
_CMD_CHARS = 500

_SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)((?:api[_-]?key|auth[_-]?token|secret)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"),
)


def redact(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


def canonical_hash(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


# --------------------------------------------------------------------------- schema


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    seq: int
    ts: str
    run_id: str
    instance_id: str
    variant: str
    seed: int
    config_hash: str


class LLMCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    latency_ms: float
    stop_reason: str | None
    request_id: str | None
    artifact: str | None


class SandboxExecRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd: str
    exit_code: int | None
    duration_ms: float
    timed_out: bool


class RunStart(_Record):
    event: Literal["run_start"] = "run_start"
    config: dict[str, Any]
    pricing: dict[str, Any]
    git_sha: str
    git_dirty: bool
    packages: dict[str, str]
    python: str


class NodeRecord(_Record):
    event: Literal["node"] = "node"
    node: str
    step: int
    attempt_count: int
    reflection_depth: int
    input_hash: str
    output_summary: dict[str, Any]
    llm_calls: list[LLMCallRecord]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cost_cumulative: float
    latency_ms: float
    sandbox_execs: list[SandboxExecRecord]
    status_after: str
    error: str | None


class RunTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    sandbox_runs: int = 0
    node_latency_ms: float = 0.0

    def add(self, rec: NodeRecord) -> None:
        self.nodes += 1
        self.llm_calls += len(rec.llm_calls)
        self.tokens_in += rec.tokens_in
        self.tokens_out += rec.tokens_out
        self.cost_usd += rec.cost_usd
        self.sandbox_runs += len(rec.sandbox_execs)
        self.node_latency_ms += rec.latency_ms


class RunEnd(_Record):
    event: Literal["run_end"] = "run_end"
    status: str
    attempts: int
    reflection_depth: int
    totals: RunTotals
    wall_time_s: float
    error: str | None
    # Path (relative to the trace dir) of the full final AgentState: candidate diffs,
    # critic notes, validation results. Optional so earlier v1 traces stay valid.
    final_state: str | None = None


TraceRecord = Annotated[RunStart | NodeRecord | RunEnd, Field(discriminator="event")]
_RECORD_ADAPTER: TypeAdapter[RunStart | NodeRecord | RunEnd] = TypeAdapter(TraceRecord)


# --------------------------------------------------------------------------- writer


class NodeScope:
    """Collects sub-events (LLM calls, sandbox execs) while one node runs."""

    def __init__(self, node: str, step: int, state: dict[str, Any]) -> None:
        self.node = node
        self.step = step
        self.attempt_count = int(state.get("attempt_count", 0))
        self.reflection_depth = int(state.get("reflection_depth", 0))
        self.input_hash = canonical_hash(state)
        self.llm_calls: list[LLMCallRecord] = []
        self.sandbox_execs: list[SandboxExecRecord] = []
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000


class Tracer:
    def __init__(
        self, trace_dir: Path, run_id: str, instance_id: str, config: ExperimentConfig
    ) -> None:
        self.run_id = run_id
        self.instance_id = instance_id
        self.config = config
        self._config_hash = config.config_hash()
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = trace_dir / f"{run_id}.jsonl"
        self.artifact_dir = trace_dir / run_id
        self._fh: IO[str] | None = None
        self._seq = 0
        self._step = 0
        self._scope: NodeScope | None = None
        self._totals = RunTotals()
        self._cost_cumulative = 0.0
        self._t0 = time.perf_counter()
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def start_run(self, pricing: dict[str, Any]) -> None:
        if self._fh is not None:
            raise RuntimeError("run already started")
        self._fh = self.path.open("x", encoding="utf-8")
        git_sha, git_dirty = _git_state()
        self._write(
            RunStart(
                **self._common(),
                config=self.config.model_dump(mode="json"),
                pricing=pricing,
                git_sha=git_sha,
                git_dirty=git_dirty,
                packages=_package_versions(),
                python=platform.python_version(),
            )
        )

    def end_run(
        self,
        status: str,
        attempts: int,
        reflection_depth: int,
        error: str | None,
        final_state: dict[str, Any] | None = None,
    ) -> None:
        if self._closed:
            return
        if self._scope is not None:
            raise RuntimeError(f"run ended inside node scope {self._scope.node!r}")
        state_ref = self._write_final_state(final_state) if final_state is not None else None
        self._write(
            RunEnd(
                **self._common(),
                status=status,
                attempts=attempts,
                reflection_depth=reflection_depth,
                totals=self._totals,
                wall_time_s=time.perf_counter() - self._t0,
                error=redact(error) if error else None,
                final_state=state_ref,
            )
        )
        assert self._fh is not None
        self._fh.close()
        self._closed = True

    # -- node scopes -------------------------------------------------------

    def begin_node(self, node: str, state: dict[str, Any]) -> NodeScope:
        if self._scope is not None:
            raise RuntimeError(f"node {node!r} started inside {self._scope.node!r}")
        self._scope = NodeScope(node, self._step, state)
        self._step += 1
        return self._scope

    def end_node(
        self,
        scope: NodeScope,
        update: dict[str, Any],
        status_after: str,
        error: str | None,
    ) -> NodeRecord:
        if scope is not self._scope:
            raise RuntimeError("end_node called with a scope that is not active")
        self._scope = None
        cost = sum(c.cost_usd for c in scope.llm_calls)
        self._cost_cumulative += cost
        rec = NodeRecord(
            **self._common(),
            node=scope.node,
            step=scope.step,
            attempt_count=scope.attempt_count,
            reflection_depth=scope.reflection_depth,
            input_hash=scope.input_hash,
            output_summary=summarize_update(update),
            llm_calls=scope.llm_calls,
            tokens_in=sum(
                c.input_tokens + c.cache_read_tokens + c.cache_write_tokens for c in scope.llm_calls
            ),
            tokens_out=sum(c.output_tokens for c in scope.llm_calls),
            cost_usd=cost,
            cost_cumulative=self._cost_cumulative,
            latency_ms=scope.elapsed_ms(),
            sandbox_execs=scope.sandbox_execs,
            status_after=status_after,
            error=redact(error) if error else None,
        )
        self._totals.add(rec)
        self._write(rec)
        return rec

    def abort_open_scope(self, error: str) -> None:
        """Close a node scope left open by an interrupt so the trace can still be ended."""
        if self._scope is not None:
            self.end_node(self._scope, {}, "error", error)

    # -- sub-events --------------------------------------------------------

    def record_llm(self, rec: LLMCallRecord) -> None:
        self._active("LLM call").llm_calls.append(rec)

    def record_exec(
        self, cmd: str, exit_code: int | None, duration_ms: float, timed_out: bool
    ) -> None:
        self._active("sandbox exec").sandbox_execs.append(
            SandboxExecRecord(
                cmd=redact(cmd[:_CMD_CHARS]),
                exit_code=exit_code,
                duration_ms=duration_ms,
                timed_out=timed_out,
            )
        )

    def write_artifact(self, kind: str, payload: dict[str, Any]) -> str:
        scope = self._active("artifact")
        d = self.artifact_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{scope.step:04d}_{scope.node}_{len(scope.llm_calls)}.json"
        path.write_text(redact(json.dumps(payload, indent=2, default=str)), encoding="utf-8")
        return str(path.relative_to(self.artifact_dir.parent))

    # -- internals ---------------------------------------------------------

    def _write_final_state(self, state: dict[str, Any]) -> str:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / FINAL_STATE_FILE
        path.write_text(redact(json.dumps(state, indent=2, default=str)), encoding="utf-8")
        return str(path.relative_to(self.artifact_dir.parent))

    def _active(self, what: str) -> NodeScope:
        if self._scope is None:
            raise RuntimeError(f"{what} recorded outside any node scope; it would be untraced")
        return self._scope

    def _common(self) -> dict[str, Any]:
        seq = self._seq
        self._seq += 1
        return {
            "seq": seq,
            "ts": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "variant": self.config.variant,
            "seed": self.config.seed,
            "config_hash": self._config_hash,
        }

    def _write(self, rec: BaseModel) -> None:
        if self._fh is None or self._closed:
            raise RuntimeError("trace is not open")
        self._fh.write(redact(rec.model_dump_json()) + "\n")
        self._fh.flush()


def summarize_update(update: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in update.items():
        if isinstance(value, list):
            out[key] = {"n_new": len(value), "hash": canonical_hash(value)}
        elif isinstance(value, str):
            out[key] = (
                value
                if len(value) <= _SUMMARY_STR_CHARS
                else (value[:_SUMMARY_STR_CHARS] + f"...[{len(value)} chars]")
            )
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- validation


def read_trace(path: Path) -> list[RunStart | NodeRecord | RunEnd]:
    return [
        _RECORD_ADAPTER.validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_trace(path: Path) -> list[str]:
    """Return a list of problems. An empty list means the run is valid for analysis."""
    if not path.exists():
        return [f"trace file missing: {path}"]
    try:
        records = read_trace(path)
    except ValidationError as exc:
        return [f"schema violation: {exc.errors()[0]}"]
    except ValueError as exc:
        return [f"unparseable trace: {exc}"]
    if not records:
        return ["empty trace"]

    problems = _check_structure(records)
    end = records[-1]
    if isinstance(end, RunEnd):
        problems += _check_totals(records, end)
        if end.final_state is not None and not (path.parent / end.final_state).is_file():
            problems.append(f"run_end.final_state points to a missing file: {end.final_state}")
    return problems


def _check_structure(records: list[RunStart | NodeRecord | RunEnd]) -> list[str]:
    problems: list[str] = []
    if not isinstance(records[0], RunStart):
        problems.append("first record is not run_start")
    if not isinstance(records[-1], RunEnd):
        problems.append("last record is not run_end (run incomplete)")
    n_start = sum(isinstance(r, RunStart) for r in records)
    n_end = sum(isinstance(r, RunEnd) for r in records)
    if n_start != 1 or n_end != 1:
        problems.append(f"expected exactly one run_start/run_end, got {n_start}/{n_end}")
    seqs = [r.seq for r in records]
    if seqs != list(range(len(records))):
        problems.append(f"seq not contiguous from 0: {seqs}")
    if len({r.run_id for r in records}) != 1:
        problems.append("mixed run_ids in one trace")
    if any(r.schema_version != SCHEMA_VERSION for r in records):
        problems.append("unexpected schema_version")
    return problems


def _check_totals(records: list[RunStart | NodeRecord | RunEnd], end: RunEnd) -> list[str]:
    problems: list[str] = []
    expected = RunTotals()
    for r in records:
        if isinstance(r, NodeRecord):
            expected.add(r)
    got = end.totals
    for field in ("nodes", "llm_calls", "tokens_in", "tokens_out", "sandbox_runs"):
        if getattr(got, field) != getattr(expected, field):
            problems.append(
                f"run_end.totals.{field}={getattr(got, field)} but nodes sum to "
                f"{getattr(expected, field)}"
            )
    if not math.isclose(got.cost_usd, expected.cost_usd, rel_tol=1e-9, abs_tol=1e-12):
        problems.append(
            f"run_end.totals.cost_usd={got.cost_usd} but nodes sum to {expected.cost_usd}"
        )
    if end.status == "running":
        problems.append("run ended with non-terminal status 'running'")
    return problems


# --------------------------------------------------------------------------- environment


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        sha = "uncommitted"
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (subprocess.SubprocessError, OSError):
        dirty = True
    return sha, dirty


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "missing"
    return out
