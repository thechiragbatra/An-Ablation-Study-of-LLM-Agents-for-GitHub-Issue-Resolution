"""Run one config over a set of instances x seeds: resumable, parallel, budget-capped.

Layout of an experiment directory:
    manifest.json             provenance; resume refuses if config/data changed
    results.jsonl             one line per finished (instance, seed); last line wins
    traces/                   full traces + final states + LLM artifacts
    predictions/seed<k>.jsonl SWE-bench predictions (submitted patch per instance)

Scheduling is seed-major (all instances for seed 0, then seed 1, ...), so a budget stop
leaves complete seeds. The per-experiment dollar cap is one shared ledger, restored
from results.jsonl on resume; when the remaining budget cannot cover one more
worst-case run, scheduling stops and the remaining pairs are reported as not run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.budget import BudgetLedger
from agent.config import REPO_ROOT, ExperimentConfig
from agent.instance import Instance
from agent.llm import LLMProvider
from agent.run import run_instance
from agent.tracing import RunEnd, read_trace
from eval.prereg import PreregStatus

PROMPTS_PATH = REPO_ROOT / "src" / "agent" / "prompts.py"
ProviderFactory = Callable[[Instance, int], LLMProvider]
Prepare = Callable[[Instance], None]


class ExperimentError(Exception):
    pass


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str
    seed: int
    run_id: str | None
    variant: str
    status: str
    trace: str | None  # relative to the experiment directory
    trace_valid: bool
    problems: list[str]
    cost_usd: float
    tokens_in: int
    tokens_out: int
    llm_calls: int
    sandbox_runs: int
    wall_time_s: float | None
    attempts: int
    model_patch: str
    aborted_by_experiment_cap: bool
    env_error: str | None
    finished_at: str

    @property
    def done(self) -> bool:
        """Counts as finished for resume purposes (otherwise it is re-run)."""
        return self.trace_valid and not self.aborted_by_experiment_cap and self.env_error is None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    variant: str
    set: str
    data_file: str
    data_sha256: str
    config: dict[str, Any]
    config_hash: str  # see base_config_hash
    seeds: list[int]
    git_sha: str
    git_dirty: bool
    prompts_sha256: str
    prereg: dict[str, Any] | None
    swebench_version: str
    created: str


@dataclass
class ExperimentReport:
    out_dir: Path
    ran: int = 0
    skipped_done: int = 0
    not_run: list[tuple[str, int]] = field(default_factory=list)
    env_errors: int = 0
    invalid_traces: int = 0
    aborted_by_cap: int = 0
    spent_usd: float = 0.0


def base_config_hash(config: ExperimentConfig) -> str:
    """Identity of the treatment. Excludes the seed, the trace location and the
    per-experiment dollar cap (an operational limit that may be raised to finish an
    interrupted experiment; the per-run cap stays because it changes agent behaviour)."""
    data = config.model_dump(mode="json")
    data.pop("seed")
    data.pop("trace_dir")
    data["budget"].pop("per_experiment_usd")
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_results(path: Path) -> dict[tuple[str, int], RunResult]:
    latest: dict[tuple[str, int], RunResult] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = RunResult.model_validate_json(line)
                latest[(r.instance_id, r.seed)] = r
    return latest


def total_spent(path: Path) -> float:
    """Every dollar ever spent in this experiment, including superseded re-runs."""
    if not path.exists():
        return 0.0
    return sum(
        RunResult.model_validate_json(line).cost_usd
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _git_info() -> tuple[str, bool]:
    def git(*args: str) -> str:
        res = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return res.stdout.strip() if res.returncode == 0 else ""

    return git("rev-parse", "HEAD") or "uncommitted", bool(git("status", "--porcelain"))


def _swebench_version() -> str:
    try:
        return version("swebench")
    except PackageNotFoundError:
        return "missing"


def _write_manifest(
    out_dir: Path,
    *,
    config: ExperimentConfig,
    set_name: str,
    data_file: str,
    data_sha256: str,
    seeds: Sequence[int],
    prereg: PreregStatus | None,
) -> Manifest:
    path = out_dir / "manifest.json"
    chash = base_config_hash(config)
    if path.exists():
        old = Manifest.model_validate_json(path.read_text())
        if old.config_hash != chash:
            raise ExperimentError("config differs from the existing manifest; use a new --out")
        if old.data_sha256 != data_sha256 or old.set != set_name:
            raise ExperimentError("data set differs from the existing manifest; use a new --out")
        merged = old.model_copy(update={"seeds": sorted(set(old.seeds) | set(seeds))})
        path.write_text(merged.model_dump_json(indent=2))
        return merged
    git_sha, dirty = _git_info()
    manifest = Manifest(
        name=config.name,
        variant=config.variant,
        set=set_name,
        data_file=data_file,
        data_sha256=data_sha256,
        config=config.model_dump(mode="json", exclude={"seed", "trace_dir"}),
        config_hash=chash,
        seeds=sorted(seeds),
        git_sha=git_sha,
        git_dirty=dirty,
        prompts_sha256=hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest(),
        prereg=({"commit": prereg.commit, "sha256": prereg.sha256} if prereg is not None else None),
        swebench_version=_swebench_version(),
        created=datetime.now(UTC).isoformat(),
    )
    path.write_text(manifest.model_dump_json(indent=2))
    return manifest


def _submitted_patch(final_state: dict[str, Any]) -> str:
    idx = final_state.get("selected_index")
    if idx is None:
        return ""
    return final_state["candidates"][idx].get("diff") or ""


def write_predictions(out_dir: Path, config: ExperimentConfig, seeds: Sequence[int]) -> None:
    results = load_results(out_dir / "results.jsonl")
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for seed in seeds:
        lines = [
            json.dumps(
                {
                    "instance_id": r.instance_id,
                    "model_name_or_path": f"{config.name}__seed{seed}",
                    "model_patch": r.model_patch,
                }
            )
            for (_, s), r in sorted(results.items())
            if s == seed and r.done
        ]
        (pred_dir / f"seed{seed}.jsonl").write_text("".join(f"{x}\n" for x in lines))


def run_experiment(
    *,
    config: ExperimentConfig,
    set_name: str,
    data_file: str,
    data_sha256: str,
    instances: Sequence[Instance],
    seeds: Sequence[int],
    workers: int,
    out_dir: Path,
    prereg: PreregStatus | None = None,
    provider_factory: ProviderFactory | None = None,
    prepare: Prepare | None = None,
) -> ExperimentReport:
    if workers < 1:
        raise ExperimentError("workers must be >= 1")
    if config.publisher.enabled:
        raise ExperimentError("the publisher must be disabled for benchmark experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        out_dir, config=config, set_name=set_name, data_file=data_file,
        data_sha256=data_sha256, seeds=seeds, prereg=prereg,
    )  # fmt: skip
    results_path = out_dir / "results.jsonl"
    done = {k for k, r in load_results(results_path).items() if r.done}

    ledger = BudgetLedger(config.budget.per_experiment_usd, "experiment")
    spent_before = total_spent(results_path)
    if spent_before > 0:
        ledger.record(spent_before)

    report = ExperimentReport(out_dir=out_dir)
    pending: list[tuple[Instance, int]] = []
    for seed in sorted(seeds):  # seed-major: a budget stop leaves complete seeds
        for inst in instances:
            if (inst.instance_id, seed) in done:
                report.skipped_done += 1
            else:
                pending.append((inst, seed))

    write_lock = threading.Lock()

    def one(inst: Instance, seed: int) -> RunResult:
        result = _run_one(
            config, inst, seed, out_dir=out_dir, ledger=ledger,
            provider_factory=provider_factory, prepare=prepare,
        )  # fmt: skip
        with write_lock, results_path.open("a", encoding="utf-8") as fh:
            fh.write(result.model_dump_json() + "\n")
        return result

    _schedule(
        pending, one, workers=workers, ledger=ledger, reserve=config.budget.per_run_usd,
        report=report,
    )  # fmt: skip
    report.spent_usd = ledger.spent
    write_predictions(out_dir, config, seeds)
    return report


def _schedule(
    pending: Sequence[tuple[Instance, int]],
    one: Callable[[Instance, int], RunResult],
    *,
    workers: int,
    ledger: BudgetLedger,
    reserve: float,
    report: ExperimentReport,
) -> None:
    """Keep at most ``workers`` runs in flight; stop scheduling once the experiment
    ledger cannot cover ``reserve`` (the worst case one more run can spend)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        queue = iter(pending)
        inflight: set[Future[RunResult]] = set()
        stopped = False
        while True:
            while not stopped and len(inflight) < workers:
                nxt = next(queue, None)
                if nxt is None:
                    break
                if ledger.remaining < reserve:
                    stopped = True
                    report.not_run.append((nxt[0].instance_id, nxt[1]))
                    break
                inflight.add(pool.submit(one, *nxt))
            if not inflight:
                break
            finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                r = fut.result()
                report.ran += 1
                report.env_errors += r.env_error is not None
                report.invalid_traces += not r.trace_valid and r.env_error is None
                report.aborted_by_cap += r.aborted_by_experiment_cap
        if stopped:
            report.not_run.extend((i.instance_id, s) for i, s in queue)


def _run_one(
    config: ExperimentConfig,
    inst: Instance,
    seed: int,
    *,
    out_dir: Path,
    ledger: BudgetLedger,
    provider_factory: ProviderFactory | None,
    prepare: Prepare | None,
) -> RunResult:
    now = datetime.now(UTC).isoformat
    base: dict[str, Any] = {
        "instance_id": inst.instance_id,
        "seed": seed,
        "variant": config.variant,
        "finished_at": now(),
    }
    try:
        if prepare is not None:
            prepare(inst)
        cfg = config.model_copy(update={"seed": seed, "trace_dir": out_dir / "traces"})
        provider = provider_factory(inst, seed) if provider_factory else None
        outcome = run_instance(cfg, inst, provider=provider, experiment_ledger=ledger)
    except Exception as exc:  # environment failures (image pull, sandbox infra)
        return RunResult(
            **base, run_id=None, status="env_error", trace=None, trace_valid=False,
            problems=[], cost_usd=0.0, tokens_in=0, tokens_out=0, llm_calls=0,
            sandbox_runs=0, wall_time_s=None, attempts=0, model_patch="",
            aborted_by_experiment_cap=False, env_error=f"{type(exc).__name__}: {exc}",
        )  # fmt: skip

    end = next((r for r in read_trace(outcome.trace_path) if isinstance(r, RunEnd)), None)
    fs = outcome.final_state
    error = fs["error"] or ""
    return RunResult(
        **{**base, "finished_at": now()},
        run_id=outcome.run_id,
        status=fs["status"],
        trace=str(outcome.trace_path.relative_to(out_dir)),
        trace_valid=not outcome.trace_problems,
        problems=outcome.trace_problems,
        cost_usd=end.totals.cost_usd if end else fs["cost_so_far"],
        tokens_in=end.totals.tokens_in if end else 0,
        tokens_out=end.totals.tokens_out if end else 0,
        llm_calls=end.totals.llm_calls if end else 0,
        sandbox_runs=end.totals.sandbox_runs if end else 0,
        wall_time_s=end.wall_time_s if end else None,
        attempts=fs["attempt_count"],
        model_patch=_submitted_patch(dict(fs)),
        aborted_by_experiment_cap=(
            fs["status"] == "budget_exceeded" and "experiment budget exceeded" in error
        ),
        env_error=None,
    )
