"""Experiment runner on the fixture repo: real Docker sandbox, scripted LLM."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent.config import ExperimentConfig
from agent.instance import Instance
from agent.llm import Usage
from eval.experiment import ExperimentError, load_results, run_experiment, total_spent
from tests.fake_llm import FakeLLM, Scripted
from tests.scenarios import CORRECT, WRONG_FIX, loc

pytestmark = pytest.mark.docker


@pytest.fixture
def instances(instance: Instance) -> list[Instance]:
    return [instance.model_copy(update={"instance_id": f"calc-{k}"}) for k in ("a", "b", "c")]


class Factory:
    """calc-c gets a wrong fix, the others a correct one; counts provider creations."""

    def __init__(self, usage: Usage | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.usage = usage

    def __call__(self, inst: Instance, seed: int) -> FakeLLM:
        self.calls.append((inst.instance_id, seed))
        fix = WRONG_FIX if inst.instance_id == "calc-c" else CORRECT
        if self.usage is None:
            return FakeLLM({"localize": [loc()], "architect": [fix]})
        return FakeLLM({
            "localize": [Scripted(loc(), usage=self.usage)],
            "architect": [Scripted(fix, usage=self.usage)],
        })  # fmt: skip


def _run(config: ExperimentConfig, instances: list[Instance], out: Path, factory: Factory,
         *, seeds: tuple[int, ...] = (0, 1), workers: int = 2, **kw: Any) -> Any:  # fmt: skip
    return run_experiment(
        config=config, set_name="fixture", data_file="calc_001.json", data_sha256="abc",
        instances=instances, seeds=seeds, workers=workers, out_dir=out,
        provider_factory=factory, **kw,
    )  # fmt: skip


def test_full_run_then_resume(cfg: Callable[..., ExperimentConfig], instances: list[Instance],
                              tmp_path: Path, fixture_image: str) -> None:  # fmt: skip
    out = tmp_path / "exp"
    factory = Factory()
    report = _run(cfg("A0"), instances, out, factory)
    assert report.ran == 6
    assert report.invalid_traces == 0
    assert report.not_run == []

    results = load_results(out / "results.jsonl")
    assert len(results) == 6
    assert all(r.done for r in results.values())
    assert results[("calc-a", 0)].status == "sandbox_verified"
    assert results[("calc-c", 1)].status == "unverified"
    assert "+    if not values:" in results[("calc-a", 1)].model_patch
    assert results[("calc-c", 0)].model_patch  # unverified patches are still submitted

    for seed in (0, 1):
        preds = [
            json.loads(x) for x in (out / f"predictions/seed{seed}.jsonl").read_text().splitlines()
        ]
        assert [p["instance_id"] for p in preds] == ["calc-a", "calc-b", "calc-c"]
        assert {p["model_name_or_path"] for p in preds} == {f"a0-fixture__seed{seed}"}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seeds"] == [0, 1]
    assert manifest["variant"] == "A0"
    assert len(manifest["prompts_sha256"]) == 64

    again = Factory()
    report2 = _run(cfg("A0"), instances, out, again)
    assert (report2.ran, report2.skipped_done) == (0, 6)
    assert again.calls == []


def test_interrupted_run_resumes_only_missing(cfg: Callable[..., ExperimentConfig],
                                             instances: list[Instance], tmp_path: Path,
                                             fixture_image: str) -> None:  # fmt: skip
    out = tmp_path / "exp"
    _run(cfg("A0"), instances, out, Factory())
    lines = (out / "results.jsonl").read_text().splitlines()
    (out / "results.jsonl").write_text("\n".join(lines[:2]) + "\n")  # "crash" after 2 runs
    kept = {(json.loads(x)["instance_id"], json.loads(x)["seed"]) for x in lines[:2]}
    factory = Factory()
    report = _run(cfg("A0"), instances, out, factory)
    assert report.ran == 4
    assert set(factory.calls).isdisjoint(kept)
    assert len(load_results(out / "results.jsonl")) == 6


def test_resume_refuses_changed_treatment(cfg: Callable[..., ExperimentConfig],
                                          instances: list[Instance], tmp_path: Path,
                                          fixture_image: str) -> None:  # fmt: skip
    out = tmp_path / "exp"
    _run(cfg("A0"), instances[:1], out, Factory(), seeds=(0,))
    with pytest.raises(ExperimentError, match="config differs"):
        _run(cfg("A0", max_localized_files=3), instances[:1], out, Factory(), seeds=(0,))
    # Raising only the per-experiment cap is allowed (operational, not treatment).
    raised = cfg("A0", budget={"per_experiment_usd": 99.0})
    assert _run(raised, instances[:1], out, Factory(), seeds=(0,)).skipped_done == 1


def test_experiment_cap_stops_scheduling_and_survives_restart(
    cfg: Callable[..., ExperimentConfig], instances: list[Instance], tmp_path: Path,
    fixture_image: str,
) -> None:  # fmt: skip
    # Each run costs $0.003 (2 x $0.0015). With a $0.004 per-run reserve and a $0.0075 cap,
    # exactly two sequential runs fit; the remaining four are reported as not run.
    config = cfg("A0", budget={"per_run_usd": 0.004, "per_experiment_usd": 0.0075})
    out = tmp_path / "exp"
    report = _run(config, instances, out, Factory(), workers=1)
    assert report.ran == 2
    assert len(report.not_run) == 4
    assert report.spent_usd == pytest.approx(0.006)
    # Restart: the ledger is restored from results.jsonl, so nothing more is scheduled.
    factory = Factory()
    report2 = _run(config, instances, out, factory, workers=1)
    assert (report2.ran, len(report2.not_run)) == (0, 4)
    assert factory.calls == []
    assert total_spent(out / "results.jsonl") == pytest.approx(0.006)


def test_parallel_runs_hitting_the_cap_are_tagged(
    cfg: Callable[..., ExperimentConfig], instances: list[Instance], tmp_path: Path,
    fixture_image: str,
) -> None:  # fmt: skip
    # Two runs start together (each reserve check sees $0.025 >= $0.02). Every call costs
    # $0.01, so exactly three calls fit under $0.025 and the fourth aborts its run.
    config = cfg("A0", budget={"per_run_usd": 0.02, "per_experiment_usd": 0.025})
    factory = Factory(usage=Usage(input_tokens=9000, output_tokens=1000))
    report = _run(config, instances[:2], tmp_path / "exp", factory, seeds=(0,), workers=2)
    assert report.ran == 2
    assert report.aborted_by_cap == 1
    results = load_results(tmp_path / "exp" / "results.jsonl")
    aborted = [r for r in results.values() if r.aborted_by_experiment_cap]
    assert len(aborted) == 1
    assert not aborted[0].done  # will be re-run on resume once the cap is raised
    assert aborted[0].trace_valid  # the abort itself is fully traced
    assert report.spent_usd <= 0.025 + 1e-9 + 0.01  # at most one call over the check


def test_environment_failure_is_recorded_not_counted(
    cfg: Callable[..., ExperimentConfig], instances: list[Instance], tmp_path: Path,
    fixture_image: str,
) -> None:  # fmt: skip
    def prepare(inst: Instance) -> None:
        if inst.instance_id == "calc-b":
            raise RuntimeError("image pull failed")

    out = tmp_path / "exp"
    report = _run(cfg("A0"), instances, out, Factory(), seeds=(0,), prepare=prepare)
    assert report.env_errors == 1
    r = load_results(out / "results.jsonl")[("calc-b", 0)]
    assert r.env_error == "RuntimeError: image pull failed"
    assert not r.done
    preds = (out / "predictions/seed0.jsonl").read_text()
    assert "calc-b" not in preds
