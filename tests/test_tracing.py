from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from agent.config import ExperimentConfig
from agent.tracing import (
    LLMCallRecord,
    NodeRecord,
    RunEnd,
    RunStart,
    Tracer,
    canonical_hash,
    read_trace,
    redact,
    validate_trace,
)


def _llm_call(cost: float = 0.01) -> LLMCallRecord:
    return LLMCallRecord(
        model="m", input_tokens=100, output_tokens=50, cache_read_tokens=10,
        cache_write_tokens=0, cost_usd=cost, latency_ms=12.0, stop_reason="end_turn",
        request_id="r1", artifact=None,
    )  # fmt: skip


def _complete_trace(cfg: ExperimentConfig, run_id: str = "run-1") -> Tracer:
    t = Tracer(cfg.trace_dir, run_id, "inst-1", cfg)
    t.start_run(pricing={"m": {"input": 1}})
    for name in ("localizer", "architect"):
        scope = t.begin_node(name, {"attempt_count": 0, "x": [1, 2]})
        t.record_llm(_llm_call())
        t.record_exec("git ls-files", 0, 5.0, False)
        t.end_node(scope, {"localized_files": ["a.py"], "note": "y" * 500}, "running", None)
    t.end_run("unverified", attempts=1, reflection_depth=0, error=None)
    return t


@pytest.fixture
def cfg(make_config: Callable[..., ExperimentConfig]) -> ExperimentConfig:
    return make_config("A0")


def test_complete_trace_is_valid(cfg: ExperimentConfig) -> None:
    t = _complete_trace(cfg)
    assert validate_trace(t.path) == []
    recs = read_trace(t.path)
    assert isinstance(recs[0], RunStart)
    assert isinstance(recs[-1], RunEnd)
    nodes = [r for r in recs if isinstance(r, NodeRecord)]
    assert [n.node for n in nodes] == ["localizer", "architect"]
    assert [r.seq for r in recs] == list(range(len(recs)))
    for r in recs:
        assert (r.run_id, r.instance_id, r.variant, r.seed) == ("run-1", "inst-1", "A0", 0)
        assert r.config_hash == cfg.config_hash()
    assert nodes[1].cost_cumulative == pytest.approx(0.02)
    assert nodes[0].tokens_in == 110
    assert nodes[0].tokens_out == 50
    end = recs[-1]
    assert isinstance(end, RunEnd)
    assert end.totals.sandbox_runs == 2
    assert nodes[0].output_summary["localized_files"]["n_new"] == 1
    assert nodes[0].output_summary["note"].endswith("[500 chars]")


def test_run_start_captures_provenance(cfg: ExperimentConfig) -> None:
    start = read_trace(_complete_trace(cfg).path)[0]
    assert isinstance(start, RunStart)
    assert start.config["variant"] == "A0"
    assert {"anthropic", "langgraph", "docker", "pydantic"} <= set(start.packages)
    assert start.git_sha


def test_missing_run_end_is_invalid(cfg: ExperimentConfig) -> None:
    t = _complete_trace(cfg)
    lines = t.path.read_text().splitlines()
    t.path.write_text("\n".join(lines[:-1]) + "\n")
    assert any("run_end" in p for p in validate_trace(t.path))


def test_dropped_node_line_detected_by_totals(cfg: ExperimentConfig) -> None:
    t = _complete_trace(cfg)
    lines = t.path.read_text().splitlines()
    del lines[1]  # drop a node record
    t.path.write_text("\n".join(lines) + "\n")
    problems = validate_trace(t.path)
    assert any("seq not contiguous" in p for p in problems)
    assert any("totals.nodes" in p for p in problems)


def test_tampered_cost_detected(cfg: ExperimentConfig) -> None:
    t = _complete_trace(cfg)
    lines = t.path.read_text().splitlines()
    end = json.loads(lines[-1])
    end["totals"]["cost_usd"] = 0.0
    lines[-1] = json.dumps(end)
    t.path.write_text("\n".join(lines) + "\n")
    assert any("cost_usd" in p for p in validate_trace(t.path))


def test_missing_file_and_garbage(tmp_path: Path) -> None:
    assert validate_trace(tmp_path / "nope.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    assert validate_trace(bad)


def test_events_outside_scope_are_rejected(cfg: ExperimentConfig) -> None:
    t = Tracer(cfg.trace_dir, "run-2", "i", cfg)
    t.start_run(pricing={})
    with pytest.raises(RuntimeError, match="untraced"):
        t.record_exec("ls", 0, 1.0, False)
    with pytest.raises(RuntimeError, match="untraced"):
        t.record_llm(_llm_call())


def test_nested_scope_and_double_start_rejected(cfg: ExperimentConfig) -> None:
    t = Tracer(cfg.trace_dir, "run-3", "i", cfg)
    t.start_run(pricing={})
    with pytest.raises(RuntimeError, match="already started"):
        t.start_run(pricing={})
    t.begin_node("a", {})
    with pytest.raises(RuntimeError, match="inside"):
        t.begin_node("b", {})


def test_abort_open_scope_keeps_trace_valid(cfg: ExperimentConfig) -> None:
    t = Tracer(cfg.trace_dir, "run-4", "i", cfg)
    t.start_run(pricing={})
    t.begin_node("architect", {})
    t.abort_open_scope("KeyboardInterrupt: ")
    t.end_run("error", attempts=0, reflection_depth=0, error="KeyboardInterrupt")
    assert validate_trace(t.path) == []


def test_secrets_redacted_everywhere(cfg: ExperimentConfig) -> None:
    secret = "sk-ant-api03-ABCDEFGHIJKLMNOP"
    t = Tracer(cfg.trace_dir, "run-5", "i", cfg)
    t.start_run(pricing={})
    scope = t.begin_node("n", {})
    t.record_exec(f"curl -H 'x-api-key: {secret}'", 1, 1.0, False)
    t.write_artifact("llm", {"system": f"key {secret}", "api_key": "plainsecret"})
    t.end_node(scope, {"error": f"boom {secret}"}, "error", f"failed with {secret}")
    t.end_run("error", 0, 0, error=secret)
    on_disk = t.path.read_text() + "".join(p.read_text() for p in t.artifact_dir.rglob("*.json"))
    assert secret not in on_disk
    assert "plainsecret" not in on_disk
    assert "[REDACTED]" in on_disk


def test_redact_leaves_normal_text() -> None:
    text = '{"tokens_in": 5, "status": "running"}'
    assert redact(text) == text


def test_canonical_hash_order_independent() -> None:
    assert canonical_hash({"a": 1, "b": [1, 2]}) == canonical_hash({"b": [1, 2], "a": 1})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_final_state_persisted_and_checked(cfg: ExperimentConfig) -> None:
    t = Tracer(cfg.trace_dir, "run-6", "i", cfg)
    t.start_run(pricing={})
    state = {"candidates": [{"index": 0, "diff": "+x"}], "status": "unverified"}
    t.end_run("unverified", 1, 0, None, final_state=state)
    end = read_trace(t.path)[-1]
    assert isinstance(end, RunEnd)
    assert end.final_state == "run-6/final_state.json"
    assert json.loads((t.path.parent / end.final_state).read_text()) == state
    assert validate_trace(t.path) == []
    (t.path.parent / end.final_state).unlink()
    assert any("final_state" in p for p in validate_trace(t.path))
