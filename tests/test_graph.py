"""A0/A1 graph routing tests. The LLM is scripted (FakeLLM); the sandbox is real Docker.

Every edge of the A0/A1 graphs is exercised: happy path, retry, cap exhaustion, budget
abort (run- and experiment-level), node error, refusal, and patch apply failure.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from langgraph.graph import END

from agent.budget import BudgetLedger
from agent.config import ExperimentConfig
from agent.graph import _after_validator, _continue_unless_aborted
from agent.instance import Instance
from agent.llm import Usage
from agent.run import RunOutcome
from agent.tracing import RunEnd, read_trace
from tests.fake_llm import FakeLLM, Scripted
from tests.scenarios import (
    BUGGY,
    CORRECT,
    UNAPPLIABLE,
    WRONG_FIX,
    graph_nodes,
    loc,
    router_state,
    run,
    trace_nodes,
)

pytestmark = pytest.mark.docker

_loc = loc
_nodes = trace_nodes
_graph_nodes = graph_nodes
_state = router_state


def _run(
    config: ExperimentConfig,
    instance: Instance,
    fake: FakeLLM,
    fixture_image: str,
    experiment_ledger: BudgetLedger | None = None,
) -> RunOutcome:
    assert config.sandbox.image == fixture_image
    return run(config, instance, fake, experiment_ledger)


# --------------------------------------------------------------------------- A0


def test_a0_happy_path(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({"localize": [_loc()], "architect": [CORRECT]})
    out = _run(cfg("A0"), instance, fake, fixture_image)

    assert out.trace_problems == []
    s = out.final_state
    assert s["status"] == "sandbox_verified"
    assert _graph_nodes(out) == ["localizer", "architect", "patcher", "selector", "validator"]
    assert _nodes(out)[0].node == "_sandbox_setup"
    assert [n.node for n in _nodes(out)][-1] == "_sandbox_teardown"
    assert s["attempt_count"] == 1
    assert s["selected_index"] == 0
    assert s["candidates"][0]["selected_by"] == "only"
    assert s["localized_files"] == ["calc/stats.py"]
    cand = s["candidates"][0]
    assert cand["apply_error"] is None
    assert cand["diff"] is not None
    assert "+    if not values:" in cand["diff"]
    assert [v["passed"] for v in s["validation_results"]] == [True]
    # 2 calls x (1000 in @ $1/M + 500 out @ $1/M)
    assert s["cost_so_far"] == pytest.approx(0.003)
    assert fake.remaining() == {}
    end = read_trace(out.trace_path)[-1]
    assert isinstance(end, RunEnd)
    assert end.final_state is not None
    persisted = json.loads((out.trace_path.parent / end.final_state).read_text())
    assert persisted == json.loads(json.dumps(s))  # the full final state, round-tripped
    assert "+    if not values:" in persisted["candidates"][0]["diff"]


def test_a0_trace_records_accounting(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    out = _run(
        cfg("A0"), instance, FakeLLM({"localize": [_loc()], "architect": [CORRECT]}), fixture_image
    )
    by_name = {n.node: n for n in _nodes(out)}
    assert len(by_name["localizer"].llm_calls) == 1
    assert by_name["architect"].tokens_out == 500
    assert by_name["validator"].llm_calls == []
    assert by_name["validator"].sandbox_execs[-1].exit_code == 0
    assert "pytest" in by_name["validator"].sandbox_execs[-1].cmd
    assert by_name["selector"].llm_calls == []
    assert by_name["validator"].cost_cumulative == pytest.approx(0.003)
    art = by_name["architect"].llm_calls[0].artifact
    assert art is not None
    assert (out.trace_path.parent / art).exists()
    end = read_trace(out.trace_path)[-1]
    assert isinstance(end, RunEnd)
    assert end.status == "sandbox_verified"
    assert end.totals.llm_calls == 2


def test_a0_wrong_fix_is_unverified_without_retry(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    out = _run(
        cfg("A0"),
        instance,
        FakeLLM({"localize": [_loc()], "architect": [WRONG_FIX]}),
        fixture_image,
    )
    s = out.final_state
    assert out.trace_problems == []
    assert s["status"] == "unverified"
    assert _graph_nodes(out) == ["localizer", "architect", "patcher", "selector", "validator"]
    vr = s["validation_results"][0]
    assert vr["applied"]
    assert not vr["passed"]
    assert vr["exit_code"] == 1
    assert "1 failed, 2 passed" in vr["stdout_tail"]


# --------------------------------------------------------------------------- A1


def test_a1_retries_with_feedback_until_pass(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({"localize": [_loc()], "architect": [UNAPPLIABLE, WRONG_FIX, CORRECT]})
    out = _run(cfg("A1", N=3), instance, fake, fixture_image)
    s = out.final_state

    assert out.trace_problems == []
    assert s["status"] == "sandbox_verified"
    assert s["attempt_count"] == 3
    assert (
        _graph_nodes(out) == ["localizer"] + ["architect", "patcher", "selector", "validator"] * 3
    )
    assert [c["index"] for c in s["candidates"]] == [0, 1, 2]
    assert s["candidates"][0]["apply_error"] is not None
    assert s["candidates"][0]["diff"] is None
    assert [v["applied"] for v in s["validation_results"]] == [False, True, True]
    assert [v["passed"] for v in s["validation_results"]] == [False, False, True]
    assert s["selected_index"] == 2

    arch_prompts = [r.messages[0].content for r in fake.requests if r.purpose == "architect"]
    assert "Earlier attempts failed" not in arch_prompts[0]
    assert "could not be applied" in arch_prompts[1]
    assert "Tests failed (exit code 1)" in arch_prompts[2]
    # Feedback is built on the base files, not on the previous (failed) patch.
    assert BUGGY in arch_prompts[2]


def test_a1_exhausts_n_and_stops(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({"localize": [_loc()], "architect": [WRONG_FIX, WRONG_FIX]})
    out = _run(cfg("A1", N=2), instance, fake, fixture_image)
    assert out.trace_problems == []
    assert out.final_state["status"] == "unverified"
    assert out.final_state["attempt_count"] == 2
    assert _graph_nodes(out).count("architect") == 2
    assert fake.remaining() == {}


# --------------------------------------------------------------------------- aborts


def test_run_budget_abort_is_clean(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    # Localizer costs $0.01. The architect's pre-call worst case includes at least
    # max_tokens(1000) x $1/M = $0.001, so a $0.0105 cap must stop it *before* spending.
    expensive = Usage(input_tokens=9000, output_tokens=1000)
    fake = FakeLLM({"localize": [Scripted(_loc(), usage=expensive)], "architect": [CORRECT]})
    out = _run(cfg("A1", N=3, budget={"per_run_usd": 0.0105}), instance, fake, fixture_image)
    s = out.final_state

    assert out.trace_problems == []
    assert s["status"] == "budget_exceeded"
    assert s["error"] is not None
    assert "run budget exceeded" in s["error"]
    assert _graph_nodes(out) == ["localizer", "architect"]
    assert s["cost_so_far"] == pytest.approx(0.01)
    assert s["cost_so_far"] <= 0.0105
    assert fake.remaining() == {"architect": 1}  # never called
    end = read_trace(out.trace_path)[-1]
    assert isinstance(end, RunEnd)
    assert end.status == "budget_exceeded"


def test_experiment_budget_abort(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    exp = BudgetLedger(0.05, "experiment")
    exp.record(0.0495)  # earlier runs consumed nearly everything
    fake = FakeLLM({"localize": [_loc()]})
    out = _run(cfg("A0"), instance, fake, fixture_image, experiment_ledger=exp)
    assert out.trace_problems == []
    assert out.final_state["status"] == "budget_exceeded"
    assert "experiment budget exceeded" in (out.final_state["error"] or "")
    assert _graph_nodes(out) == ["localizer"]
    assert fake.requests == []


def test_node_error_routes_to_end(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({"localize": [_loc("does/not/exist.py")]})
    out = _run(cfg("A1", N=3), instance, fake, fixture_image)
    assert out.trace_problems == []
    assert out.final_state["status"] == "error"
    assert "LocalizationError" in (out.final_state["error"] or "")
    assert _graph_nodes(out) == ["localizer"]


def test_refusal_is_recorded_as_error(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({"localize": [_loc()], "architect": [Scripted(None, stop_reason="refusal")]})
    out = _run(cfg("A1", N=3), instance, fake, fixture_image)
    assert out.trace_problems == []
    assert out.final_state["status"] == "error"
    assert "LLMRefusal" in (out.final_state["error"] or "")
    arch = next(n for n in _nodes(out) if n.node == "architect")
    assert len(arch.llm_calls) == 1  # refused call is still billed and traced
    assert arch.llm_calls[0].stop_reason == "refusal"


def test_setup_failure_still_yields_valid_trace(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    bad = instance.model_copy(update={"base_commit": "0" * 40})
    out = _run(cfg("A0"), bad, FakeLLM({}), fixture_image)
    assert out.trace_problems == []
    assert out.final_state["status"] == "error"
    assert [n.node for n in _nodes(out)] == ["_sandbox_setup", "_sandbox_teardown"]


# --------------------------------------------------------------------------- routers


@pytest.mark.parametrize(
    ("variant", "n", "status", "attempts", "expected"),
    [
        ("A1", 3, "running", 1, "architect"),
        ("A1", 3, "running", 3, END),  # hard cap enforced in the router too
        ("A1", 3, "sandbox_verified", 1, END),
        ("A1", 3, "unverified", 3, END),
        ("A1", 3, "budget_exceeded", 1, END),
        ("A1", 3, "error", 1, END),
        ("A0", 1, "running", 0, END),  # A0 never loops, whatever the state says
    ],
)
def test_after_validator_router(
    make_config: Callable[..., ExperimentConfig],
    *,
    variant: str,
    n: int,
    status: str,
    attempts: int,
    expected: str,
) -> None:
    route = _after_validator(make_config(variant, N=n))
    assert route(_state(status=status, attempt_count=attempts)) == expected


@pytest.mark.parametrize("status", ["budget_exceeded", "error"])
def test_every_forward_edge_aborts(status: str) -> None:
    assert _continue_unless_aborted("x")(_state(status=status)) == END
    assert _continue_unless_aborted("x")(_state(status="running")) == "x"
