"""A2/A3 graph routing tests. The LLM is scripted (FakeLLM); the sandbox is real Docker.

Covers every Phase 2 edge: critic selection (valid, invalid -> fallback, forced), critic
sandbox checks, critic off (seeded random), duplicates, K shortfall/overflow, the A3
reflection loop to success and to max depth, D=0, critic-off A3, budget aborts inside
both critic modes, refusal, and node immutability.
"""

from __future__ import annotations

import copy
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langgraph.graph import END

from agent.budget import BudgetLedger
from agent.config import ExperimentConfig
from agent.deps import Deps
from agent.graph import _after_validator, traced_call
from agent.instance import Instance
from agent.llm import AccountedLLM, PricingTable
from agent.nodes import (
    NodeFn,
    architect,
    critic,
    critic_reflect,
    localizer,
    patcher,
    selector,
    validator,
)
from agent.policy import next_after_failure
from agent.sandbox import DockerSandbox, E2BSandbox
from agent.state import AgentState, Candidate, initial_state
from agent.tracing import Tracer
from tests.fake_llm import FakeLLM, Scripted
from tests.scenarios import (
    BUGGY,
    C_CORRECT,
    C_CORRECT_ALT,
    C_UNAPPLIABLE,
    C_WRONG,
    arch,
    cand,
    graph_nodes,
    loc,
    merge,
    reflect,
    router_state,
    run,
    score,
    select,
    trace_nodes,
)

pytestmark = pytest.mark.docker

CRITIC_ON = {"enabled": True, "uses_sandbox": False}
CRITIC_SANDBOX = {"enabled": True, "uses_sandbox": True}
CRITIC_OFF = {"enabled": False, "uses_sandbox": False}
C_WRONG_2 = cand(BUGGY, "    return sum(values) / len(values) if values else 1.0\n", why="1.0")
ROUND = ["architect", "patcher", "critic", "validator"]


def _prompts(fake: FakeLLM, purpose: str) -> list[str]:
    return [r.messages[0].content for r in fake.requests if r.purpose == purpose]


# --------------------------------------------------------------------------- A2, critic on


def test_a2_critic_selects_correct(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_UNAPPLIABLE, C_WRONG, C_CORRECT)],
        "critic_select": [select(2, score(1, correctness=2), score(2, correctness=5))],
    })  # fmt: skip
    out = run(cfg("A2", K=3, critic=CRITIC_ON), instance, fake)
    s = out.final_state

    assert out.trace_problems == []
    assert s["status"] == "sandbox_verified"
    assert graph_nodes(out) == ["localizer", *ROUND]
    assert s["candidates"][0]["apply_error"] is not None
    assert s["selected_index"] == 2
    assert s["candidates"][2]["selected_by"] == "critic"
    notes = s["critic_notes"]
    assert [(n["kind"], n["candidate_index"], n["selected"]) for n in notes] == [
        ("selection", 1, False),
        ("selection", 2, True),
    ]
    assert notes[1]["scores"]["correctness_likelihood"] == 5.0
    assert "[selection] pick 2" in notes[1]["critique"]
    prompt = _prompts(fake, "critic_select")[0]
    assert "<not_applicable>" in prompt
    assert "Selectable indices: 1, 2" in prompt
    assert [v["stage"] for v in s["validation_results"]] == ["final"]
    assert fake.remaining() == {}


def test_a2_critic_wrong_pick_does_not_loop(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_WRONG, C_CORRECT)],
        "critic_select": [select(0, score(0), score(1))],
    })  # fmt: skip
    out = run(cfg("A2", K=2, critic=CRITIC_ON), instance, fake)
    assert out.trace_problems == []
    assert out.final_state["status"] == "unverified"
    assert graph_nodes(out) == ["localizer", *ROUND]


@pytest.mark.parametrize("bad_pick", [0, 99])  # 0 is unappliable, 99 is out of range
def test_a2_invalid_pick_uses_rubric_fallback(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str, bad_pick: int
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_UNAPPLIABLE, C_WRONG, C_CORRECT)],
        "critic_select": [
            select(bad_pick, score(1, correctness=2), score(2, correctness=5, risk=1))
        ],
    })  # fmt: skip
    out = run(cfg("A2", K=3, critic=CRITIC_ON), instance, fake)
    s = out.final_state
    assert out.trace_problems == []
    assert s["selected_index"] == 2
    assert s["candidates"][2]["selected_by"] == "critic_fallback"
    selected_note = next(n for n in s["critic_notes"] if n["selected"])
    assert f"invalid index {bad_pick}" in selected_note["critique"]
    assert s["status"] == "sandbox_verified"


def test_a2_critic_sandbox_runs_each_candidate(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_WRONG, C_CORRECT, C_UNAPPLIABLE)],
        "critic_select": [select(1, score(0), score(1, correctness=5))],
    })  # fmt: skip
    out = run(cfg("A2", K=3, critic=CRITIC_SANDBOX), instance, fake)
    s = out.final_state

    assert out.trace_problems == []
    assert s["status"] == "sandbox_verified"
    critic_runs = [v for v in s["validation_results"] if v["stage"] == "critic"]
    assert [(v["candidate_index"], v["passed"]) for v in critic_runs] == [(0, False), (1, True)]
    assert [v["stage"] for v in s["validation_results"]][-1] == "final"
    prompt = _prompts(fake, "critic_select")[0]
    assert "PASSED" in prompt
    assert "FAILED (exit code 1)" in prompt
    critic_rec = next(n for n in trace_nodes(out) if n.node == "critic")
    assert sum("pytest" in e.cmd for e in critic_rec.sandbox_execs) == 2


def test_duplicates_marked_and_not_rerun(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_CORRECT, C_CORRECT, C_WRONG)],
        "critic_select": [select(0, score(0, correctness=5), score(2))],
    })  # fmt: skip
    out = run(cfg("A2", K=3, critic=CRITIC_SANDBOX), instance, fake)
    s = out.final_state
    assert [c["duplicate_of"] for c in s["candidates"]] == [None, 0, None]
    assert [v["candidate_index"] for v in s["validation_results"] if v["stage"] == "critic"] == [
        0,
        2,
    ]
    assert "<duplicate_of>0</duplicate_of>" in _prompts(fake, "critic_select")[0]
    assert "Selectable indices: 0, 2" in _prompts(fake, "critic_select")[0]


def test_all_unappliable_skips_critic_llm(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    other = cand("also not in the file\n", "y\n")
    fake = FakeLLM({"localize": [loc()], "architect": [arch(C_UNAPPLIABLE, other)]})
    out = run(cfg("A2", K=2, critic=CRITIC_ON), instance, fake)
    s = out.final_state
    assert out.trace_problems == []
    assert s["status"] == "unverified"
    assert _prompts(fake, "critic_select") == []
    assert s["candidates"][0]["selected_by"] == "only"
    assert s["validation_results"][-1]["applied"] is False


def test_single_applicable_candidate_is_forced(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({"localize": [loc()], "architect": [arch(C_UNAPPLIABLE, C_CORRECT)]})
    out = run(cfg("A2", K=2, critic=CRITIC_ON), instance, fake)
    assert out.final_state["status"] == "sandbox_verified"
    assert out.final_state["candidates"][1]["selected_by"] == "only"
    assert fake.requests[-1].purpose == "architect"  # no critic call


@pytest.mark.parametrize(("returned", "kept"), [(1, 1), (5, 3)])
def test_k_shortfall_and_overflow(
    cfg: Callable[..., ExperimentConfig],
    instance: Instance,
    fixture_image: str,
    returned: int,
    kept: int,
) -> None:
    cands = [C_CORRECT, C_WRONG, C_WRONG_2, C_CORRECT_ALT, C_UNAPPLIABLE][:returned]
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(*cands)],
        "critic_select": [select(0, score(0, correctness=5))],
    })  # fmt: skip
    out = run(cfg("A2", K=3, critic=CRITIC_ON), instance, fake)
    assert out.trace_problems == []
    assert len(out.final_state["candidates"]) == kept
    arch_rec = next(n for n in trace_nodes(out) if n.node == "architect")
    assert arch_rec.output_summary["candidates"]["n_new"] == kept


# --------------------------------------------------------------------------- A2, critic off


def test_a2_critic_off_seeded_random(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({"localize": [loc()], "architect": [arch(C_UNAPPLIABLE, C_WRONG, C_CORRECT)]})
    out = run(cfg("A2", K=3, critic=CRITIC_OFF), instance, fake)
    s = out.final_state
    assert out.trace_problems == []
    assert graph_nodes(out) == ["localizer", "architect", "patcher", "selector", "validator"]
    assert s["selected_index"] in (1, 2)
    assert s["candidates"][s["selected_index"]]["selected_by"] == "random"
    assert s["critic_notes"] == []


def _selector_choice(config: ExperimentConfig, cands: list[Candidate], tmp_path: Path) -> int:
    tracer = Tracer(tmp_path, "sel", "i", config)
    llm = AccountedLLM(FakeLLM({}), PricingTable.load(), BudgetLedger(1, "run"), tracer)
    deps = Deps(config=config, llm=llm, sandbox=E2BSandbox("t"), tracer=tracer)
    state = router_state(instance_id="inst", candidates=cands, attempt_count=1)
    idx = selector(state, deps)["selected_index"]
    assert idx is not None
    return idx


def _fake_cand(i: int, applicable: bool = True) -> Candidate:
    return Candidate(
        index=i, edits=[], rationale="", diff=f"diff {i}" if applicable else None, attempt=1,
        reflection_depth=0, apply_error=None if applicable else "nope", duplicate_of=None,
        selected_by=None,
    )  # fmt: skip


def test_selector_is_deterministic_per_seed_and_skips_unappliable(
    make_config: Callable[..., ExperimentConfig], tmp_path: Path
) -> None:
    cands = [_fake_cand(0, applicable=False)] + [_fake_cand(i) for i in range(1, 5)]
    picks = {}
    for seed in range(20):
        config = make_config("A2", K=5, seed=seed, critic=CRITIC_OFF)
        first = _selector_choice(config, cands, tmp_path)
        assert first == _selector_choice(config, cands, tmp_path)  # same seed, same pick
        assert first != 0  # never the unappliable candidate
        picks[seed] = first
    assert len(set(picks.values())) > 1  # the seed actually matters


# --------------------------------------------------------------------------- A3


def test_a3_reflects_until_pass(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_WRONG), arch(C_WRONG_2), arch(C_CORRECT)],
        "critic_reflect": [reflect("REFLECTION-ONE"), reflect("REFLECTION-TWO")],
    })  # fmt: skip
    out = run(cfg("A3", K=1, D=2, critic=CRITIC_ON), instance, fake)
    s = out.final_state

    assert out.trace_problems == []
    assert s["status"] == "sandbox_verified"
    assert graph_nodes(out) == [
        "localizer", *ROUND, "critic_reflect", *ROUND, "critic_reflect", *ROUND,
    ]  # fmt: skip
    assert s["attempt_count"] == 3
    assert s["reflection_depth"] == 2
    assert [c["reflection_depth"] for c in s["candidates"]] == [0, 1, 2]
    reflections = [n for n in s["critic_notes"] if n["kind"] == "reflection"]
    assert [(n["attempt"], n["candidate_index"]) for n in reflections] == [(1, 0), (2, 1)]

    arch_prompts = _prompts(fake, "architect")
    assert "REFLECTION-ONE" in arch_prompts[1]
    assert "Tests failed (exit code 1)" in arch_prompts[1]
    assert "REFLECTION-ONE" in arch_prompts[2]
    assert "REFLECTION-TWO" in arch_prompts[2]
    reflect_prompts = _prompts(fake, "critic_reflect")
    assert "<earlier_reviews>" not in reflect_prompts[0]
    assert "REFLECTION-ONE" in reflect_prompts[1]
    assert fake.remaining() == {}


def test_a3_max_depth_exit(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_WRONG, C_WRONG_2), arch(C_WRONG, C_WRONG_2)],
        "critic_select": [select(0, score(0), score(1)), select(3, score(2), score(3))],
        "critic_reflect": [reflect()],
    })  # fmt: skip
    out = run(cfg("A3", K=2, D=1, critic=CRITIC_ON), instance, fake)
    s = out.final_state
    assert out.trace_problems == []
    assert s["status"] == "unverified"
    assert s["attempt_count"] == 2
    assert graph_nodes(out) == ["localizer", *ROUND, "critic_reflect", *ROUND]
    assert sum(n["kind"] == "reflection" for n in s["critic_notes"]) == 1
    assert [v["candidate_index"] for v in s["validation_results"]] == [0, 3]
    assert fake.remaining() == {}


def test_a3_depth_zero_is_a2(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_WRONG, C_WRONG_2)],
        "critic_select": [select(0, score(0), score(1))],
    })  # fmt: skip
    out = run(cfg("A3", K=2, D=0, critic=CRITIC_ON), instance, fake)
    assert out.final_state["status"] == "unverified"
    assert graph_nodes(out) == ["localizer", *ROUND]


def test_a3_critic_off_retries_without_reflection(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({"localize": [loc()], "architect": [arch(C_WRONG), arch(C_CORRECT)]})
    out = run(cfg("A3", K=1, D=1, critic=CRITIC_OFF), instance, fake)
    s = out.final_state
    assert s["status"] == "sandbox_verified"
    assert graph_nodes(out) == ["localizer"] + ["architect", "patcher", "selector", "validator"] * 2
    assert s["critic_notes"] == []
    assert s["reflection_depth"] == 1


# --------------------------------------------------------------------------- aborts


def test_budget_abort_inside_critic(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    # Two $0.0015 calls spend $0.003; the critic's worst case (>= $0.001 of max_tokens plus
    # a prompt holding files and diffs) cannot fit under $0.0035, so it aborts pre-call.
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_WRONG, C_CORRECT)],
        "critic_select": [select(1)],
    })  # fmt: skip
    config = cfg("A2", K=2, critic=CRITIC_ON, budget={"per_run_usd": 0.0035})
    out = run(config, instance, fake)
    s = out.final_state
    assert out.trace_problems == []
    assert s["status"] == "budget_exceeded"
    assert graph_nodes(out) == ["localizer", "architect", "patcher", "critic"]
    assert fake.remaining() == {"critic_select": 1}
    assert s["cost_so_far"] <= 0.0035


def test_budget_abort_inside_reflection(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_WRONG), arch(C_CORRECT)],
        "critic_reflect": [reflect()],
    })  # fmt: skip
    config = cfg("A3", K=1, D=2, critic=CRITIC_ON, budget={"per_run_usd": 0.0035})
    out = run(config, instance, fake)
    s = out.final_state
    assert out.trace_problems == []
    assert s["status"] == "budget_exceeded"
    assert graph_nodes(out) == ["localizer", *ROUND, "critic_reflect"]
    assert fake.remaining() == {"architect": 1, "critic_reflect": 1}


def test_critic_refusal_is_error(
    cfg: Callable[..., ExperimentConfig], instance: Instance, fixture_image: str
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_WRONG, C_CORRECT)],
        "critic_select": [Scripted(None, stop_reason="refusal")],
    })  # fmt: skip
    out = run(cfg("A2", K=2, critic=CRITIC_ON), instance, fake)
    assert out.trace_problems == []
    assert out.final_state["status"] == "error"
    assert "LLMRefusal" in (out.final_state["error"] or "")


# --------------------------------------------------------------------------- routers / policy


@pytest.mark.parametrize(
    ("overrides", "attempts", "action", "route"),
    [
        ({"variant": "A0"}, 1, "stop", END),
        ({"variant": "A1", "N": 3}, 1, "retry", "architect"),
        ({"variant": "A1", "N": 3}, 3, "stop", END),
        ({"variant": "A2", "K": 3, "critic": CRITIC_ON}, 1, "stop", END),
        ({"variant": "A2", "K": 3, "critic": CRITIC_OFF}, 1, "stop", END),
        ({"variant": "A3", "K": 3, "D": 2, "critic": CRITIC_ON}, 1, "reflect", "critic_reflect"),
        ({"variant": "A3", "K": 3, "D": 2, "critic": CRITIC_ON}, 2, "reflect", "critic_reflect"),
        ({"variant": "A3", "K": 3, "D": 2, "critic": CRITIC_ON}, 3, "stop", END),
        ({"variant": "A3", "K": 3, "D": 2, "critic": CRITIC_OFF}, 1, "retry", "architect"),
        ({"variant": "A3", "K": 3, "D": 2, "critic": CRITIC_OFF}, 3, "stop", END),
        ({"variant": "A3", "K": 3, "D": 0, "critic": CRITIC_ON}, 1, "stop", END),
    ],
)
def test_policy_and_router_agree(
    make_config: Callable[..., ExperimentConfig],
    overrides: dict[str, Any],
    attempts: int,
    action: str,
    route: str,
) -> None:
    variant = overrides.pop("variant")
    config = make_config(variant, **overrides)
    assert next_after_failure(config, attempts) == action
    status = "unverified" if action == "stop" else "running"
    assert _after_validator(config)(router_state(status=status, attempt_count=attempts)) == route
    for terminal in ("sandbox_verified", "budget_exceeded", "error"):
        assert _after_validator(config)(router_state(status=terminal, attempt_count=1)) == END


# --------------------------------------------------------------------------- immutability


def _check_immutable(
    config: ExperimentConfig,
    instance: Instance,
    sandbox: DockerSandbox,
    *,
    tmp_path: Path,
    fake: FakeLLM,
    sequence: list[tuple[str, NodeFn]],
) -> AgentState:
    tracer = Tracer(tmp_path, f"imm-{config.variant}", instance.instance_id, config)
    tracer.start_run(pricing={})
    llm = AccountedLLM(fake, PricingTable.load(), BudgetLedger(1, "run"), tracer)
    deps = Deps(config=config, llm=llm, sandbox=sandbox, tracer=tracer)
    state = initial_state(instance, "t.jsonl")
    state = merge(state, {"base_commit": sandbox.base_commit or ""})
    for name, fn in sequence:
        before = copy.deepcopy(state)
        update = traced_call(tracer, name, state, functools.partial(fn, state, deps), float)
        assert state == before, f"{name} mutated its input state"
        assert "error" not in update, update
        state = merge(state, dict(update))
    return state


def test_every_node_leaves_input_untouched_critic_on(
    cfg: Callable[..., ExperimentConfig], instance: Instance, sandbox: DockerSandbox, tmp_path: Path
) -> None:
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_WRONG, C_WRONG_2), arch(C_WRONG, C_CORRECT)],
        "critic_select": [select(0, score(0)), select(3, score(3))],
        "critic_reflect": [reflect()],
    })  # fmt: skip
    rnd: list[tuple[str, NodeFn]] = [
        ("architect", architect), ("patcher", patcher), ("critic", critic),
        ("validator", validator),
    ]  # fmt: skip
    final = _check_immutable(
        cfg("A3", K=2, D=1, critic=CRITIC_SANDBOX), instance, sandbox, tmp_path=tmp_path,
        fake=fake,
        sequence=[("localizer", localizer), *rnd, ("critic_reflect", critic_reflect), *rnd],
    )  # fmt: skip
    assert final["status"] == "sandbox_verified"


def test_every_node_leaves_input_untouched_critic_off(
    cfg: Callable[..., ExperimentConfig], instance: Instance, sandbox: DockerSandbox, tmp_path: Path
) -> None:
    fake = FakeLLM({"localize": [loc()], "architect": [arch(C_CORRECT, C_CORRECT_ALT)]})
    final = _check_immutable(
        cfg("A2", K=2, critic=CRITIC_OFF), instance, sandbox, tmp_path=tmp_path, fake=fake,
        sequence=[("localizer", localizer), ("architect", architect), ("patcher", patcher),
         ("selector", selector), ("validator", validator)],
    )  # fmt: skip
    assert final["status"] == "sandbox_verified"
