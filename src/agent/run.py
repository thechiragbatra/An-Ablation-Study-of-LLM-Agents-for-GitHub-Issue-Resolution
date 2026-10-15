"""Run one config on one instance, end to end, with a complete trace.

    uv run python -m agent.run --config configs/a1.yaml \\
        --instance tests/fixtures/instances/calc_001.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from agent.budget import BudgetLedger
from agent.config import ExperimentConfig, Settings
from agent.deps import Deps
from agent.graph import build_graph, run_graph, traced_call
from agent.instance import Instance
from agent.llm import AccountedLLM, AnthropicProvider, LLMProvider, PricingTable
from agent.sandbox import DockerSandbox, E2BSandbox, Sandbox
from agent.state import TERMINAL_STATUSES, AgentState, StateUpdate, initial_state
from agent.tracing import Tracer, new_run_id, validate_trace


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    final_state: AgentState
    trace_path: Path
    trace_problems: list[str]


def make_sandbox(
    config: ExperimentConfig, tracer: Tracer, run_id: str, instance: Instance
) -> Sandbox:
    """The instance's own image (SWE-bench) takes precedence over the config default."""
    sc = config.sandbox
    if sc.backend == "docker":
        return DockerSandbox(
            image=instance.image or sc.image,
            workdir=instance.workdir,
            env_setup=instance.env_setup,
            setup_script=instance.setup_script,
            mem_limit=sc.mem_limit,
            cpus=sc.cpus,
            network=sc.network,
            observer=tracer.record_exec,
            run_label=run_id,
        )
    return E2BSandbox(template=sc.image, observer=tracer.record_exec)


def run_instance(
    config: ExperimentConfig,
    instance: Instance,
    provider: LLMProvider | None = None,
    experiment_ledger: BudgetLedger | None = None,
    pricing: PricingTable | None = None,
) -> RunOutcome:
    pricing = pricing or PricingTable.load()
    run_id = new_run_id()
    tracer = Tracer(config.trace_dir, run_id, instance.instance_id, config)
    tracer.start_run(pricing={config.model: pricing.price(config.model).model_dump()})

    if experiment_ledger is None:
        experiment_ledger = BudgetLedger(config.budget.per_experiment_usd, "experiment")
    ledger = BudgetLedger(config.budget.per_run_usd, "run", parent=experiment_ledger)
    if provider is None:
        key = Settings().anthropic_api_key
        provider = AnthropicProvider(api_key=key.get_secret_value() if key else None)
    llm = AccountedLLM(provider, pricing, ledger, tracer)
    sandbox = make_sandbox(config, tracer, run_id, instance)
    deps = Deps(config=config, llm=llm, sandbox=sandbox, tracer=tracer, instance=instance)

    state = initial_state(instance, str(tracer.path))
    try:
        setup = traced_call(
            tracer,
            "_sandbox_setup",
            state,
            lambda: StateUpdate(base_commit=sandbox.create(instance.repo, instance.base_commit)),
            lambda: ledger.spent,
        )
        state = AgentState(**{**state, **setup})
        if state["status"] not in TERMINAL_STATUSES:
            state = run_graph(build_graph(config, deps), state, config)
    except Exception as exc:  # e.g. GraphRecursionError; nodes themselves never raise
        state = _fail(tracer, state, exc)
    except BaseException as exc:  # KeyboardInterrupt etc.: close the trace, then propagate
        state = _fail(tracer, state, exc)
        raise
    finally:
        traced_call(
            tracer, "_sandbox_teardown", state, lambda: _destroy(sandbox), lambda: ledger.spent
        )
        if state["status"] == "running":  # graph returned without a verdict: a bug
            state = AgentState(**{**state, "status": "error", "error": "non-terminal exit"})
        tracer.end_run(
            status=state["status"],
            attempts=state["attempt_count"],
            reflection_depth=state["reflection_depth"],
            error=state["error"],
            final_state=dict(state),
        )
    return RunOutcome(run_id, state, tracer.path, validate_trace(tracer.path))


def _fail(tracer: Tracer, state: AgentState, exc: BaseException) -> AgentState:
    msg = f"{type(exc).__name__}: {exc}"
    tracer.abort_open_scope(msg)
    return AgentState(**{**state, "status": "error", "error": msg})


def _destroy(sandbox: Sandbox) -> StateUpdate:
    sandbox.destroy()
    return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--instance", type=Path, required=True)
    args = ap.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    instance = Instance.from_json(args.instance)
    outcome = run_instance(config, instance)
    s = outcome.final_state
    summary = {
        "run_id": outcome.run_id,
        "variant": config.variant,
        "instance_id": s["instance_id"],
        "status": s["status"],
        "attempts": s["attempt_count"],
        "cost_usd": round(s["cost_so_far"], 6),
        "trace": str(outcome.trace_path),
        "trace_valid": not outcome.trace_problems,
        "trace_problems": outcome.trace_problems,
        "error": s["error"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if not outcome.trace_problems else 2


if __name__ == "__main__":
    sys.exit(main())
