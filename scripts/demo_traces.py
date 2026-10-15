"""Generate SCRIPTED demo traces for the dashboard: FakeLLM + real Docker sandbox.

    uv run python scripts/demo_traces.py [--out out/demo_traces]
    uv run python -m agent.dashboard --trace-dir out/demo_traces

These exercise every pipeline path (verified, unverified, retry, critic, reflection,
budget abort, error) so the dashboard can be inspected before any real API run. They use
model "fake-llm", are tagged "scripted" in the UI, and must never be mixed with
experimental results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.config import ExperimentConfig  # noqa: E402
from agent.instance import Instance  # noqa: E402
from agent.llm import Usage  # noqa: E402
from agent.run import run_instance  # noqa: E402
from agent.sandbox.docker_sandbox import build_image  # noqa: E402
from tests.fake_llm import FakeLLM, Scripted  # noqa: E402
from tests.scenarios import (  # noqa: E402
    BUGGY,
    C_CORRECT,
    C_CORRECT_ALT,
    C_UNAPPLIABLE,
    C_WRONG,
    arch,
    cand,
    loc,
    reflect,
    score,
    select,
)

FIXTURES = REPO_ROOT / "tests" / "fixtures"
C_WRONG_2 = cand(BUGGY, "    return sum(values) / len(values) if values else 1.0\n", why="1.0")
ON = {"enabled": True, "uses_sandbox": False}
ON_SANDBOX = {"enabled": True, "uses_sandbox": True}
OFF = {"enabled": False, "uses_sandbox": False}

SCENARIOS: list[tuple[str, dict[str, Any], dict[str, list[Any]]]] = [
    ("A0", {}, {"localize": [loc()], "architect": [arch(C_CORRECT)]}),
    ("A0", {"seed": 1}, {"localize": [loc()], "architect": [arch(C_WRONG)]}),
    (
        "A1",
        {"N": 3},
        {"localize": [loc()], "architect": [arch(C_UNAPPLIABLE), arch(C_WRONG), arch(C_CORRECT)]},
    ),
    (
        "A1",
        {"N": 3, "seed": 1, "budget": {"per_run_usd": 0.0105}},
        {"localize": [Scripted(loc(), usage=Usage(9000, 1000))], "architect": [arch(C_CORRECT)]},
    ),
    (
        "A2",
        {"K": 3, "critic": ON_SANDBOX},
        {
            "localize": [loc()],
            "architect": [arch(C_WRONG, C_CORRECT, C_UNAPPLIABLE)],
            "critic_select": [select(1, score(0, correctness=2), score(1, correctness=5, risk=1))],
        },
    ),
    (
        "A2",
        {"K": 3, "critic": OFF, "seed": 7},
        {"localize": [loc()], "architect": [arch(C_WRONG, C_CORRECT, C_CORRECT_ALT)]},
    ),
    (
        "A3",
        {"K": 1, "D": 2, "critic": ON},
        {
            "localize": [loc()],
            "architect": [arch(C_WRONG), arch(C_WRONG_2), arch(C_CORRECT)],
            "critic_reflect": [
                reflect("Empty input still returns None."),
                reflect("Returns 1.0; the docstring requires 0.0."),
            ],
        },
    ),
    (
        "A3",
        {"K": 2, "D": 1, "critic": ON, "seed": 1},
        {
            "localize": [loc()],
            "architect": [arch(C_WRONG, C_WRONG_2), arch(C_WRONG, C_WRONG_2)],
            "critic_select": [select(0, score(0), score(1)), select(3, score(2), score(3))],
            "critic_reflect": [reflect()],
        },
    ),
    (
        "A3",
        {"K": 2, "D": 2, "critic": ON, "seed": 2},
        {
            "localize": [loc()],
            "architect": [arch(C_WRONG, C_CORRECT)],
            "critic_select": [Scripted(None, stop_reason="refusal")],
        },
    ),
]


def _config(variant: str, out: Path, overrides: dict[str, Any]) -> ExperimentConfig:
    data = ExperimentConfig.from_yaml(
        REPO_ROOT / "configs" / f"{variant.lower()}.yaml"
    ).model_dump()
    data.update({"model": "fake-llm", "max_tokens": 1000, "trace_dir": out})
    for key, value in overrides.items():
        data[key] = {**data[key], **value} if isinstance(value, dict) else value
    return ExperimentConfig.model_validate(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "out" / "demo_traces")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    build_image(FIXTURES / "docker", "ablation-fixture:py3.12")
    instance = Instance.from_json(FIXTURES / "instances" / "calc_001.json")
    for variant, overrides, script in SCENARIOS:
        outcome = run_instance(
            _config(variant, args.out, overrides), instance, provider=FakeLLM(script)
        )
        valid = "valid" if not outcome.trace_problems else f"INVALID {outcome.trace_problems}"
        print(f"{variant:3} {outcome.final_state['status']:17} {outcome.run_id}  trace {valid}")
    print(f"\nView: uv run python -m agent.dashboard --trace-dir {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
