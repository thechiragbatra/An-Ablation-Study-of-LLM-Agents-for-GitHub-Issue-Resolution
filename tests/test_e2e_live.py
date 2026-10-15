"""End-to-end A0-A3 on the fixture with the real Anthropic API and real Docker.

Opt-in (costs money): ``uv run pytest -m live --run-live``. Success of the patch is not
asserted (the model is nondeterministic); a complete, valid trace and a terminal status
are.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agent.config import ExperimentConfig
from agent.instance import Instance
from agent.run import run_instance
from agent.state import TERMINAL_STATUSES

pytestmark = [pytest.mark.live, pytest.mark.docker]


@pytest.mark.parametrize("variant", ["A0", "A1", "A2", "A3"])
def test_live_end_to_end(
    variant: str,
    make_config: Callable[..., ExperimentConfig],
    instance: Instance,
    fixture_image: str,
) -> None:
    config = make_config(variant)
    assert config.sandbox.image == fixture_image
    outcome = run_instance(config, instance)
    s = outcome.final_state
    print(
        f"\n[{variant}] status={s['status']} attempts={s['attempt_count']} "
        f"cost=${s['cost_so_far']:.4f} trace={outcome.trace_path}"
    )
    assert outcome.trace_problems == []
    assert s["status"] in TERMINAL_STATUSES
    assert s["status"] != "error", s["error"]
    assert s["cost_so_far"] <= config.budget.per_run_usd
