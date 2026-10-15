"""Real SWE-bench integration (opt-in: ``pytest -m swebench --run-swebench``).

Pulls one official environment image (~GBs) and checks the pieces the experiment relies
on, with no LLM involved:
1. the agent sandbox starts from the image with the repo at the base commit;
2. the agent-visible regression command runs at base;
3. the GOLD patch applies with ``git apply`` and keeps the regression command green
   (sanity check that the regression signal does not punish correct fixes);
4. the official harness, run through eval.score, marks the gold patch as resolved.
Needs the disk and Docker resources described in docs/EVAL_SETUP.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.sandbox import DockerSandbox
from eval.data import load_set
from eval.score import _run_harness, parse_report
from eval.swebench_adapter import ensure_image, to_instance

pytestmark = [pytest.mark.swebench, pytest.mark.docker]


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    # A small, fast repo from the DEV set (never the frozen subset).
    return next(r for r in load_set("dev").records if r["repo"] == "psf/requests")


def test_sandbox_regression_and_gold_patch(record: dict[str, Any], docker_client: Any) -> None:
    inst = to_instance(record)
    assert inst.image is not None
    ensure_image(inst.image, docker_client)
    with DockerSandbox(
        inst.image, client=docker_client, workdir=inst.workdir, env_setup=inst.env_setup,
        setup_script=inst.setup_script, mem_limit="4g",
    ) as sb:  # fmt: skip
        assert sb.create(inst.repo, inst.base_commit) == inst.base_commit
        assert sb.read_diff() == ""
        base = sb.run(inst.test_command, 900)
        assert not base.timed_out
        assert sb.apply_diff(record["patch"]).ok
        after = sb.run(inst.test_command, 900)
        assert after.ok, after.stdout[-2000:]


def test_official_harness_resolves_gold(record: dict[str, Any], tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(record) + "\n")
    preds = tmp_path / "gold.jsonl"
    iid = record["instance_id"]
    preds.write_text(json.dumps(
        {"instance_id": iid, "model_name_or_path": "gold", "model_patch": record["patch"]}
    ) + "\n")  # fmt: skip
    report = _run_harness(
        preds, "integration", dataset, [iid], workdir=tmp_path, workers=1, timeout=1800
    )
    assert parse_report(report, [iid]) == {iid: "resolved"}
