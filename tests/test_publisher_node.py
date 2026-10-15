"""Publisher node on the fixture repo (real Docker, scripted LLM): packages only
sandbox-verified runs, locally, and is a terminal leaf of the graph."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agent.budget import BudgetLedger
from agent.config import ExperimentConfig
from agent.deps import Deps
from agent.graph import build_graph
from agent.instance import Instance, IssueOrigin
from agent.llm import AccountedLLM, PricingTable
from agent.publishing import (
    DISCLOSURE_MARKER,
    REVIEWER_PLACEHOLDER,
    TITLE_PREFIX,
    load_package,
    package_invariants,
)
from agent.sandbox import E2BSandbox
from agent.tracing import Tracer
from tests.fake_llm import FakeLLM
from tests.scenarios import (
    CORRECT,
    WRONG_FIX,
    arch,
    cand,
    graph_nodes,
    loc,
    reflect,
    run,
    score,
    select,
)

pytestmark = pytest.mark.docker


@pytest.fixture
def pilot_instance(instance: Instance) -> Instance:
    return instance.model_copy(update={
        "instance_id": "me__calc-7",
        "origin": IssueOrigin(repo_slug="me/calc", issue_number=7,
                              issue_url="https://github.com/me/calc/issues/7",
                              default_branch="main", base_sha="c" * 40),
    })  # fmt: skip


def _config(cfg: Callable[..., ExperimentConfig], tmp_path: Path, variant: str = "A0",
            **kw: object) -> ExperimentConfig:  # fmt: skip
    return cfg(variant, publisher={"enabled": True, "pending_dir": tmp_path / "pending"}, **kw)


def test_verified_run_writes_a_complete_package(
    cfg: Callable[..., ExperimentConfig], pilot_instance: Instance, tmp_path: Path,
    fixture_image: str,
) -> None:  # fmt: skip
    out = run(_config(cfg, tmp_path), pilot_instance,
              FakeLLM({"localize": [loc()], "architect": [CORRECT]}))  # fmt: skip
    assert out.final_state["status"] == "sandbox_verified"
    assert graph_nodes(out)[-1] == "publisher"
    assert out.trace_problems == []

    (pkg_dir,) = list((tmp_path / "pending").iterdir())
    pkg = load_package(pkg_dir)
    assert package_invariants(pkg) == []
    assert pkg.title.startswith(TITLE_PREFIX)
    assert "mean() crashes on an empty list" in pkg.title
    assert pkg.body.startswith(DISCLOSURE_MARKER)
    assert "Resolves #7" in pkg.body
    assert REVIEWER_PLACEHOLDER in pkg.body
    assert "Only the repository's **existing** tests were run" in pkg.body
    assert pkg.head_branch.startswith("ai-patch/issue-7-")
    assert pkg.base_sha == "c" * 40
    assert "if not values:" in pkg.files["calc/stats.py"]  # full new file contents
    assert (pkg_dir / "SHA256").read_text().strip() == pkg.content_sha256()
    for name in ("title.txt", "body.md", "patch.diff"):
        assert (pkg_dir / name).read_text()


def test_unverified_run_writes_nothing(
    cfg: Callable[..., ExperimentConfig], pilot_instance: Instance, tmp_path: Path,
    fixture_image: str,
) -> None:  # fmt: skip
    out = run(_config(cfg, tmp_path), pilot_instance,
              FakeLLM({"localize": [loc()], "architect": [WRONG_FIX]}))  # fmt: skip
    assert out.final_state["status"] == "unverified"
    assert "publisher" not in graph_nodes(out)
    assert not (tmp_path / "pending").exists()


def test_a3_package_explains_selection_and_reflection(
    cfg: Callable[..., ExperimentConfig], pilot_instance: Instance, tmp_path: Path,
    fixture_image: str,
) -> None:  # fmt: skip
    wrong2 = cand("    return sum(values) / len(values)\n",
                  "    return sum(values) / len(values) if values else 1.0\n")  # fmt: skip
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(wrong2, cand("x\n", "y\n")), arch(CORRECT.candidates[0], wrong2)],
        "critic_select": [select(2, score(2, correctness=5))],
        "critic_reflect": [reflect("REFLECTION: must return 0.0")],
    })  # fmt: skip
    config = _config(cfg, tmp_path, "A3", K=2, D=1, critic={"enabled": True})
    out = run(config, pilot_instance, fake)
    assert out.final_state["status"] == "sandbox_verified"
    pkg = load_package(next((tmp_path / "pending").iterdir()))
    assert "Pipeline variant **A3**" in pkg.body
    assert "Selected candidate **#2**" in pkg.body
    assert "Critic rubric scores" in pkg.body
    assert "REFLECTION: must return 0.0" in pkg.body


def test_publisher_is_a_terminal_leaf(cfg: Callable[..., ExperimentConfig], tmp_path: Path) -> None:
    for variant, kw in (
        ("A0", {}),
        ("A1", {"N": 3}),
        ("A3", {"K": 2, "D": 2, "critic": {"enabled": True}}),
    ):
        config = _config(cfg, tmp_path, variant, **kw)
        tracer = Tracer(tmp_path, f"t-{variant}", "i", config)
        llm = AccountedLLM(FakeLLM({}), PricingTable.load(), BudgetLedger(1, "run"), tracer)
        g = build_graph(config, Deps(config, llm, E2BSandbox("t"), tracer)).get_graph()
        out_edges = {(e.source, e.target) for e in g.edges if e.source == "publisher"}
        assert out_edges == {("publisher", "__end__")}
        assert ("validator", "publisher") in {(e.source, e.target) for e in g.edges}
