"""The README's per-variant mermaid diagrams must match the compiled graphs exactly
(abort edges X -> END are summarised in prose and therefore excluded, except the
validator's verdict edge)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.budget import BudgetLedger
from agent.config import REPO_ROOT, ExperimentConfig
from agent.deps import Deps
from agent.graph import build_graph
from agent.llm import AccountedLLM, PricingTable
from agent.sandbox import E2BSandbox
from agent.tracing import Tracer
from tests.fake_llm import FakeLLM

README = REPO_ROOT / "README.md"
_BLOCK = re.compile(r"<!-- graph:(A\d) -->\s*```mermaid\n(.*?)```", re.S)
_EDGE = re.compile(r"(\w+)(?:\(\([^)]*\)\))?\s*-->(?:\|[^|]*\|)?\s*(?=(\w+))")
_ALIASES = {"START": "__start__", "END": "__end__"}


def readme_edges() -> dict[str, set[tuple[str, str]]]:
    out: dict[str, set[tuple[str, str]]] = {}
    for variant, body in _BLOCK.findall(README.read_text()):
        edges: set[tuple[str, str]] = set()
        for line in body.splitlines():
            for src, dst in _EDGE.findall(line):
                edges.add((_ALIASES.get(src, src), _ALIASES.get(dst, dst)))
        out[variant] = edges
    return out


def graph_edges(variant: str, tmp: Path) -> set[tuple[str, str]]:
    config = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "eval" / f"{variant.lower()}.yaml")
    tracer = Tracer(tmp, "readme", "i", config)
    llm = AccountedLLM(FakeLLM({}), PricingTable.load(), BudgetLedger(1, "run"), tracer)
    g = build_graph(config, Deps(config, llm, E2BSandbox("t"), tracer)).get_graph()
    return {
        (e.source, e.target) for e in g.edges if e.target != "__end__" or e.source == "validator"
    }


def test_readme_has_all_four_diagrams() -> None:
    assert sorted(readme_edges()) == ["A0", "A1", "A2", "A3"]


@pytest.mark.parametrize("variant", ["A0", "A1", "A2", "A3"])
def test_readme_diagram_matches_compiled_graph(variant: str, tmp_path: Path) -> None:
    assert readme_edges()[variant] == graph_edges(variant, tmp_path)
