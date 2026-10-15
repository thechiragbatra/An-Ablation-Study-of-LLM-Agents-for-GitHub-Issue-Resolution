"""Report/résumé placeholders are all accounted for, fill only from real analysis files,
and the Makefile exposes the required targets (checked with ``make -n``; nothing runs)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent.config import REPO_ROOT
from eval.analyze import analyze, load_experiment, write_outputs
from eval.render_docs import MANUAL, TEMPLATES, TOKEN, fill, from_analysis, known_keys
from eval.render_docs import main as render_main
from tests.synthetic_experiments import cand, make_experiment

IDS = [f"i{k:02d}" for k in range(20)]


def test_every_template_placeholder_is_known_or_manual() -> None:
    allowed = known_keys() | MANUAL
    for template in TEMPLATES:
        used = set(TOKEN.findall(template.read_text()))
        assert used, template
        assert used <= allowed, (template.name, sorted(used - allowed))


def test_templates_contain_no_hand_typed_results() -> None:
    # Result sections must use placeholders, not literal percentages.
    report = TEMPLATES[0].read_text()
    results = report.split("## 5. Results", 1)[1].split("## 7. Threats", 1)[0]
    assert "%" not in results.replace("95%", "")


def _analysis(tmp: Path) -> Path:
    a0 = make_experiment(tmp, "A0", {i: [k < 2] * 3 for k, i in enumerate(IDS)}, cost=0.1)
    a1 = make_experiment(tmp, "A1", {i: [k < 12] * 3 for k, i in enumerate(IDS)}, cost=0.3)
    cands = {f"A2-{i}-s0": [cand(0, selected=True), cand(1)] for i in IDS[:2]}
    gold = {f"A2-{i}-s0": {"0": "resolved", "1": "unresolved"} for i in IDS[:2]}
    a2 = make_experiment(tmp, "A2", {i: [k < 12] * 3 for k, i in enumerate(IDS)}, cost=0.5,
                         critic=True, candidates=cands, candidate_outcomes=gold)  # fmt: skip
    out = tmp / "analysis"
    write_outputs(analyze([load_experiment(p) for p in (a0, a1, a2)], n_boot=300), out)
    return out


def test_values_come_from_analysis_files(tmp_path: Path) -> None:
    values = from_analysis(_analysis(tmp_path))
    assert values["N_INSTANCES"] == "20"
    assert values["RESOLVE_A0"] == "10.0%"
    assert values["RESOLVE_A1"] == "60.0%"
    assert values["COST_A1"] == "$0.300"
    assert values["H1_DIFF"] == "+50.0 pp"
    assert values["H1_B_ONLY"] == "10"
    assert values["H1_P"] == "0.0020"
    assert values["H1_COST_RATIO"] == "3.00x"
    assert values["CRITIC_A2_P_SEL_GIVEN_ANY"] == "100.0%"
    assert "RESOLVE_A3" not in values  # no data -> no value


def test_fill_keeps_unknown_and_manual_tokens_visible() -> None:
    text, missing = fill("{{AUTHOR}} got {{RESOLVE_A0}} and {{RESOLVE_A3}}",
                         {"RESOLVE_A0": "10.0%", "AUTHOR": "should not be used"})  # fmt: skip
    assert text == "{{AUTHOR}} got 10.0% and {{RESOLVE_A3}}"
    assert missing == ["AUTHOR", "RESOLVE_A3"]


def test_render_writes_copies_and_leaves_templates_alone(tmp_path: Path) -> None:
    before = [t.read_text() for t in TEMPLATES]
    out = tmp_path / "rendered"
    assert render_main(["--analysis", str(_analysis(tmp_path)), "--out", str(out)]) == 0
    assert [t.read_text() for t in TEMPLATES] == before
    rendered = (out / "REPORT_SKELETON.md").read_text()
    assert "| A1 | 60.0% |" in rendered
    assert "{{RESOLVE_A3}}" in rendered  # gap stays visible
    assert "{{AUTHOR}}" in rendered


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("lint", ["ruff check .", "ruff format --check ."]),
        ("typecheck", ["mypy ."]),
        ("test", ["pytest"]),
        ("eval-dev", ["--set dev", "eval.score", "eval.analyze"]),
        ("eval-full", ["--set frozen", "--seeds 0 1 2", "--score-candidates"]),
        ("analyze", ["eval.analyze", "failure_taxonomy export"]),
    ],
)
def test_make_targets(target: str, expected: list[str]) -> None:
    res = subprocess.run(["make", "-n", target], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=False)  # fmt: skip
    assert res.returncode == 0, res.stderr
    for fragment in expected:
        assert fragment in res.stdout, (target, fragment)
