"""Analysis on synthetic experiments with known answers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from eval.analyze import analyze, load_experiment, paired_instances, write_outputs
from eval.score import candidate_slots, is_resolved, parse_report
from tests.synthetic_experiments import cand, make_experiment

IDS = [f"i{k:02d}" for k in range(20)]


def _a0_a1(tmp_path: Path, **a1_kw: Any) -> list[Path]:
    # A0 resolves i00-i01; A1 resolves i00-i11 (10 discordant pairs, all in A1's favour).
    a0 = make_experiment(tmp_path, "A0", {i: [k < 2] * 3 for k, i in enumerate(IDS)}, cost=0.1)
    a1 = make_experiment(
        tmp_path, "A1", {i: [k < 12] * 3 for k, i in enumerate(IDS)}, cost=0.3, **a1_kw
    )
    return [a0, a1]


def test_hypothesis_h1_known_answer(tmp_path: Path) -> None:
    an = analyze([load_experiment(p) for p in _a0_a1(tmp_path)], n_boot=500)
    assert len(an.instances) == 20
    assert an.summaries["A0"].resolve_rate == pytest.approx(0.10)
    assert an.summaries["A1"].resolve_rate == pytest.approx(0.60)
    h1 = an.hypotheses["H1"]
    assert (h1["b_only"], h1["a_only"]) == (10, 0)
    assert h1["p_value"] == pytest.approx(2 * 0.5**10)
    assert h1["p_holm"] == pytest.approx(h1["p_value"])  # single hypothesis present
    assert h1["diff"] == pytest.approx(0.5)
    assert h1["diff_ci_lo"] > 0.2
    assert h1["cost_ratio"] == pytest.approx(3.0)
    assert an.summaries["A1"].cost_per_resolved == pytest.approx(0.3 / 0.6)
    assert an.summaries["A1"].seed_sd == 0.0
    assert len(an.per_seed_tests) == 3  # one per seed for H1


def test_majority_over_seeds(tmp_path: Path) -> None:
    a0 = make_experiment(tmp_path, "A0", {"x": [True, True, False], "y": [True, False, False]})
    an = analyze([load_experiment(a0)], n_boot=200)
    s = an.summaries["A0"]
    assert s.resolve_rate == pytest.approx(0.5)  # x by majority, y not
    assert s.per_seed_resolve == {0: 1.0, 1: 0.5, 2: 0.0}


def test_paired_design_excludes_errors_and_invalid_traces(tmp_path: Path) -> None:
    a0 = make_experiment(tmp_path, "A0", {"x": [True] * 3, "y": [False] * 3, "z": [True] * 3})
    a1 = make_experiment(
        tmp_path, "A1", {"x": [True] * 3, "y": [False, None, False], "z": [True] * 3},
        invalid={("z", 2)},
    )  # fmt: skip
    exps = [load_experiment(a0), load_experiment(a1)]
    # y has an evaluation error in A1, z an invalid trace: both leave ALL variants.
    assert paired_instances(exps) == ["x"]
    an = analyze(exps, n_boot=100)
    assert an.summaries["A1"].excluded_invalid == 1


def test_warnings(tmp_path: Path) -> None:
    exps = _a0_a1(tmp_path, prompts_sha="q" * 64, set_name="dev", scored=False)
    an = analyze([load_experiment(p) for p in exps], n_boot=100)
    text = " ".join(an.warnings)
    assert "DIFFERENT prompts" in text
    assert "mix data sets" in text
    assert "PROXY" in text


def test_critic_quality_from_gold_candidate_scores(tmp_path: Path) -> None:
    # Round 1 of each run has 3 applicable candidates; the critic selects candidate 0.
    # i00: selected resolves; i01: another resolves; i02: none resolves.
    cands = {f"A2-{i}-s0": [cand(0, selected=True), cand(1), cand(2)] for i in IDS[:3]}
    gold = {
        "A2-i00-s0": {"0": "resolved", "1": "unresolved", "2": "resolved"},
        "A2-i01-s0": {"0": "unresolved", "1": "resolved", "2": "unresolved"},
        "A2-i02-s0": {"0": "unresolved", "1": "unresolved", "2": "unresolved"},
    }
    a2 = make_experiment(
        tmp_path, "A2", {i: [True] for i in IDS[:3]}, critic=True, candidates=cands,
        candidate_outcomes=gold,
    )  # fmt: skip
    c = analyze([load_experiment(a2)], n_boot=100).critic["A2"]
    assert c["source"] == "gold"
    assert c["rounds"] == 3
    assert c["p_selected_resolves"] == pytest.approx(1 / 3)
    assert c["p_any_resolves"] == pytest.approx(2 / 3)
    assert c["p_selected_given_any"] == pytest.approx(1 / 2)
    assert c["oracle_gap"] == pytest.approx(1 / 3)
    # random baseline: (2/3 + 1/3 + 0) / 3 = 1/3 -> lift 0
    assert c["lift_over_random"] == pytest.approx(0.0)


def test_outputs_written(tmp_path: Path) -> None:
    an = analyze([load_experiment(p) for p in _a0_a1(tmp_path)], n_boot=200)
    out = tmp_path / "analysis"
    write_outputs(an, out)
    report = (out / "report.md").read_text()
    assert "| H1 | A1 > A0 | 10 / 0 |" in report
    assert (out / "frontier.png").stat().st_size > 1000
    with (out / "hypotheses.csv").open() as fh:
        assert next(csv.DictReader(fh))["comparison"] == "H1"


# --------------------------------------------------------------------------- scoring glue


def test_parse_report_precedence_and_missing() -> None:
    report = {
        "resolved_ids": ["a", "e"], "unresolved_ids": ["b"], "empty_patch_ids": ["c"],
        "error_ids": ["e"], "infra_failure_ids": [], "incomplete_ids": [],
    }  # fmt: skip
    out = parse_report(report, ["a", "b", "c", "d", "e"])
    assert out == {
        "a": "resolved", "b": "unresolved", "c": "empty_patch", "d": "incomplete", "e": "error",
    }  # fmt: skip
    assert [is_resolved(out[k]) for k in "abcde"] == [True, False, False, None, None]


def test_candidate_slots_one_prediction_per_instance_per_file(tmp_path: Path) -> None:
    cands = {
        "A2-i00-s0": [cand(0, selected=True), cand(1), cand(2)],
        "A2-i01-s0": [cand(0, selected=True), {**cand(1), "duplicate_of": 0}],  # 1 distinct
    }
    exp = make_experiment(tmp_path, "A2", {"i00": [True], "i01": [True]}, candidates=cands)
    slots = candidate_slots(exp)
    assert sorted(slots) == [(0, 0), (0, 1), (0, 2)]
    for entries in slots.values():
        assert list(entries) == ["i00"]  # i01 has < 2 distinct candidates
    assert slots[(0, 1)]["i00"] == ("A2-i00-s0", 1, "d1")
