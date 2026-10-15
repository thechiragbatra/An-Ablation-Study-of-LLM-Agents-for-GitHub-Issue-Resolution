from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eval.analyze import load_experiment
from eval.data import load_set
from eval.failure_taxonomy import CATEGORIES, COLUMNS, TaxonomyError, export, main, summarize
from eval.swebench_adapter import patch_files
from tests.synthetic_experiments import make_experiment

REAL = load_set("frozen").records[:12]  # real ids so auto-hints can use the gold patch


def _exps(tmp_path: Path) -> list[Path]:
    fails: dict[str, list[bool | None]] = {r["instance_id"]: [False] for r in REAL}
    gold_file = sorted(patch_files(REAL[0]["patch"])[0])[0]
    return [
        make_experiment(tmp_path, "A0", fails, localized=["nowhere/near.py"]),
        make_experiment(tmp_path, "A1", fails, localized=[gold_file]),
    ]


def test_export_samples_evenly_with_hints(tmp_path: Path) -> None:
    rows = export([load_experiment(p) for p in _exps(tmp_path)], n=7, seed=0)
    assert len(rows) == 7
    assert sorted(r["variant"] for r in rows).count("A0") == 4  # remainder to first variant
    assert all(r["category"] == "" for r in rows)
    assert {r["auto_hint"] for r in rows if r["variant"] == "A0"} == {"localization error"}
    again = export([load_experiment(p) for p in _exps(tmp_path / "again")], n=7, seed=0)
    assert [r["instance_id"] for r in again] == [r["instance_id"] for r in rows]


def _write(path: Path, labels: list[tuple[str, str]]) -> Path:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for i, (variant, category) in enumerate(labels):
            w.writerow({c: "" for c in COLUMNS} | {"run_id": f"r{i}", "variant": variant,
                                                    "category": category})  # fmt: skip
    return path


def test_summarize_counts(tmp_path: Path) -> None:
    path = _write(tmp_path / "l.csv", [("A0", "wrong fix"), ("A0", "wrong fix"),
                                       ("A1", "localization error")])  # fmt: skip
    table = summarize(path)
    assert table["A0"]["wrong fix"] == 2
    assert table["A1"]["localization error"] == 1


def test_summarize_rejects_bad_or_missing_labels(tmp_path: Path) -> None:
    with pytest.raises(TaxonomyError, match="unknown categories"):
        summarize(_write(tmp_path / "bad.csv", [("A0", "cosmic rays")]))
    unlabeled = _write(tmp_path / "u.csv", [("A0", "wrong fix"), ("A0", "")])
    with pytest.raises(TaxonomyError, match="1 row"):
        summarize(unlabeled)
    assert summarize(unlabeled, allow_unlabeled=True)["A0"]["wrong fix"] == 1


def test_cli_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exps = [str(p) for p in _exps(tmp_path)]
    out = tmp_path / "to_label.csv"
    assert main(["export", *exps, "--n", "4", "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    for r in rows:
        r["category"] = CATEGORIES[1]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert main(["summarize", str(out)]) == 0
    assert "| wrong fix | 2 | 2 | 4 |" in capsys.readouterr().out
