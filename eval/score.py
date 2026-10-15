"""Official SWE-bench scoring of an experiment's predictions.

    uv run python -m eval.score --experiment out/experiments/a1-frozen [--max-workers 4]
                                [--score-candidates]

Runs ``python -m swebench.harness.run_evaluation`` UNMODIFIED, once per seed. The dataset
is passed as a local JSONL written from the frozen records, so scoring uses exactly the
pinned revision rather than whatever is current on Hugging Face. Writes
``<experiment>/scores.json``.

``--score-candidates`` additionally scores every applicable, non-duplicate candidate of
every multi-candidate round (critic-quality analysis). Predictions files must hold one
prediction per instance, so candidates are bucketed into one file per (seed, slot).
See docs/EVAL_SETUP.md for disk, Docker and runtime requirements.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from eval.data import load_set
from eval.experiment import Manifest, load_results

Outcome = Literal["resolved", "unresolved", "empty_patch", "error", "infra_failure", "incomplete"]
# Outcomes that are a real verdict on the patch; the rest are excluded from analysis.
VERDICTS: frozenset[str] = frozenset({"resolved", "unresolved", "empty_patch"})
_PRECEDENCE: list[tuple[str, Outcome]] = [
    ("error_ids", "error"),
    ("infra_failure_ids", "infra_failure"),
    ("resolved_ids", "resolved"),
    ("unresolved_ids", "unresolved"),
    ("empty_patch_ids", "empty_patch"),
    ("incomplete_ids", "incomplete"),
]


def parse_report(report: Mapping[str, Any], instance_ids: list[str]) -> dict[str, Outcome]:
    """Map the official final report (schema_version 2) to one outcome per instance.
    Errors take precedence; an instance missing from every list is 'incomplete'."""
    out: dict[str, Outcome] = {}
    for key, outcome in _PRECEDENCE:
        for iid in report.get(key, []):
            out.setdefault(iid, outcome)
    return {iid: out.get(iid, "incomplete") for iid in instance_ids}


def is_resolved(outcome: str | None) -> bool | None:
    """True/False for verdicts, None for outcomes excluded from analysis."""
    if outcome is None or outcome not in VERDICTS:
        return None
    return outcome == "resolved"


def candidate_slots(
    experiment: Path,
) -> dict[tuple[int, int], dict[str, tuple[str, int, str]]]:
    """(seed, slot) -> {instance_id: (run_id, candidate_index, diff)} for every applicable,
    non-duplicate candidate of rounds with >= 2 such candidates."""
    slots: dict[tuple[int, int], dict[str, tuple[str, int, str]]] = {}
    for (iid, seed), r in load_results(experiment / "results.jsonl").items():
        if not r.done or r.run_id is None:
            continue
        fs = json.loads((experiment / "traces" / r.run_id / "final_state.json").read_text())
        rounds: dict[int, list[dict[str, Any]]] = {}
        for c in fs["candidates"]:
            if c["diff"] and c["apply_error"] is None and c["duplicate_of"] is None:
                rounds.setdefault(c["attempt"], []).append(c)
        slot = 0
        for cands in rounds.values():
            if len(cands) < 2:
                continue
            for c in cands:
                slots.setdefault((seed, slot), {})[iid] = (r.run_id, c["index"], c["diff"])
                slot += 1
    return slots


def _run_harness(
    preds: Path,
    run_id: str,
    dataset: Path,
    ids: list[str],
    *,
    workdir: Path,
    workers: int,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", str(dataset), "--split", "test",
        "--predictions_path", str(preds), "--run_id", run_id,
        "--max_workers", str(workers), "--timeout", str(timeout),
        "--report_dir", str(workdir), "--instance_ids", *ids,
    ]  # fmt: skip
    subprocess.run(cmd, cwd=workdir, check=True)
    model = json.loads(preds.read_text().splitlines()[0])["model_name_or_path"]
    report = workdir / f"{model.replace('/', '__')}.{run_id}.json"
    return dict(json.loads(report.read_text()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--score-candidates", action="store_true")
    args = ap.parse_args(argv)
    exp: Path = args.experiment.resolve()
    manifest = Manifest.model_validate_json((exp / "manifest.json").read_text())
    data = load_set("frozen" if manifest.set == "frozen" else "dev")
    if data.sha256 != manifest.data_sha256:
        print("error: data file changed since the experiment ran", file=sys.stderr)
        return 2

    workdir = exp / "scoring"
    workdir.mkdir(exist_ok=True)
    dataset = workdir / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(r) + "\n" for r in data.records))

    scores: dict[str, Any] = {"harness": manifest.swebench_version, "data": data.sha256}
    seeds: dict[str, dict[str, Outcome]] = {}
    for seed in manifest.seeds:
        preds = exp / "predictions" / f"seed{seed}.jsonl"
        ids = [json.loads(x)["instance_id"] for x in preds.read_text().splitlines() if x]
        if not ids:
            continue
        report = _run_harness(
            preds, f"{manifest.name}-seed{seed}", dataset, ids, workdir=workdir,
            workers=args.max_workers, timeout=args.timeout,
        )  # fmt: skip
        seeds[str(seed)] = parse_report(report, ids)
    scores["seeds"] = seeds

    if args.score_candidates:
        cand_scores: dict[str, dict[str, Outcome]] = {}
        for (seed, slot), entries in sorted(candidate_slots(exp).items()):
            name = f"{manifest.name}__seed{seed}__cand{slot}"
            preds = workdir / f"cand_seed{seed}_slot{slot}.jsonl"
            preds.write_text("".join(
                json.dumps({"instance_id": iid, "model_name_or_path": name, "model_patch": d})
                + "\n" for iid, (_, _, d) in sorted(entries.items())
            ))  # fmt: skip
            report = _run_harness(
                preds, f"{manifest.name}-s{seed}-c{slot}", dataset, sorted(entries),
                workdir=workdir, workers=args.max_workers, timeout=args.timeout,
            )  # fmt: skip
            for iid, outcome in parse_report(report, sorted(entries)).items():
                run_id, cidx, _ = entries[iid]
                cand_scores.setdefault(run_id, {})[str(cidx)] = outcome
        scores["candidates"] = cand_scores

    (exp / "scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True))
    n = sum(1 for s in seeds.values() for o in s.values() if o == "resolved")
    print(f"scored {len(seeds)} seed(s); {n} resolved instance-seeds -> {exp / 'scores.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
