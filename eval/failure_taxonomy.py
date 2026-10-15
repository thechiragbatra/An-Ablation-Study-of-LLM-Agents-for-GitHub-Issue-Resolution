"""Failure taxonomy: export a CSV of sampled failures for hand labelling, then summarize.

    uv run python -m eval.failure_taxonomy export out/experiments/a*-frozen \\
        --n 30 --seed 0 --out analysis/failures_to_label.csv
    uv run python -m eval.failure_taxonomy summarize analysis/failures_labeled.csv

``auto_hint`` is a heuristic suggestion computed on the analysis side (it may use the
gold patch, which the agent never sees). The ``category`` column is filled in BY HAND.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval.analyze import Experiment, load_experiment
from eval.data import load_set
from eval.swebench_adapter import patch_files

CATEGORIES = (
    "localization error",
    "wrong fix",
    "test misunderstanding",
    "env/sandbox error",
    "budget exhausted",
    "other",
)
COLUMNS = (
    "run_id", "experiment", "variant", "seed", "instance_id", "status", "resolved",
    "trace", "auto_hint", "category", "notes",
)  # fmt: skip


class TaxonomyError(Exception):
    pass


def _failures(e: Experiment) -> list[tuple[str, int]]:
    out = []
    for (iid, seed), r in sorted(e.valid().items()):
        verdict = e.resolved.get((iid, seed)) if e.scored else r.status == "sandbox_verified"
        if verdict is False:
            out.append((iid, seed))
    return out


def auto_hint(e: Experiment, iid: str, seed: int, gold_files: dict[str, set[str]]) -> str:
    r = e.results[(iid, seed)]
    if r.status == "budget_exceeded":
        return "budget exhausted"
    if r.run_id is None:
        return ""
    fs_path = e.path / "traces" / r.run_id / "final_state.json"
    if not fs_path.exists():
        return ""
    fs = json.loads(fs_path.read_text())
    if r.status == "error":
        err = fs.get("error") or ""
        return "env/sandbox error" if ("Sandbox" in err or "docker" in err.lower()) else "other"
    if iid in gold_files and not set(fs["localized_files"]) & gold_files[iid]:
        return "localization error"
    return ""


def export(exps: list[Experiment], n: int, seed: int) -> list[dict[str, Any]]:
    """Sample n failures, as evenly as possible across variants, seeded."""
    rng = random.Random(seed)
    pools = {e.variant: (e, _failures(e)) for e in exps}
    variants = sorted(pools)
    quota = {v: n // len(variants) for v in variants}
    for v in variants[: n % len(variants)]:
        quota[v] += 1
    gold: dict[str, set[str]] = {}
    for set_name in {e.manifest.set for e in exps}:
        for rec in load_set("frozen" if set_name == "frozen" else "dev").records:
            gold[rec["instance_id"]] = patch_files(rec["patch"])[0]
    rows: list[dict[str, Any]] = []
    for v in variants:
        e, pool = pools[v]
        for iid, s in sorted(rng.sample(pool, min(quota[v], len(pool)))):
            r = e.results[(iid, s)]
            rows.append({
                "run_id": r.run_id, "experiment": e.manifest.name, "variant": v, "seed": s,
                "instance_id": iid, "status": r.status,
                "resolved": e.resolved.get((iid, s)) if e.scored else "",
                "trace": r.trace or "", "auto_hint": auto_hint(e, iid, s, gold),
                "category": "", "notes": "",
            })  # fmt: skip
    return rows


def summarize(path: Path, allow_unlabeled: bool = False) -> dict[str, Counter[str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    unlabeled = [r["run_id"] for r in rows if not r["category"].strip()]
    invalid = sorted({r["category"] for r in rows if r["category"].strip()} - set(CATEGORIES))
    if invalid:
        raise TaxonomyError(f"unknown categories {invalid}; allowed: {list(CATEGORIES)}")
    if unlabeled and not allow_unlabeled:
        raise TaxonomyError(f"{len(unlabeled)} row(s) still unlabeled")
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        if r["category"].strip():
            table[r["variant"]][r["category"].strip()] += 1
    return dict(table)


def render(table: dict[str, Counter[str]]) -> str:
    variants = sorted(table)
    lines = [
        "| Category | " + " | ".join(variants) + " | Total |",
        "|---" * (len(variants) + 2) + "|",
    ]
    for cat in CATEGORIES:
        counts = [table[v][cat] for v in variants]
        lines.append(f"| {cat} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
    totals = [sum(table[v].values()) for v in variants]
    lines.append("| **total** | " + " | ".join(map(str, totals)) + f" | {sum(totals)} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("experiments", type=Path, nargs="+")
    ex.add_argument("--n", type=int, default=30)
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--out", type=Path, required=True)
    sm = sub.add_parser("summarize")
    sm.add_argument("csv", type=Path)
    sm.add_argument("--allow-unlabeled", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "export":
        rows = export([load_experiment(p) for p in args.experiments], args.n, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} failures to {args.out}; categories: {', '.join(CATEGORIES)}")
        return 0
    try:
        print(render(summarize(args.csv, args.allow_unlabeled)))
    except TaxonomyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
