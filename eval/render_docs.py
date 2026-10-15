"""Fill ``{{PLACEHOLDER}}`` tokens in the report and résumé templates from REAL analysis
outputs, so result numbers are never typed by hand.

    uv run python -m eval.render_docs --analysis analysis/ \\
        [--experiments out/experiments/a{0,1,2,3}-frozen] \\
        [--taxonomy analysis/failures_labeled.csv] [--pilot-outcomes out/pilot/pr_outcomes.csv] \\
        [--out docs/rendered/]

Templates are never modified; filled copies go to ``--out``. Any placeholder without a
value (or listed in MANUAL) stays visible as ``{{KEY}}`` and is reported, so gaps are
never silently papered over.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from agent.config import REPO_ROOT
from eval.experiment import Manifest
from eval.failure_taxonomy import CATEGORIES, render, summarize

TEMPLATES = (
    REPO_ROOT / "docs" / "REPORT_SKELETON.md",
    REPO_ROOT / "docs" / "RESUME_BULLETS.md",
)
TOKEN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
VARIANTS = ("A0", "A1", "A2", "A3")
HYPOTHESES = ("H1", "H2", "H3")
# Filled by the author, never by this tool.
MANUAL = frozenset({
    "AUTHOR", "SUPERVISOR", "INSTITUTION", "DATE", "ALPHA", "H4_THRESHOLD",
    "PLATFORM_NOTE", "N_TESTS", "PLACEHOLDER", "PLACEHOLDERS",
})  # fmt: skip


def known_keys() -> frozenset[str]:
    keys = {
        "N_INSTANCES", "N_LABELLED", "TAXONOMY_TABLE", "TOP_FAILURE_CATEGORY", "MODEL",
        "EFFORT", "K", "D", "N_RETRIES", "CRITIC_SANDBOX", "CAP_RUN", "CAP_EXPERIMENT",
        "PREREG_COMMIT", "PROMPTS_SHA", "N_PILOT_PRS", "N_PILOT_MERGED",
    }  # fmt: skip
    for v in VARIANTS:
        keys |= {f"{k}_{v}" for k in ("RESOLVE", "CI_LO", "CI_HI", "SEED_SD", "SV_RATE",
                                      "COST", "COST_PER_RESOLVED", "BUDGET_ABORT")}  # fmt: skip
    for h in HYPOTHESES:
        keys |= {f"{h}_{k}" for k in ("B_ONLY", "A_ONLY", "DIFF", "CI_LO", "CI_HI", "P",
                                      "P_HOLM", "COST_RATIO")}  # fmt: skip
    for v in ("A2", "A3"):
        keys |= {f"CRITIC_{v}_{k}" for k in ("ROUNDS", "P_SEL_GIVEN_ANY", "ORACLE_GAP",
                                             "LIFT", "SOURCE")}  # fmt: skip
    return frozenset(keys)


# --------------------------------------------------------------------------- formatting


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pp(x: float) -> str:
    return f"{100 * x:+.1f} pp"


def usd(x: float) -> str:
    return f"${x:.2f}" if x >= 0.995 else f"${x:.3f}"


def pval(x: float) -> str:
    return "< 0.0001" if x < 1e-4 else f"{x:.4f}"


def _num(s: str | None) -> float | None:
    if s is None or s in {"", "None"}:
        return None
    return float(s)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- collectors


def from_analysis(analysis: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in _read_csv(analysis / "variants.csv"):
        v = row["variant"]
        out["N_INSTANCES"] = row["n_instances"]
        fmt = {
            "RESOLVE": (row.get("resolve_rate"), pct),
            "SEED_SD": (row.get("seed_sd"), lambda x: f"{100 * x:.1f} pp"),
            "SV_RATE": (row.get("sandbox_verified_rate"), pct),
            "COST": (row.get("mean_cost"), usd),
            "COST_PER_RESOLVED": (row.get("cost_per_resolved"), usd),
            "BUDGET_ABORT": (row.get("budget_abort_rate"), pct),
        }
        for key, (raw, f) in fmt.items():
            val = _num(raw)
            if val is not None:
                out[f"{key}_{v}"] = f(val)
        ci = row.get("resolve_ci")
        if ci and ci != "None":
            lo, hi = ast.literal_eval(ci)
            out[f"CI_LO_{v}"], out[f"CI_HI_{v}"] = pct(lo), pct(hi)
    for row in _read_csv(analysis / "hypotheses.csv"):
        h = row["comparison"]
        out[f"{h}_B_ONLY"], out[f"{h}_A_ONLY"] = row["b_only"], row["a_only"]
        out[f"{h}_DIFF"] = pp(float(row["diff"]))
        out[f"{h}_CI_LO"] = pp(float(row["diff_ci_lo"]))
        out[f"{h}_CI_HI"] = pp(float(row["diff_ci_hi"]))
        out[f"{h}_P"] = pval(float(row["p_value"]))
        out[f"{h}_P_HOLM"] = pval(float(row["p_holm"]))
        ratio = _num(row.get("cost_ratio"))
        if ratio is not None:
            out[f"{h}_COST_RATIO"] = f"{ratio:.2f}x"
    critic_path = analysis / "critic_quality.json"
    if critic_path.exists():
        for v, c in json.loads(critic_path.read_text()).items():
            out[f"CRITIC_{v}_SOURCE"] = c.get("source", "")
            out[f"CRITIC_{v}_ROUNDS"] = str(c.get("rounds", 0))
            metrics = (
                ("P_SEL_GIVEN_ANY", "p_selected_given_any", pct),
                ("ORACLE_GAP", "oracle_gap", pp),
                ("LIFT", "lift_over_random", pp),
            )
            for key, field, f in metrics:
                if c.get(field) is not None:
                    out[f"CRITIC_{v}_{key}"] = f(float(c[field]))
    return out


def from_experiments(dirs: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    manifests = {
        m.variant: m
        for m in (Manifest.model_validate_json((d / "manifest.json").read_text()) for d in dirs)
    }
    if not manifests:
        return out
    any_m = next(iter(manifests.values()))
    out["MODEL"] = f"`{any_m.config['model']}`"
    out["EFFORT"] = f"`{any_m.config['effort']}`"
    out["CAP_RUN"] = usd(float(any_m.config["budget"]["per_run_usd"]))
    out["CAP_EXPERIMENT"] = usd(float(any_m.config["budget"]["per_experiment_usd"]))
    out["PROMPTS_SHA"] = f"`{any_m.prompts_sha256[:12]}`"
    if any_m.prereg and any_m.prereg.get("commit"):
        out["PREREG_COMMIT"] = f"`{str(any_m.prereg['commit'])[:12]}`"
    if "A1" in manifests:
        out["N_RETRIES"] = str(manifests["A1"].config["N"])
    if "A3" in manifests:
        a3 = manifests["A3"].config
        out["K"], out["D"] = str(a3["K"]), str(a3["D"])
        out["CRITIC_SANDBOX"] = "on" if a3["critic"]["uses_sandbox"] else "off"
    return out


def from_taxonomy(labeled_csv: Path) -> dict[str, str]:
    table = summarize(labeled_csv, allow_unlabeled=False)
    totals: Counter[str] = Counter()
    for counts in table.values():
        totals.update(counts)
    out = {"TAXONOMY_TABLE": render(table), "N_LABELLED": str(sum(totals.values()))}
    if totals:
        top, n = max(((c, totals[c]) for c in CATEGORIES), key=lambda kv: kv[1])
        if n:
            out["TOP_FAILURE_CATEGORY"] = top
    return out


def from_pilot(outcomes_csv: Path) -> dict[str, str]:
    rows = _read_csv(outcomes_csv)
    return {
        "N_PILOT_PRS": str(len(rows)),
        "N_PILOT_MERGED": str(sum(r["outcome"] == "merged" for r in rows)),
    }


# --------------------------------------------------------------------------- rendering


def fill(text: str, values: dict[str, str]) -> tuple[str, list[str]]:
    missing: list[str] = []

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key in values and key not in MANUAL:
            return values[key]
        if key not in missing:
            missing.append(key)
        return m.group(0)

    return TOKEN.sub(sub, text), missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--experiments", type=Path, nargs="*", default=[])
    ap.add_argument("--taxonomy", type=Path)
    ap.add_argument("--pilot-outcomes", type=Path)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "rendered")
    args = ap.parse_args(argv)

    values: dict[str, Any] = from_analysis(args.analysis)
    values |= from_experiments(args.experiments)
    if args.taxonomy:
        values |= from_taxonomy(args.taxonomy)
    if args.pilot_outcomes:
        values |= from_pilot(args.pilot_outcomes)

    args.out.mkdir(parents=True, exist_ok=True)
    for template in TEMPLATES:
        text, missing = fill(template.read_text(), values)
        (args.out / template.name).write_text(text)
        auto_missing = [k for k in missing if k not in MANUAL]
        manual = [k for k in missing if k in MANUAL]
        print(f"{template.name}: wrote {args.out / template.name}")
        if auto_missing:
            print(f"  still missing (no data yet): {', '.join(auto_missing)}", file=sys.stderr)
        if manual:
            print(f"  fill by hand: {', '.join(manual)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
