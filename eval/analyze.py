"""Analysis across variants: resolve rates with bootstrap CIs, paired McNemar tests
(H1-H3, Holm-corrected), cost/efficiency, seed variance and Critic quality.

    uv run python -m eval.analyze out/experiments/a{0,1,2,3}-frozen --out analysis/

Rules applied mechanically (docs/PREREGISTRATION.md):
- only runs with valid traces count; exclusions are reported;
- an instance is analysed only if it has a scoring verdict for EVERY seed of EVERY
  compared variant (paired design); evaluation/infra errors exclude it everywhere;
- primary unit: per-instance majority across seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.experiment import Manifest, RunResult, load_results
from eval.score import is_resolved
from eval.stats import (
    McNemarResult,
    bootstrap_ci,
    holm,
    majority,
    mcnemar,
    mean,
    paired_diff_ci,
    stdev,
)

HYPOTHESES = {"H1": ("A0", "A1"), "H2": ("A1", "A2"), "H3": ("A2", "A3")}


@dataclass
class Experiment:
    path: Path
    manifest: Manifest
    results: dict[tuple[str, int], RunResult]
    resolved: dict[tuple[str, int], bool | None]  # None = no verdict (excluded)
    candidate_outcomes: dict[str, dict[str, str]]  # run_id -> candidate index -> outcome
    scored: bool

    @property
    def variant(self) -> str:
        return self.manifest.variant

    def valid(self) -> dict[tuple[str, int], RunResult]:
        return {k: r for k, r in self.results.items() if r.done}


def load_experiment(path: Path) -> Experiment:
    manifest = Manifest.model_validate_json((path / "manifest.json").read_text())
    results = load_results(path / "results.jsonl")
    scores_path = path / "scores.json"
    resolved: dict[tuple[str, int], bool | None] = {}
    candidates: dict[str, dict[str, str]] = {}
    scored = scores_path.exists()
    if scored:
        scores = json.loads(scores_path.read_text())
        for seed, per_instance in scores.get("seeds", {}).items():
            for iid, outcome in per_instance.items():
                resolved[(iid, int(seed))] = is_resolved(outcome)
        candidates = scores.get("candidates", {})
    return Experiment(path, manifest, results, resolved, candidates, scored)


@dataclass
class VariantSummary:
    variant: str
    name: str
    n_instances: int
    seeds: list[int]
    resolve_rate: float | None
    resolve_ci: tuple[float, float] | None
    per_seed_resolve: dict[int, float]
    seed_sd: float | None
    sandbox_verified_rate: float | None
    mean_cost: float | None
    cost_per_resolved: float | None
    mean_tokens: float | None
    mean_wall_s: float | None
    mean_llm_calls: float | None
    mean_sandbox_runs: float | None
    budget_abort_rate: float | None
    excluded_invalid: int
    excluded_env: int


@dataclass
class Analysis:
    instances: list[str]
    summaries: dict[str, VariantSummary]
    hypotheses: dict[str, dict[str, Any]]
    pairwise: list[dict[str, Any]]
    per_seed_tests: list[dict[str, Any]]
    critic: dict[str, dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


def paired_instances(exps: Sequence[Experiment]) -> list[str]:
    """Instances with a valid run AND a verdict for every seed of every experiment."""
    common: set[str] | None = None
    for e in exps:
        valid = e.valid()
        ok = {
            iid
            for iid, _ in valid
            if all(
                (iid, s) in valid and (not e.scored or e.resolved.get((iid, s)) is not None)
                for s in e.manifest.seeds
            )
        }
        common = ok if common is None else common & ok
    return sorted(common or set())


def _outcome(e: Experiment, iid: str, seed: int) -> bool:
    if e.scored:
        return bool(e.resolved[(iid, seed)])
    return e.results[(iid, seed)].status == "sandbox_verified"  # proxy when unscored


def majority_outcomes(e: Experiment, ids: Sequence[str]) -> dict[str, bool]:
    return {i: majority([_outcome(e, i, s) for s in e.manifest.seeds]) for i in ids}


def summarize(e: Experiment, ids: Sequence[str], n_boot: int) -> VariantSummary:
    seeds = e.manifest.seeds
    runs = [e.results[(i, s)] for i in ids for s in seeds]
    maj = majority_outcomes(e, ids)
    per_seed = {s: sum(_outcome(e, i, s) for i in ids) / len(ids) for s in seeds} if ids else {}
    n_resolved_runs = sum(_outcome(e, i, s) for i in ids for s in seeds)
    total_cost = sum(r.cost_usd for r in runs)
    return VariantSummary(
        variant=e.variant,
        name=e.manifest.name,
        n_instances=len(ids),
        seeds=list(seeds),
        resolve_rate=mean([float(v) for v in maj.values()]),
        resolve_ci=bootstrap_ci([float(v) for v in maj.values()], n_resamples=n_boot)
        if ids
        else None,
        per_seed_resolve=per_seed,
        seed_sd=stdev(list(per_seed.values())),
        sandbox_verified_rate=mean([float(r.status == "sandbox_verified") for r in runs]),
        mean_cost=mean([r.cost_usd for r in runs]),
        cost_per_resolved=total_cost / n_resolved_runs if n_resolved_runs else None,
        mean_tokens=mean([float(r.tokens_in + r.tokens_out) for r in runs]),
        mean_wall_s=mean([r.wall_time_s for r in runs if r.wall_time_s is not None]),
        mean_llm_calls=mean([float(r.llm_calls) for r in runs]),
        mean_sandbox_runs=mean([float(r.sandbox_runs) for r in runs]),
        budget_abort_rate=mean([float(r.status == "budget_exceeded") for r in runs]),
        excluded_invalid=sum(
            1 for r in e.results.values() if not r.trace_valid and r.env_error is None
        ),
        excluded_env=sum(1 for r in e.results.values() if r.env_error is not None),
    )


def _test_row(
    label: str,
    a: Experiment,
    b: Experiment,
    ids: Sequence[str],
    *,
    n_boot: int,
    seed: int | None = None,
) -> dict[str, Any]:
    if seed is None:
        oa, ob = majority_outcomes(a, ids), majority_outcomes(b, ids)
    else:
        oa = {i: _outcome(a, i, seed) for i in ids}
        ob = {i: _outcome(b, i, seed) for i in ids}
    res: McNemarResult = mcnemar(oa, ob)
    diff, lo, hi = paired_diff_ci(
        {i: float(v) for i, v in oa.items()}, {i: float(v) for i, v in ob.items()},
        n_resamples=n_boot,
    )  # fmt: skip
    cost_a = mean([a.results[(i, s)].cost_usd for i in ids for s in a.manifest.seeds])
    cost_b = mean([b.results[(i, s)].cost_usd for i in ids for s in b.manifest.seeds])
    return {
        "comparison": label, "a": a.variant, "b": b.variant, "seed": seed, "n": res.n,
        "a_rate": mean([float(v) for v in oa.values()]),
        "b_rate": mean([float(v) for v in ob.values()]),
        "diff": diff, "diff_ci_lo": lo, "diff_ci_hi": hi,
        "b_only": res.b_only, "a_only": res.a_only, "p_value": res.p_value,
        "cost_ratio": (cost_b / cost_a) if cost_a and cost_b is not None else None,
    }  # fmt: skip


def critic_quality(e: Experiment) -> dict[str, Any]:
    """Selection quality over rounds with >= 2 applicable, non-duplicate candidates.
    Gold source: candidate scores from ``score --score-candidates``; otherwise the critic's
    own sandbox runs (stage 'critic') as a proxy."""
    rounds = sel_ok = any_ok = 0
    random_expect = 0.0
    source = "gold" if e.candidate_outcomes else "sandbox-proxy"
    for r in e.valid().values():
        if r.run_id is None:
            continue
        fs_path = e.path / "traces" / r.run_id / "final_state.json"
        if not fs_path.exists():
            continue
        fs = json.loads(fs_path.read_text())
        proxy = {
            v["candidate_index"]: v["passed"]
            for v in fs["validation_results"]
            if v["stage"] == "critic"
        }
        gold = e.candidate_outcomes.get(r.run_id, {})
        by_round: dict[int, list[dict[str, Any]]] = {}
        for c in fs["candidates"]:
            if c["diff"] and c["apply_error"] is None and c["duplicate_of"] is None:
                by_round.setdefault(c["attempt"], []).append(c)
        for cands in by_round.values():
            selected = [c for c in cands if c["selected_by"] is not None]
            if len(cands) < 2 or not selected:
                continue
            if source == "gold":
                outcome = {c["index"]: gold.get(str(c["index"])) == "resolved" for c in cands}
            else:
                if not all(c["index"] in proxy for c in cands):
                    continue
                outcome = {c["index"]: bool(proxy[c["index"]]) for c in cands}
            rounds += 1
            sel_ok += outcome[selected[0]["index"]]
            any_ok += any(outcome.values())
            random_expect += sum(outcome.values()) / len(cands)
    if rounds == 0:
        return {"source": source, "rounds": 0}
    return {
        "source": source,
        "rounds": rounds,
        "p_selected_resolves": sel_ok / rounds,
        "p_any_resolves": any_ok / rounds,
        "p_selected_given_any": sel_ok / any_ok if any_ok else None,
        "oracle_gap": (any_ok - sel_ok) / rounds,
        "lift_over_random": (sel_ok - random_expect) / rounds,
    }


def analyze(exps: Sequence[Experiment], n_boot: int = 10_000) -> Analysis:
    by_variant: dict[str, Experiment] = {}
    warnings: list[str] = []
    for e in exps:
        if e.variant in by_variant:
            raise ValueError(f"two experiments for variant {e.variant}")
        by_variant[e.variant] = e
    if len({e.manifest.prompts_sha256 for e in exps}) > 1:
        warnings.append("experiments ran with DIFFERENT prompts (prompts.py hash differs)")
    if len({e.manifest.set for e in exps}) > 1:
        warnings.append("experiments mix data sets (dev and frozen)")
    if any(e.manifest.set == "dev" for e in exps):
        warnings.append("dev-set results: for tuning only, never report as findings")
    if any(e.manifest.git_dirty for e in exps):
        warnings.append("some experiments ran from a dirty working tree")
    if not all(e.scored for e in exps):
        warnings.append(
            "unscored experiments use sandbox_verified as a PROXY for resolved; run eval.score"
        )

    ids = paired_instances(exps)
    summaries = {v: summarize(e, ids, n_boot) for v, e in sorted(by_variant.items())}

    hyp: dict[str, dict[str, Any]] = {}
    for h, (a, b) in HYPOTHESES.items():
        if a in by_variant and b in by_variant and ids:
            hyp[h] = _test_row(h, by_variant[a], by_variant[b], ids, n_boot=n_boot)
    for h, p in holm({h: row["p_value"] for h, row in hyp.items()}).items():
        hyp[h]["p_holm"] = p

    variants = sorted(by_variant)
    pairwise = [
        _test_row(f"{a} vs {b}", by_variant[a], by_variant[b], ids, n_boot=n_boot)
        for i, a in enumerate(variants) for b in variants[i + 1:] if ids
    ]  # fmt: skip
    per_seed = [
        _test_row(h, by_variant[a], by_variant[b], ids, n_boot=n_boot, seed=s)
        for h, (a, b) in HYPOTHESES.items() if a in by_variant and b in by_variant and ids
        for s in sorted(set(by_variant[a].manifest.seeds) & set(by_variant[b].manifest.seeds))
    ]  # fmt: skip
    critic = {
        v: critic_quality(e)
        for v, e in by_variant.items()
        if e.manifest.config["critic"]["enabled"]
    }
    return Analysis(ids, summaries, hyp, pairwise, per_seed, critic, warnings)


# --------------------------------------------------------------------------- outputs


def _f(x: float | None, pct: bool = False, money: bool = False) -> str:
    if x is None:
        return "—"
    if pct:
        return f"{100 * x:.1f}%"
    if money:
        return f"${x:.3f}"
    return f"{x:.3g}"


def render_markdown(an: Analysis) -> str:
    out = ["# Results", ""]
    for w in an.warnings:
        out.append(f"> **Warning:** {w}")
    out += ["", f"Paired instances analysed: **{len(an.instances)}**", "", "## Per variant", ""]
    out.append(
        "| Variant | n | Resolve (majority) | 95% CI | Per-seed | Seed SD | Sandbox-verified | "
        "Mean cost | Cost/resolved | Tokens | Wall | LLM calls | Sandbox runs | Budget aborts "
        "| Excl. invalid / env |"
    )
    out.append("|" + "---|" * 15)
    for s in an.summaries.values():
        ci = (
            f"{_f(s.resolve_ci[0], pct=True)} to {_f(s.resolve_ci[1], pct=True)}"
            if s.resolve_ci
            else "—"
        )
        seeds = ", ".join(_f(v, pct=True) for v in s.per_seed_resolve.values())
        out.append(
            f"| {s.variant} | {s.n_instances} | {_f(s.resolve_rate, pct=True)} | {ci} | {seeds} "
            f"| {_f(s.seed_sd, pct=True)} | {_f(s.sandbox_verified_rate, pct=True)} "
            f"| {_f(s.mean_cost, money=True)} | {_f(s.cost_per_resolved, money=True)} "
            f"| {_f(s.mean_tokens)} | {_f(s.mean_wall_s)}s | {_f(s.mean_llm_calls)} "
            f"| {_f(s.mean_sandbox_runs)} | {_f(s.budget_abort_rate, pct=True)} "
            f"| {s.excluded_invalid} / {s.excluded_env} |"
        )  # fmt: skip
    out += ["", "## Preregistered hypotheses (exact McNemar, majority unit, Holm)", ""]
    out.append("| H | Comparison | b-only / a-only | Δ rate | 95% CI | p | p (Holm) | Cost ratio |")
    out.append("|" + "---|" * 8)
    for h, r in an.hypotheses.items():
        out.append(
            f"| {h} | {r['b']} > {r['a']} | {r['b_only']} / {r['a_only']} | "
            f"{_f(r['diff'], pct=True)} | {_f(r['diff_ci_lo'], pct=True)} to "
            f"{_f(r['diff_ci_hi'], pct=True)} | {r['p_value']:.4f} | {r['p_holm']:.4f} "
            f"| {_f(r['cost_ratio'])}x |"
        )
    out += ["", "## Critic quality", ""]
    for v, c in an.critic.items():
        if not c.get("rounds"):
            out.append(f"- {v}: no multi-candidate rounds with outcomes ({c['source']})")
            continue
        out.append(
            f"- {v} ({c['source']}, {c['rounds']} rounds): P(selected resolves) "
            f"{_f(c['p_selected_resolves'], pct=True)}, P(any resolves) "
            f"{_f(c['p_any_resolves'], pct=True)}, P(selected | any) "
            f"{_f(c['p_selected_given_any'], pct=True)}, oracle gap "
            f"{_f(c['oracle_gap'], pct=True)}, lift over random "
            f"{_f(c['lift_over_random'], pct=True)}"
        )
    out += ["", "Exploratory pairwise comparisons and per-seed tests are in the CSV files.", ""]
    return "\n".join(out)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def frontier_plot(an: Analysis, path: Path) -> None:
    # Lazy: matplotlib is only in the `eval` dependency group, and the headless Agg
    # backend must be selected before pyplot is imported.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6, 4))
    for s in an.summaries.values():
        if s.mean_cost is None or s.resolve_rate is None:
            continue
        lo, hi = s.resolve_ci or (s.resolve_rate, s.resolve_rate)
        ax.errorbar(
            s.mean_cost, 100 * s.resolve_rate,
            yerr=[[100 * (s.resolve_rate - lo)], [100 * (hi - s.resolve_rate)]],
            fmt="o", capsize=4,
        )  # fmt: skip
        ax.annotate(s.variant, (s.mean_cost, 100 * s.resolve_rate), xytext=(6, 6),
                    textcoords="offset points")  # fmt: skip
    ax.set_xlabel("mean cost per instance-seed (USD)")
    ax.set_ylabel("resolve rate, majority over seeds (%)")
    ax.set_title(f"Cost vs resolve rate (n={len(an.instances)}, 95% bootstrap CI)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(an: Analysis, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_markdown(an))
    _write_csv(out / "variants.csv", [vars(s) for s in an.summaries.values()])
    _write_csv(out / "hypotheses.csv", list(an.hypotheses.values()))
    _write_csv(out / "pairwise_exploratory.csv", an.pairwise)
    _write_csv(out / "per_seed_tests.csv", an.per_seed_tests)
    (out / "critic_quality.json").write_text(json.dumps(an.critic, indent=2))
    frontier_plot(an, out / "frontier.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("experiments", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    ap.add_argument("--bootstrap", type=int, default=10_000)
    args = ap.parse_args(argv)
    an = analyze([load_experiment(p) for p in args.experiments], n_boot=args.bootstrap)
    write_outputs(an, args.out)
    for w in an.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"{len(an.instances)} paired instances; wrote {args.out}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
