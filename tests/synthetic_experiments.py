"""Build synthetic experiment directories (manifest, results, scores, final states) for
testing analysis code without running anything."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.experiment import Manifest, RunResult


def make_experiment(
    root: Path,
    variant: str,
    outcomes: dict[str, list[bool | None]],  # instance -> per-seed resolved (None = error)
    *,
    cost: float = 0.1,
    critic: bool = False,
    prompts_sha: str = "p" * 64,
    set_name: str = "frozen",
    invalid: set[tuple[str, int]] | None = None,
    candidates: dict[str, list[dict[str, Any]]] | None = None,
    candidate_outcomes: dict[str, dict[str, str]] | None = None,
    scored: bool = True,
    localized: list[str] | None = None,
) -> Path:
    path = root / f"{variant.lower()}-exp"
    (path / "traces").mkdir(parents=True)
    seeds = sorted({s for v in outcomes.values() for s in range(len(v))})
    manifest = Manifest(
        name=f"{variant.lower()}-exp", variant=variant, set=set_name,
        data_file="subset_v1.json", data_sha256="d" * 64,
        config={"critic": {"enabled": critic, "uses_sandbox": False}},
        config_hash="c" * 64, seeds=seeds, git_sha="g" * 40, git_dirty=False,
        prompts_sha256=prompts_sha, prereg=None, swebench_version="5.0.2", created="now",
    )  # fmt: skip
    (path / "manifest.json").write_text(manifest.model_dump_json())
    invalid = invalid or set()
    lines: list[str] = []
    score_seeds: dict[str, dict[str, str]] = {}
    for iid, per_seed in outcomes.items():
        for seed, resolved in enumerate(per_seed):
            run_id = f"{variant}-{iid}-s{seed}"
            ok = (iid, seed) not in invalid
            lines.append(RunResult(
                instance_id=iid, seed=seed, run_id=run_id, variant=variant,
                status="sandbox_verified" if resolved else "unverified",
                trace=f"traces/{run_id}.jsonl", trace_valid=ok,
                problems=[] if ok else ["last record is not run_end"], cost_usd=cost,
                tokens_in=1000, tokens_out=200, llm_calls=2, sandbox_runs=3, wall_time_s=5.0,
                attempts=1, model_patch="diff", aborted_by_experiment_cap=False,
                env_error=None, finished_at="now",
            ).model_dump_json())  # fmt: skip
            outcome = "error" if resolved is None else ("resolved" if resolved else "unresolved")
            score_seeds.setdefault(str(seed), {})[iid] = outcome
            fs = {
                "localized_files": localized or ["a.py"],
                "candidates": (candidates or {}).get(run_id, []),
                "validation_results": [],
                "error": None,
            }
            (path / "traces" / run_id).mkdir()
            (path / "traces" / run_id / "final_state.json").write_text(json.dumps(fs))
    (path / "results.jsonl").write_text("\n".join(lines) + "\n")
    if scored:
        scores = {"seeds": score_seeds, "candidates": candidate_outcomes or {}}
        (path / "scores.json").write_text(json.dumps(scores))
    return path


def cand(index: int, attempt: int = 1, selected: bool = False) -> dict[str, Any]:
    return {
        "index": index, "attempt": attempt, "diff": f"d{index}", "apply_error": None,
        "duplicate_of": None, "selected_by": "critic" if selected else None,
    }  # fmt: skip
