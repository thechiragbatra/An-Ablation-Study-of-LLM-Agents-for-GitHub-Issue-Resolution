# Evaluation setup

How to run the Phase 3 harness on SWE-bench Lite: the agent runs, official scoring, and
the analysis.

## 1. Where to run

Official SWE-bench environment images are **x86-64**. One image per instance, 1–5 GB
each.

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| Disk (Docker data root) | 120 GB | 250 GB | 75 frozen + 15 dev instance images, plus harness logs |
| RAM | 16 GB | 32–64 GB | each sandbox gets `mem_limit` (4 GB in `configs/eval/*.yaml`) |
| CPU | 8 | 16 | `--workers` × 2 CPUs per sandbox, plus scoring |
| Docker | 24+ | latest | the official harness uses `docker buildx` / the Docker SDK |
| Network | during image pulls only | | sandboxes run with `network_mode=none` |

**Recommended host:** a **x86-64 Linux VM** (e.g. 16 vCPU / 64 GB / 250 GB SSD). Images run
natively and every instance behaves as in the official leaderboard setup.

**Apple Silicon (arm64):**
- `eval.swebench_adapter.ensure_image` pulls the **official x86-64 image**, which runs
  under emulation. This is correct but slow: expect 3–10× longer test runs.
- No instance is dropped, so there is **no selection bias**.
- Building native arm64 images is possible with `swebench images build` and a task
  repo. It is **not used**: some instances do not build on arm64, and dropping them
  would bias the subset.
- Raise Docker Desktop's memory to ≥ 16 GB and its disk image to ≥ 150 GB first.
  As of this writing the development Mac has **2 GB free**, so it cannot run real
  instances.

Check before starting:

```bash
df -h "$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /)"
docker info -f 'arch={{.Architecture}} cpus={{.NCPU}} mem={{.MemTotal}}'
```

## 2. Install

```bash
uv sync --all-groups      # agent + eval group (swebench==5.0.2, datasets, matplotlib)
export ANTHROPIC_API_KEY=...   # never commit it
```

## 3. Data (already frozen, do not regenerate)

| File | What | Checksum test |
|---|---|---|
| `eval/data/subset_v1.json` | 75 benchmark instances, stratified by repo | `tests/test_eval_data.py` |
| `eval/data/dev_set.json` | 15 tuning instances, disjoint from the subset | `tests/test_eval_data.py` |

- **Source:** `SWE-bench/SWE-bench_Lite`, split `test`, revision
  `b0dde1093fe417d83b7184254edf8199c1f0dff5`.
- **Contents:** full records, so no Hugging Face access is needed for runs or scoring.
- **Overwrite protection:** `eval.subset` refuses to overwrite either file.

## 4. Workflow

```bash
# (a) Tune ONLY on the dev set. No preregistration is needed.
uv run python -m eval.run_experiment --config configs/eval/a0.yaml --set dev \
    --seeds 0 --workers 2 --out out/experiments/a0-dev
uv run python -m eval.score --experiment out/experiments/a0-dev
uv run python -m eval.analyze out/experiments/a0-dev --out out/analysis-dev

# (b) Measure cost on the dev set for all four variants, then set budget.per_run_usd and
#     per_experiment_usd in configs/eval/*.yaml from those measurements.

# (c) Fill in docs/PREREGISTRATION.md (every TODO(prereg)) and commit it together with
#     the final configs and prompts. The working tree must be clean.

# (d) Frozen runs. Run each variant to completion before looking at any results.
for v in a0 a1 a2 a3; do
  uv run python -m eval.run_experiment --config configs/eval/$v.yaml --set frozen \
      --seeds 0 1 2 --workers 4 --out out/experiments/$v-frozen
done
# Re-running the same command resumes: finished (instance, seed) pairs are skipped.

# (e) Official scoring. --score-candidates enables the Critic-quality analysis (A2/A3).
for v in a0 a1; do uv run python -m eval.score --experiment out/experiments/$v-frozen; done
for v in a2 a3; do
  uv run python -m eval.score --experiment out/experiments/$v-frozen --score-candidates
done

# (f) Analysis and failure taxonomy.
uv run python -m eval.analyze out/experiments/a{0,1,2,3}-frozen --out analysis/
uv run python -m eval.failure_taxonomy export out/experiments/a{0,1,2,3}-frozen \
    --n 30 --seed 0 --out analysis/failures_to_label.csv
#   ...label the `category` column by hand, save as failures_labeled.csv...
uv run python -m eval.failure_taxonomy summarize analysis/failures_labeled.csv
```

**Verify a new host first.** This pulls one image, checks the sandbox and the regression
command, and checks that the official harness resolves a gold patch:

```bash
uv run pytest -m swebench --run-swebench tests/test_swebench_integration.py
```

## 5. What the agent sees (and what it never sees)

- **Sandbox:** each run starts a container from the instance's official image, with the
  repo at `/testbed` at the base commit and `network_mode=none`.
- **Test signal: regression tests.** This is the official test command run at the base
  commit **without** the gold test patch, so only the *existing* tests in the touched
  test files run:
  - test files that the gold patch creates are removed from the command;
  - if nothing remains, a per-repo import smoke test is used instead (3 of 300 Lite
    instances).
- **Hidden from the agent:**
  - the gold patch, the test patch contents, `FAIL_TO_PASS` / `PASS_TO_PASS`, and hints;
  - `pip install` lines (the image already has the environment).
- **Known leak:** the agent-visible command reveals test *file paths*. This is
  documented in the preregistration.
- **Submitted patch:** the final selected candidate's diff, even if the sandbox status
  is `unverified`, because the hidden tests may still pass. If no candidate applies, the
  submitted patch is empty and counts as unresolved.

## 6. Scoring details

- **Harness version:** `eval.score` runs `python -m swebench.harness.run_evaluation` from
  `swebench==5.0.2`, unmodified, once per seed.
- **Pinned data:** the dataset is passed as a local JSONL written from the frozen
  records, so scoring uses the pinned revision.
- **Outputs:**
  - harness logs: `<experiment>/scoring/logs/run_evaluation/…`
  - official reports: `<experiment>/scoring/<model>.<run_id>.json`
  - parsed outcomes: `<experiment>/scores.json`
- **Outcome mapping:**

| Harness outcome | Treated as |
|---|---|
| `resolved` | resolved |
| `unresolved`, `empty_patch` | not resolved |
| `error`, `infra_failure`, `incomplete` | **excluded for all variants** (paired design) |

- **Runtime:** a harness run takes roughly 2–15 minutes per instance on x86-64,
  depending on the repo (django and sympy are slow). `--max-workers` parallelizes.

## 7. Cleanup

```bash
docker ps -a --filter label=ablation.sandbox -q | xargs -r docker rm -f   # agent sandboxes
uv run python -m swebench.harness.remove_containers --help                # harness leftovers
docker image prune                                                        # dangling layers
docker images 'swebench/*' -q | xargs -r docker rmi                       # all env images (re-pull later)
```

## 8. Cost (measure before trusting)

Real-repo prompts are large. The localizer lists every `*.py` path, and the Architect
reads whole files.

- **Rough prior with Opus 5.5:** about $7 per instance per seed across all four variants,
  so 75 × 3 seeds ≈ $1.6k. A3 dominates.
- **Measure the real numbers on the dev set** before filling in the preregistration.
- **Cost levers** (prompt caching, smaller files, Sonnet 5.5) must be applied **before
  freezing** and identically for every variant.
