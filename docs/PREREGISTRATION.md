# Preregistration: Architect/Critic Ablation on SWE-bench Lite

> **Fill this in BEFORE any run on the frozen subset, then commit it.**
> `eval/run_experiment.py --set frozen` refuses to start while this file is missing,
> uncommitted, modified, or still contains any `TODO(prereg)` marker, or while the
> working tree is dirty. Tune prompts and configs on `eval/data/dev_set.json` only.

- **Author:** TODO(prereg)
- **Date registered:** TODO(prereg)
- **Frozen subset:** `eval/data/subset_v1.json` (sha256 pinned in `tests/test_eval_data.py`)
- **Code version:** the commit that adds this filled-in file (recorded automatically in every
  experiment manifest)

## 1. Research question

Does an Architect/Critic reflection loop improve the rate of resolved SWE-bench Lite
instances over simpler baselines, and at what cost?

## 2. Variants (differ only in graph wiring)

| Variant | Description | Config |
|---|---|---|
| A0 | single-shot patch | `configs/eval/a0.yaml` |
| A1 | single-shot + sandbox test feedback, retry up to N | `configs/eval/a1.yaml` |
| A2 | Architect proposes K candidates, Critic selects one | `configs/eval/a2.yaml` |
| A3 | Architect → Critic → reflect → revise, max depth D | `configs/eval/a3.yaml` |

- **Model, effort, max_tokens (identical for all variants):** TODO(prereg)
- **N (A1), K (A2/A3), D (A3):** TODO(prereg)
- **Critic uses sandbox (A2/A3):** TODO(prereg)
- **Prompt file hash** (`sha256 src/agent/prompts.py`, frozen after dev tuning): TODO(prereg)

## 3. Hypotheses

- **H1:** A1 resolves more instances than A0.
- **H2:** A2 resolves more instances than A1.
- **H3:** A3 resolves more instances than A2.
- **H4:** Any gain in H1–H3 costs at most **TODO(prereg)×** the mean cost per instance of the
  simpler variant (cost-effectiveness bound).

Direction: one-sided in the stated direction, tested with two-sided exact tests (conservative).

## 4. Outcome measures

- **Primary:** an instance counts as *resolved* when the official SWE-bench harness
  (`swebench==5.0.2`, unmodified) reports it resolved for the submitted patch. The
  submitted patch is the final selected candidate's diff, or empty if none applies.
- **Primary analysis unit:** per-instance majority across seeds (resolved in ≥ 2 of 3
  seeds).
- **Secondary:** per-seed resolve rates; sandbox-verified rate; cost per instance and
  per resolved instance; tokens; wall time; sandbox runs; LLM calls; budget-abort rate;
  Critic quality (P(selected resolves | any candidate resolves), oracle gap, lift over
  random pick).

## 5. Statistical tests

- **Paired comparisons:** exact McNemar (binomial on discordant pairs) for H1–H3, on the
  primary unit. α = TODO(prereg). Holm–Bonferroni correction across H1–H3.
- **Uncertainty:** instance-level bootstrap 95% CIs (10,000 resamples, seed 0) for resolve
  rates and their differences.
- **Seed variance:** per-seed resolve rate and its SD are reported, but not tested.
- **All other pairwise comparisons:** exploratory and labelled as such.

## 6. Sample and seeds

- **Instances:** frozen subset, 75 instances, stratified by repo (`eval/data/subset_v1.json`).
- **Seeds:** 0, 1, 2. The API has no sampling seed, so these capture run-to-run
  nondeterminism.
- **Dev set** (`eval/data/dev_set.json`, disjoint): used for all tuning; never reported as a
  result.

## 7. Budget and stopping rules

- **Per-run cap (USD), from dev-set measurements:** TODO(prereg)
- **Per-experiment cap (USD, per variant):** TODO(prereg)
- **If the experiment cap stops scheduling:** remaining (instance, seed) pairs are reported
  as *not run*. Analysis uses only instances completed by **all** variants for all seeds
  (paired design). Runs aborted by the experiment cap are not counted as failures.
- **No peeking:** variants are not compared before every variant has finished.
- **Order of variant execution:** TODO(prereg)

## 8. Exclusions (decided now, applied mechanically)

- Runs with invalid traces (`validate_trace` problems) are excluded and their count is
  reported. They are re-run once on resume.
- Instances whose environment image fails to build or pull are excluded for **all**
  variants and listed.
- Instances where the harness reports an evaluation error (not a test failure) are
  excluded for all variants and listed.

## 9. Known threats to validity

- **Data contamination:** SWE-bench Lite issues and fixes predate the model's training
  cutoff.
- **Model nondeterminism:** handled with 3 seeds; no sampling control exists.
- **Small n:** 75 instances limit power. Only differences of roughly 10–15 percentage
  points are likely to be detectable.
- **Cost bias:** variants that cost more may be cut short more often by budget caps.
- **Leakage:** the agent-visible regression command uses the *paths* of test files from the
  gold test patch (existing tests only, at the base commit). It sees no test contents and
  no new tests.
- **Platform:** if images were built on arm64 rather than pulled as official x86-64, note
  which instances differ.

## 10. Deviations

Any deviation from this plan after registration is listed here with a date and reason:

- (none)
