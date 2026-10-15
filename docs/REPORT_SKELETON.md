# Does Reflection Help? An Ablation Study of Architect/Critic LLM Agents for GitHub Issue Resolution

**Author:** {{AUTHOR}} · **Supervisor:** {{SUPERVISOR}} · **Institution:** {{INSTITUTION}} · **Date:** {{DATE}}

> **How to use this skeleton.**
> - **Drafted sections** (Introduction, Method, Experimental Setup, Threats) describe the
>   implemented system and should stay true as long as the code does.
> - **Results-dependent text** is a `{{PLACEHOLDER}}`. `make docs` fills those from
>   `analysis/` into `docs/rendered/REPORT_SKELETON.md`. Never type result numbers by hand.
> - **Guidance** is in *italic notes* below. Delete the notes before submission.

---

## Abstract

*Template: write it last, from the filled results. 150–250 words.*

LLM agents for repository-level issue resolution increasingly add self-critique and
reflection, but such loops add cost and their benefit is rarely isolated. We compare four
agent pipelines that differ **only** in graph wiring:
- single-shot generation (A0);
- retry with sandbox test feedback (A1);
- K candidates with an LLM Critic selecting one (A2);
- a full Architect → Critic → reflect → revise loop (A3).

On a preregistered, stratified subset of {{N_INSTANCES}} SWE-bench Lite instances (3 seeds,
official scoring), resolve rates were {{RESOLVE_A0}}, {{RESOLVE_A1}}, {{RESOLVE_A2}} and
{{RESOLVE_A3}} respectively. *State which preregistered hypotheses were supported, the
cost ratio of the best variant, and the main failure mode. Do not overclaim. If a
difference is not significant, say so.*

## 1. Introduction

- **Motivation.** Resolving real GitHub issues needs repository navigation, editing and
  validation. Published agent systems often combine many mechanisms at once (tools,
  retrieval, multiple samples, critics, reflection), which makes it hard to attribute
  their gains.
- **Research question.** Does an Architect/Critic reflection loop improve the rate of
  sandbox-verified (and officially resolved) patches over simpler baselines, and at what
  cost?
- **Approach.** A controlled ablation: one codebase, state model, toolset, sandbox,
  prompts and model, with four wirings selected by configuration. The study is
  preregistered and uses paired statistical tests, and every run is fully traced.
- **Contributions.**
  1. An evaluated, fully traced multi-agent pipeline where the variants differ only in
     wiring.
  2. A preregistered paired comparison on SWE-bench Lite with official scoring.
  3. An analysis of cost-effectiveness and Critic selection quality.
  4. A failure taxonomy.
  5. A controlled, human-gated live pilot on real issues.

## 2. Related Work

*Verify every citation (authors, venue, year) against the original before submission. No
numbers from other papers are quoted here on purpose: if you add any, take them from the
primary source.*

- **Benchmarks.**
  - SWE-bench (Jimenez et al., *ICLR 2024*) frames issue resolution as producing a patch
    that makes held-out tests pass.
  - SWE-bench Lite is its smaller, self-contained subset, which this study samples from.
  - Data contamination is a known concern for any benchmark built from public
    repositories.
- **Agent systems.**
  - **SWE-agent** (Yang et al., *NeurIPS 2024*) argues that the *agent–computer
    interface*, meaning the tools and their feedback, matters as much as the model.
  - **OpenHands**, formerly OpenDevin (Wang et al., 2024), is an open platform for
    generalist software-engineering agents with sandboxed execution.
  - **Aider** (Gauthier) is a widely used open-source pair-programming tool. Its edit
    formats, such as search/replace blocks, and its repository map inform practical
    patch generation. This study uses search/replace edits for the same reason.
  - **Agentless** (Xia et al., 2024) shows that a fixed localize → repair → validate
    pipeline can be competitive with autonomous agents, motivating simple baselines like
    A0/A1.
- **Self-correction.**
  - Reflexion (Shinn et al., *NeurIPS 2023*) and Self-Refine (Madaan et al., *NeurIPS
    2023*) report gains from verbal feedback and iterative refinement.
  - Self-Debugging (Chen et al., *ICLR 2024*) uses execution feedback for code.
  - Huang et al. (*ICLR 2024*) find that LLMs often cannot self-correct reasoning
    *without external feedback*. This motivates separating test feedback (A1) from
    LLM-critic feedback (A2/A3).
- **Positioning.** Rather than proposing a new agent, this work isolates one design
  choice, the Critic and reflection loop, under controlled conditions, with
  preregistration and paired tests.

## 3. Method

### 3.1 Shared pipeline
Every variant runs `localizer → architect → patcher → select → validator`, operating on a
single typed state (`AgentState`).
- **Localizer:** chooses up to *M* files from the repository's Python file list.
- **Architect:** produces K candidate patches as exact search/replace edits with
  rationales, in one structured-output call.
- **Patcher:** applies each candidate to a clean checkout in the sandbox and records its
  diff or a precise apply error. It also marks duplicate candidates.
- **Select:**
  - *Critic on:* an LLM Critic scores each candidate on a 1–5 rubric (correctness
    likelihood, minimality, test impact, risk) and picks one. The output is
    pydantic-validated JSON. An invalid pick falls back deterministically to the highest
    correctness score, then the lowest risk.
  - *Critic off:* a seeded-random pick among the applicable, non-duplicate candidates.
- **Validator:** resets the sandbox, applies the selected diff with `git apply` and runs
  the agent-visible test command.

### 3.2 Variants
| | Candidates (K) | Selection | After a failed validation | Max architect rounds |
|---|---|---|---|---|
| A0 | 1 | trivial | stop | 1 |
| A1 | 1 | trivial | retry with test output as feedback | N |
| A2 | K | Critic | stop | 1 |
| A3 | K | Critic | Critic writes a critique; Architect revises | D + 1 |

### 3.3 Termination and budget
- **Termination.** Every graph cycle passes through the Architect, which increments an
  attempt counter, and every back-edge requires `attempt_count < max_rounds`.
  Termination is verified structurally for all valid configurations in the ablation
  grid.
- **Budget.** A guard checks a worst-case cost estimate before each LLM call, against
  both a per-run and a per-experiment cap. Exceeding either ends the run cleanly as
  `budget_exceeded`.

### 3.4 Observability
- **Trace contents:** every node invocation writes a trace record (input hash, tokens,
  cost, latency, sandbox exit codes). Each run also records its configuration, git
  commit, package versions and full final state.
- **Validity:** a run is valid only if its trace is complete and internally consistent.

## 4. Experimental Setup

- **Data.**
  - {{N_INSTANCES}} instances drawn from SWE-bench Lite (test split, revision pinned) by
    stratified random sampling per repository with seed 20261015. The subset is frozen by
    checksum.
  - A disjoint dev set of 15 instances was used for **all** tuning.
- **Environment.**
  - Official per-instance SWE-bench images, with no network access during agent runs.
  - **Agent-visible test signal:** the official test command run at the base commit
    *without* the gold test patch, i.e. existing tests in the touched test files only.
    Hidden tests are never exposed.
- **Model.** {{MODEL}}, effort {{EFFORT}}, identical for every role and variant.
  - Knobs: N = {{N_RETRIES}}, K = {{K}}, D = {{D}}; critic-uses-sandbox
    {{CRITIC_SANDBOX}}.
  - Caps: per-run {{CAP_RUN}}, per-experiment {{CAP_EXPERIMENT}}.
- **Seeds.** 0, 1, 2. The API exposes no sampling seed, so seeds capture run-to-run
  nondeterminism.
- **Outcome.**
  - **Resolved:** as reported by the unmodified official harness for the submitted patch,
    which is the final selected candidate's diff.
  - **Primary unit:** per-instance majority over seeds.
- **Tests.**
  - Exact McNemar for H1 (A1 > A0), H2 (A2 > A1) and H3 (A3 > A2), Holm-corrected,
    α = {{ALPHA}}.
  - Instance-level bootstrap 95% CIs (10,000 resamples).
  - H4: cost ratio ≤ {{H4_THRESHOLD}}×.
- **Preregistration.** Hypotheses, metrics, tests, exclusions and stopping rules were
  committed before any frozen run (commit {{PREREG_COMMIT}}).

## 5. Results

*Report every preregistered hypothesis, including those that are not supported. Use effect
sizes with CIs, not just p-values.*

### 5.1 Resolve rates (n = {{N_INSTANCES}} paired instances)

| Variant | Resolved (majority) | 95% CI | Seed SD | Sandbox-verified | Mean cost | Cost / resolved |
|---|---|---|---|---|---|---|
| A0 | {{RESOLVE_A0}} | {{CI_LO_A0}} – {{CI_HI_A0}} | {{SEED_SD_A0}} | {{SV_RATE_A0}} | {{COST_A0}} | {{COST_PER_RESOLVED_A0}} |
| A1 | {{RESOLVE_A1}} | {{CI_LO_A1}} – {{CI_HI_A1}} | {{SEED_SD_A1}} | {{SV_RATE_A1}} | {{COST_A1}} | {{COST_PER_RESOLVED_A1}} |
| A2 | {{RESOLVE_A2}} | {{CI_LO_A2}} – {{CI_HI_A2}} | {{SEED_SD_A2}} | {{SV_RATE_A2}} | {{COST_A2}} | {{COST_PER_RESOLVED_A2}} |
| A3 | {{RESOLVE_A3}} | {{CI_LO_A3}} – {{CI_HI_A3}} | {{SEED_SD_A3}} | {{SV_RATE_A3}} | {{COST_A3}} | {{COST_PER_RESOLVED_A3}} |

### 5.2 Preregistered hypotheses

| | Comparison | Discordant (b-only / a-only) | Δ resolve | 95% CI | p | p (Holm) | Cost ratio |
|---|---|---|---|---|---|---|---|
| H1 | A1 > A0 | {{H1_B_ONLY}} / {{H1_A_ONLY}} | {{H1_DIFF}} | {{H1_CI_LO}} – {{H1_CI_HI}} | {{H1_P}} | {{H1_P_HOLM}} | {{H1_COST_RATIO}} |
| H2 | A2 > A1 | {{H2_B_ONLY}} / {{H2_A_ONLY}} | {{H2_DIFF}} | {{H2_CI_LO}} – {{H2_CI_HI}} | {{H2_P}} | {{H2_P_HOLM}} | {{H2_COST_RATIO}} |
| H3 | A3 > A2 | {{H3_B_ONLY}} / {{H3_A_ONLY}} | {{H3_DIFF}} | {{H3_CI_LO}} – {{H3_CI_HI}} | {{H3_P}} | {{H3_P_HOLM}} | {{H3_COST_RATIO}} |

*For each hypothesis, one paragraph: the effect, its CI, whether it survives Holm, and
what the cost ratio implies for H4. If CIs overlap zero, write "no evidence of a
difference at this sample size", not "no difference".*

### 5.3 Cost and efficiency
Figure: `analysis/frontier.png`, cost versus resolve rate with 95% CIs.

Tokens, latency, LLM calls and sandbox runs per variant are in `analysis/variants.csv`.
Budget-abort rates: A0 {{BUDGET_ABORT_A0}}, A1 {{BUDGET_ABORT_A1}}, A2 {{BUDGET_ABORT_A2}},
A3 {{BUDGET_ABORT_A3}}.

### 5.4 Critic quality
| | Rounds | P(selected resolves \| any resolves) | Oracle gap | Lift over random pick | Source |
|---|---|---|---|---|---|
| A2 | {{CRITIC_A2_ROUNDS}} | {{CRITIC_A2_P_SEL_GIVEN_ANY}} | {{CRITIC_A2_ORACLE_GAP}} | {{CRITIC_A2_LIFT}} | {{CRITIC_A2_SOURCE}} |
| A3 | {{CRITIC_A3_ROUNDS}} | {{CRITIC_A3_P_SEL_GIVEN_ANY}} | {{CRITIC_A3_ORACLE_GAP}} | {{CRITIC_A3_LIFT}} | {{CRITIC_A3_SOURCE}} |

*Interpretation: the oracle gap bounds how much a perfect selector could add. A lift near
zero means the Critic picks no better than chance.*

### 5.5 Seed variance
*Per-seed resolve rates are in `analysis/per_seed_tests.csv`. Discuss whether conclusions
hold on every seed.*

## 6. Failure Analysis

{{N_LABELLED}} unresolved runs were sampled evenly across variants (seeded) and labelled by
hand (`eval/failure_taxonomy.py`).

{{TAXONOMY_TABLE}}

*Discuss:*
- *the dominant category per variant;*
- *whether reflection mainly fixes "wrong fix" errors or cannot recover from
  localization errors;*
- *two short illustrative cases, linked by run_id to their traces.*

## 7. Threats to Validity

- **Data contamination.** SWE-bench Lite issues and fixes are public and predate the
  model's training cutoff. Absolute rates may be inflated. Relative comparisons are less
  affected, because every variant shares the model.
- **Model nondeterminism.** There is no seed or temperature control. Three seeds and a
  majority unit mitigate this but do not remove it; seed SD is reported.
- **Small n.** With {{N_INSTANCES}} instances, McNemar has limited power, roughly enough
  only for differences of 10–15+ percentage points. Null results are therefore weak
  evidence of no effect.
- **Cost bias and budget aborts.** Costlier variants hit caps more often, and an aborted
  run counts as unresolved. Caps were set from dev-set measurements, and abort rates are
  reported per variant.
- **Leakage through the regression signal.** The agent's test command reveals the
  *paths* of the test files that the gold test patch touches. It never reveals test
  contents or new tests.
- **Weak feedback signal.** Existing tests may already fail at the base commit, or may not
  exercise the bug at all. This understates what A1/A3 could gain from a stronger
  signal.
- **Single model, same model for every role.** The Critic and Architect share one model,
  so correlated blind spots are likely. Results may not transfer to other model families.
- **Tuning.** All tuning used the disjoint dev set only. Prompts were frozen (hash
  {{PROMPTS_SHA}}) before frozen runs.
- **Platform.** {{PLATFORM_NOTE}} (x86-64 native, or arm64 emulation).

## 8. Conclusion

*Answer the research question in two or three sentences, directly and with the effect
size and cost. Then name limitations and next steps: a stronger test signal (reproduction
tests), a different-model critic, larger n, and other model families.*

## Appendix

- A. Preregistration (docs/PREREGISTRATION.md, commit {{PREREG_COMMIT}})
- B. Prompts (src/agent/prompts.py)
- C. Trace schema (src/agent/tracing.py)
- D. Live-pilot outcomes (`out/pilot/pr_outcomes.md`), **not** part of the benchmark
- E. Reproduction commands (README, docs/EVAL_SETUP.md)

## References (verify before submission)

- Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K.
  *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.
- Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., Press, O.
  *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* NeurIPS 2024.
- Wang, X., et al. *OpenHands: An Open Platform for AI Software Developers as Generalist
  Agents.* arXiv 2407.16741, 2024.
- Gauthier, P. *Aider: AI pair programming in your terminal.* https://aider.chat
- Xia, C. S., Deng, Y., Dunn, S., Zhang, L. *Agentless: Demystifying LLM-based Software
  Engineering Agents.* arXiv 2407.01489, 2024.
- Shinn, N., et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.*
  NeurIPS 2023.
- Madaan, A., et al. *Self-Refine: Iterative Refinement with Self-Feedback.* NeurIPS 2023.
- Chen, X., Lin, M., Schärli, N., Zhou, D. *Teaching Large Language Models to Self-Debug.*
  ICLR 2024.
- Huang, J., et al. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR 2024.
- McNemar, Q. *Note on the sampling error of the difference between correlated proportions
  or percentages.* Psychometrika, 1947.
- Holm, S. *A simple sequentially rejective multiple test procedure.* Scandinavian Journal
  of Statistics, 1979.
- Efron, B., Tibshirani, R. *An Introduction to the Bootstrap.* Chapman & Hall, 1993.
