# Résumé / portfolio bullets (template)

> **Rules for yourself**
> - **Engineering bullets** below are true *now*: each names the evidence (file or test).
>   Keep only the ones you can explain in an interview.
> - **Result bullets** contain `{{PLACEHOLDERS}}`. `make docs` fills them from `analysis/`
>   into `docs/rendered/RESUME_BULLETS.md`. Never type result numbers by hand.
> - **Report effect sizes with CIs.** If a hypothesis was not supported, say so; a clean
>   null result from a preregistered study is a strength. No "state of the art" claims:
>   this is a controlled ablation, not a leaderboard entry.

## Engineering (true today)

- Built a **multi-agent LangGraph system** for GitHub issue resolution with four
  config-selected pipelines (single-shot, test-feedback retry, K-candidate + LLM Critic,
  Architect/Critic reflection loop) that differ only in graph wiring.
  *Evidence: `src/agent/graph.py`; README diagrams verified against the compiled graphs by
  `tests/test_readme_diagrams.py`.*
- Designed a **provable-termination** graph: every cycle passes through a monotone attempt
  counter, checked structurally across all 38 valid configurations of the ablation grid.
  *Evidence: `tests/test_termination.py`.*
- Implemented **per-call token and dollar accounting** with a pre-call worst-case budget
  guard (per run and per experiment) and clean `budget_exceeded` aborts.
  *Evidence: `src/agent/budget.py`, `src/agent/llm.py`, budget-abort routing tests.*
- Built an **isolated Docker sandbox**: no network; memory, CPU and PID limits; hard
  timeouts; `git apply` patching. It runs on official SWE-bench environment images and is
  never mocked in tests.
  *Evidence: `src/agent/sandbox/docker_sandbox.py`, `tests/test_sandbox.py`.*
- Created **complete, tamper-evident JSONL tracing** for every node call (tokens, cost,
  latency, exit codes, input hashes) with automatic validity checks and secret redaction.
  Added a local read-only dashboard for inspecting runs.
  *Evidence: `src/agent/tracing.py`, `src/agent/dashboard/`.*
- Engineered a **preregistered evaluation harness** for SWE-bench Lite:
  - a stratified, checksum-frozen subset and a disjoint dev set;
  - resumable parallel runs with a persistent experiment budget;
  - official harness scoring;
  - exact McNemar tests with Holm correction and bootstrap CIs;
  - a hand-labelled failure taxonomy.

  *Evidence: `eval/`, `docs/PREREGISTRATION.md`.*
- Shipped a **human-gated live pilot** that turns real GitHub issues into draft PRs, with:
  - a repository allowlist and detection of AI-contribution policies;
  - per-PR approval cryptographically bound to the patch contents;
  - rate limits, volume caps and full AI disclosure.

  *Evidence: `pilot/`, `docs/PILOT.md`.*
- Maintained **{{N_TESTS}} tests** under ruff and mypy `--strict`, including real-Docker
  integration tests.

## Results (fill from analysis, never by hand)

- Ran a **preregistered ablation** on {{N_INSTANCES}} SWE-bench Lite instances × 3 seeds:
  the full reflection loop (A3) resolved {{RESOLVE_A3}} (95% CI {{CI_LO_A3}}–{{CI_HI_A3}})
  versus {{RESOLVE_A0}} for single-shot (A0), at {{COST_A3}} versus {{COST_A0}} mean cost
  per instance.
- Found that **sandbox test feedback** (A1 vs A0) changed the resolve rate by {{H1_DIFF}}
  (95% CI {{H1_CI_LO}} to {{H1_CI_HI}}, Holm-adjusted p = {{H1_P_HOLM}}), and the
  **Critic** (A2 vs A1) by {{H2_DIFF}} (p = {{H2_P_HOLM}}).
- Measured **Critic selection quality**: it picked a resolving candidate in
  {{CRITIC_A2_P_SEL_GIVEN_ANY}} of rounds where one existed (lift over random
  {{CRITIC_A2_LIFT}}).
- *Optional, if supported:* identified **{{TOP_FAILURE_CATEGORY}}** as the dominant failure
  mode from a hand-labelled taxonomy of {{N_LABELLED}} failures.
- *Optional, live pilot:* opened {{N_PILOT_PRS}} human-approved draft PRs on real
  repositories ({{N_PILOT_MERGED}} merged).
