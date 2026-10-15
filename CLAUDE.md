# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A final-year project built as an **evaluated engineering system**. Research question: does an Architect/Critic reflection loop improve the rate of sandbox-verified (and SWE-bench-resolved) patches over simpler baselines, and at what cost?

Four LangGraph pipelines (A0–A3) share one state, toolset, Docker sandbox, budget guard and tracer, and differ **only in graph wiring, selected by config** (`variant`, `K`, `D`, `N`, `critic.enabled`, `critic.uses_sandbox`). Never introduce variant behaviour through code edits; it must come from config. No benchmark results exist yet.

## Project rules (from the author; follow them)

- **Plan mode first.** Start any substantial change in plan mode and propose files, interfaces and tests before implementing. After a change, run `make lint typecheck test`, show the output, and summarize the decisions the author should review.
- **Never mock sandbox execution.** Sandbox tests use real Docker, and they **fail rather than skip** when Docker is unavailable.
- **FakeLLM (`tests/fake_llm.py`) was meant only for graph-routing unit tests.** It is currently also used in runner, publisher, dashboard and taxonomy tests and in `scripts/demo_traces.py`. That extension is awaiting the author's decision, so don't spread it further without asking.
  - It replaces only the network call: pricing, budget and tracing still run through `AccountedLLM`.
- **Nodes return partial `StateUpdate`s and never mutate their input.** Immutability tests deep-copy the state around every node.
- **Secrets come from env vars only** (`ANTHROPIC_API_KEY`, `GITHUB_TOKEN`). Never log or commit them; `agent.tracing.redact` scrubs traces, artifacts and errors.
- **Static analysis is limited to code health and logical bugs.** No exploit or payload generation against live infrastructure.
- **Every run must emit a complete trace.** Untraced or invalid runs are excluded from analysis.
- **Never fabricate results.** Report and résumé numbers are `{{PLACEHOLDER}}`s filled only by `make docs` (`eval/render_docs.py`) from real analysis outputs.
- **Benchmark discipline:**
  - `eval/data/subset_v1.json` is FROZEN; a checksum test fails if it changes, so create a `subset_v2.json` instead.
  - Tune prompts and configs only on `eval/data/dev_set.json`.
  - Frozen runs are refused until `docs/PREREGISTRATION.md` has no `TODO(prereg)` markers, is committed, and the working tree is clean.

## Commands

```bash
uv sync --all-groups                 # eval group pulls swebench==5.0.2, datasets, matplotlib
make check                           # = make lint typecheck test
make lint                            # ruff check . && ruff format --check .
make typecheck                       # mypy . (strict, includes tests/)
make format

uv run pytest tests/test_graph_phase2.py::test_a3_max_depth_exit   # single test
uv run pytest -k critic -q

# Opt-in tests that hit real external services (skipped by default):
uv run pytest -m live --run-live              # real Anthropic API (needs ANTHROPIC_API_KEY)
uv run pytest -m swebench --run-swebench      # pulls a real SWE-bench image (GBs of disk)
uv run pytest -m github --run-github          # READ-ONLY GitHub calls (needs GITHUB_TOKEN)

uv run python -m agent.run --config configs/a1.yaml --instance tests/fixtures/instances/calc_001.json
make demo-traces dashboard                    # scripted runs -> local dashboard on 127.0.0.1:8765

make eval-dev | eval-full | analyze | docs    # see docs/EVAL_SETUP.md (needs an x86 host, ~250 GB)
```

- **Test runtime.** The full suite takes about 1.5 minutes and builds two small fixture images (`ablation-fixture:py3.12`, `…-prebuilt`) on first run.
- **Fixture isolation.** `tests/fixtures/calc_repo` has its own `tests/` dir and is excluded from pytest, ruff and mypy.

## Architecture (the parts that span files)

### Agent graph (`src/agent/`)

- **Topology.** Every variant is `localizer → architect → patcher → select → validator`.
  - `select` is the `critic` node (LLM rubric selection) when `critic.enabled`, otherwise the `selector` node (trivial, or seeded-random among applicable candidates). Traces name whichever ran.
  - A3 with the critic on adds `critic_reflect → architect`.
  - The `publisher` leaf exists only when `publisher.enabled` (pilot configs).
- **Failure policy has one source of truth.** `agent/policy.py:next_after_failure` returns retry, reflect or stop. Both the validator (to decide `running` vs `unverified`) and the router in `graph.py` use it.
- **Termination invariant.** Every cycle passes through `architect`, which increments `attempt_count`, and every back-edge requires `attempt_count < config.max_rounds()`. `tests/test_termination.py` checks this structurally over the whole ablation grid.
  - Conditional edges declare explicit path maps so `get_graph()` introspection is exact.
  - **When you change wiring, also update the README mermaid diagrams.** `tests/test_readme_diagrams.py` compares them to the compiled graphs.
- **State.** `AgentState` (`state.py`) is the single source of truth.
  - `candidates` uses an upsert-by-index reducer: the patcher fills in `diff`/`apply_error`/`duplicate_of`, and `select` sets `selected_by`.
  - `validation_results` and `critic_notes` are append-only.
  - Validation results carry `stage: "critic" | "final"`, and only `final` is the verdict.
  - Gold benchmark data must never enter state.
- **Patch flow.**
  1. The architect emits search/replace edits and reads files at the **base commit** (`read_base_file`).
  2. The patcher applies each candidate of the round to a clean tree, records its diff, and resets.
  3. The validator re-applies the selected diff with `git apply` (as SWE-bench does) and runs `test_command`.
- **Node execution.** Nodes are wrapped by `graph.traced_call`, which turns `BudgetExceeded` into status `budget_exceeded` and any other exception into `error`. Nodes therefore never raise out of the graph; abort statuses route to END.
- **LLM layer.** `llm.AccountedLLM` wraps any `LLMProvider`:
  - a worst-case pre-call budget check against a run ledger whose parent is the experiment ledger;
  - pricing from `configs/pricing.yaml` (an unknown model is an error);
  - a trace record plus a full prompt/response artifact;
  - refusals become `LLMRefusal`.
  - Server-side model fallbacks are deliberately **not** used (they would confound the comparison). Structured output uses `messages.parse(output_format=...)` on anthropic SDK 1.x.
- **Tracing.** `Tracer` records an LLM call or sandbox exec only inside an open node scope; recording outside one raises, by design.
  - `run_instance` wraps sandbox setup and teardown as the `_sandbox_setup` / `_sandbox_teardown` pseudo-nodes.
  - `run_end` points to `<run_id>/final_state.json`.
  - `validate_trace` checks seq contiguity and that the totals reconcile with the node records.
- **Sandbox.** `DockerSandbox` (`network_mode=none`, resource limits, `timeout -s KILL`) has two modes:
  - **Copy mode:** a local repo dir is tarred into `/workspace` (used for fixtures).
  - **Prebuilt mode:** the repo is already at `workdir` in the image (SWE-bench `/testbed`).
  - Resets use `git clean -fd`, **not** `-fdx`: ignored build artifacts such as compiled extensions must survive.
  - Untracked files shipped in the image are added to `.git/info/exclude` so they never appear in diffs.

### Evaluation harness (`eval/`)

- **SWE-bench 5.0.2 is data-driven.** Dataset rows carry `image` and `eval_script`; the old spec maps and `get_test_directives` no longer exist.
- **Adapter.** `swebench_adapter.to_instance` derives the agent-visible **regression command**: the official test command at the base commit, without the gold test patch, with test files the patch creates removed (using `diff --git` headers, so empty new files are caught). If nothing remains, it falls back to an import smoke test.
  - Activation and `export` lines become `env_setup`; django's locale generation becomes `setup_script`; `pip install` lines are dropped.
- **Runner.** `experiment.run_experiment`:
  - schedules seed-major and resumes from `results.jsonl` (last line wins);
  - restores the experiment ledger from the money already spent;
  - refuses to resume if the treatment hash changed. `base_config_hash` excludes seed, trace_dir and `per_experiment_usd`.
  - **Submitted patch** = the final selected diff, even when the sandbox status is unverified.
- **Scoring.** `score.py` runs the unmodified official harness, passing a **local JSONL of the frozen records** so the dataset revision stays pinned.
- **Analysis.** `analyze.py` uses a paired design: an instance counts only if it has verdicts for every seed of every variant. The primary unit is the per-instance majority. Tests are exact McNemar with Holm correction, plus bootstrap CIs (pure Python, in `stats.py`).

### Live pilot (`pilot/`)

- **The agent graph never touches GitHub.** The `publisher` node only writes a local package under `out/pending/` for `sandbox_verified` runs.
- **Publishing** (`pilot.publish`) is a dry run by default. `--publish` additionally requires:
  - `APPROVED.json`, whose sha256 must match `PendingPackage.content_sha256()`. The body's `@@REVIEWER@@` placeholder is filled at publish time, so the hash stays stable.
  - every gate in `pilot/gates.py` to pass.
- **PR shape.** `RestGitHub.create_draft_pull` has no non-draft path. PR heads are always new `ai-patch/…` branches, never a default branch.

## Environment notes

- **Python** is pinned to 3.12 via `.python-version`; `requires-python` is `>=3.11`. The dev machine is an arm64 Mac with little free disk, so real SWE-bench runs need a separate x86 host (`docs/EVAL_SETUP.md`).
- **Default model** is `claude-opus-5-5`. Opus 5.5 rejects disabled thinking and sampling params, so `seed` labels runs and drives only our own randomness.
