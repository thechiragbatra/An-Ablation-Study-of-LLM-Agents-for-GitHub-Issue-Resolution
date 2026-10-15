# Developer and reproduction entry points. Secrets come from the environment only
# (ANTHROPIC_API_KEY, GITHUB_TOKEN); nothing here stores or prints them.

UV       ?= uv
RUN      := $(UV) run
WORKERS  ?= 2
SEEDS    ?= 0 1 2
DEV_SEEDS ?= 0
VARIANTS := a0 a1 a2 a3
OUT      ?= out/experiments
ANALYSIS ?= analysis

.PHONY: help lint format typecheck test check eval-dev eval-full analyze docs dashboard demo-traces

help:
	@echo "lint typecheck test check | eval-dev eval-full analyze docs | dashboard demo-traces"

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

typecheck:
	$(RUN) mypy .

test:
	$(RUN) pytest

check: lint typecheck test

# Tuning / cost measurement on the 15-instance DEV set only (no preregistration needed).
eval-dev:
	@for v in $(VARIANTS); do \
	  $(RUN) python -m eval.run_experiment --config configs/eval/$$v.yaml --set dev \
	    --seeds $(DEV_SEEDS) --workers $(WORKERS) --out $(OUT)/$$v-dev || exit $$?; \
	done
	@for v in $(VARIANTS); do \
	  $(RUN) python -m eval.score --experiment $(OUT)/$$v-dev || exit $$?; \
	done
	$(RUN) python -m eval.analyze $(foreach v,$(VARIANTS),$(OUT)/$(v)-dev) --out out/analysis-dev

# FROZEN subset: refused by the harness until docs/PREREGISTRATION.md is filled in and
# committed with a clean tree. Variant order is fixed (A0, A1, A2, A3).
eval-full:
	@for v in $(VARIANTS); do \
	  $(RUN) python -m eval.run_experiment --config configs/eval/$$v.yaml --set frozen \
	    --seeds $(SEEDS) --workers $(WORKERS) --out $(OUT)/$$v-frozen || exit $$?; \
	done
	$(RUN) python -m eval.score --experiment $(OUT)/a0-frozen
	$(RUN) python -m eval.score --experiment $(OUT)/a1-frozen
	$(RUN) python -m eval.score --experiment $(OUT)/a2-frozen --score-candidates
	$(RUN) python -m eval.score --experiment $(OUT)/a3-frozen --score-candidates

analyze:
	$(RUN) python -m eval.analyze $(foreach v,$(VARIANTS),$(OUT)/$(v)-frozen) --out $(ANALYSIS)
	$(RUN) python -m eval.failure_taxonomy export $(foreach v,$(VARIANTS),$(OUT)/$(v)-frozen) \
	  --n 30 --seed 0 --out $(ANALYSIS)/failures_to_label.csv

# Fill report/résumé placeholders from real analysis outputs (copies in docs/rendered/).
docs:
	$(RUN) python -m eval.render_docs --analysis $(ANALYSIS) \
	  --experiments $(foreach v,$(VARIANTS),$(OUT)/$(v)-frozen) \
	  $(if $(wildcard $(ANALYSIS)/failures_labeled.csv),--taxonomy $(ANALYSIS)/failures_labeled.csv) \
	  $(if $(wildcard out/pilot/pr_outcomes.csv),--pilot-outcomes out/pilot/pr_outcomes.csv)

dashboard:
	$(RUN) python -m agent.dashboard --trace-dir $(if $(wildcard out/traces),out/traces,out/demo_traces)

demo-traces:
	$(RUN) python scripts/demo_traces.py
