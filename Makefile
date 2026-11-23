.DEFAULT_GOAL := help
BIN := .venv/bin

.PHONY: help install data demo demo-real demo-feedback eval eval-real record-real gate reject scorecard baseline names lint typecheck test check

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install: ## create .venv on python 3.12 from uv.lock with dev tools
	uv sync --python 3.12 --extra dev --extra openai --extra langsmith --extra langfuse --extra otlp

data: ## rebuild data/engagements/ (20 cases) and evals/suites/core.json
	$(BIN)/python -m data.build_all

DEMO_MODELS := prod,stub-cand-good,stub-cand-cutoff,stub-cand-noisy
DEMO_RUN := demo-$(shell date -u +%Y%m%dT%H%M%SZ)
# Pinned on the command line so a filled-in .env cannot turn on a real judge or a remote
# sink (load_dotenv never overrides a variable that is already set, even to "").
OFFLINE_SINKS := LANGSMITH_TRACING=false \
	LANGFUSE_PUBLIC_KEY= LANGFUSE_SECRET_KEY= LANGFUSE_HOST= OTEL_EXPORTER_OTLP_ENDPOINT=
OFFLINE_ENV := AUDITLAB_JUDGE_CONFIG=stub-judge $(OFFLINE_SINKS)

demo: ## offline, no keys: data -> runner (4 stub configs, N=5) -> gate (report only) -> scorecard
	$(MAKE) data
	$(OFFLINE_ENV) $(BIN)/python -m evals.runner --suite core --models $(DEMO_MODELS) \
		--n 5 --workers $(WORKERS) --run-id $(DEMO_RUN)
	$(OFFLINE_ENV) $(BIN)/python -m evals.gate_cli gate --run $(DEMO_RUN) --report-only
	$(BIN)/python -m report.scorecard --run $(DEMO_RUN)

# Real models only: gpt-6.2-sol is the production baseline, qwen3:8b the candidate. The
# simulated prod stub is left out, so no real model is gated against a stub.
DEMO_REAL_MODELS := replay-openai-gpt-6.2-sol,replay-local-qwen3-8b
DEMO_REAL_PRODUCTION := replay-openai-gpt-6.2-sol
DEMO_REAL_RUN := demo-real-$(shell date -u +%Y%m%dT%H%M%SZ)
# Replay only: models and judge serve from the committed runs/demo/cache; a miss raises.
# OPENAI_API_KEY is blanked so no real client could be built even by mistake.
REPLAY_ENV := AUDITLAB_JUDGE_CONFIG=replay-openai-gpt-6-luna-judge OPENAI_API_KEY= $(OFFLINE_SINKS)

demo-real: ## offline, no keys: replay the recorded real models (core-real, N=3) -> gate -> scorecard
	$(MAKE) data
	$(REPLAY_ENV) $(BIN)/python -m evals.runner --suite core-real --models $(DEMO_REAL_MODELS) \
		--production $(DEMO_REAL_PRODUCTION) --n 3 --workers $(WORKERS) --run-id $(DEMO_REAL_RUN)
	$(REPLAY_ENV) $(BIN)/python -m evals.gate_cli gate --run $(DEMO_REAL_RUN) --report-only
	$(BIN)/python -m report.scorecard --run $(DEMO_REAL_RUN)

FEEDBACK_RUN := feedback-$(shell date -u +%Y%m%dT%H%M%SZ)

demo-feedback: ## after make demo: synthetic reviewer rejects a noisy bill-and-hold call -> rerun -> gate -> scorecard
	$(BIN)/python -m feedback.demo
	$(OFFLINE_ENV) $(BIN)/python -m evals.runner --suite core --models $(DEMO_MODELS) \
		--n 5 --workers $(WORKERS) --run-id $(FEEDBACK_RUN)
	$(OFFLINE_ENV) $(BIN)/python -m evals.gate_cli gate --run $(FEEDBACK_RUN) --report-only
	$(BIN)/python -m report.scorecard --run $(FEEDBACK_RUN)

SUITE ?= core
MODELS ?= prod
N ?= 3
WORKERS ?= 4

eval: ## run the suite: make eval MODELS=a,b N=3 SUITE=core WORKERS=4
	$(BIN)/python -m evals.runner --suite $(SUITE) --models $(MODELS) --n $(N) --workers $(WORKERS)

REAL_JUDGE := openai-gpt-6-luna-judge

eval-real: ## real models with the real judge: make eval-real MODELS=a,b N=3 SUITE=core (needs OPENAI_API_KEY)
	AUDITLAB_JUDGE_CONFIG=$(REAL_JUDGE) $(BIN)/python -m evals.runner --suite $(SUITE) \
		--models $(MODELS) --n $(N) --workers $(WORKERS)

REAL_ENV := AUDITLAB_JUDGE_CONFIG=$(REAL_JUDGE) LANGSMITH_TRACING=false \
	LANGFUSE_PUBLIC_KEY= LANGFUSE_SECRET_KEY= LANGFUSE_HOST= OTEL_EXPORTER_OTLP_ENDPOINT=

record-real: ## record real responses (+ judge) into runs/demo/cache: make record-real MODELS=prod,<real> WORKERS=4
	$(REAL_ENV) $(BIN)/python -m evals.runner --suite core-real --models $(MODELS) --n 3 \
		--workers $(WORKERS) --cache-dir runs/demo/cache

gate: ## gate a run against production: make gate RUN=<run_id> [BASELINE=evals/baselines/production.json]
	$(BIN)/python -m evals.gate_cli gate --run $(RUN) $(if $(BASELINE),--baseline $(BASELINE))

reject: ## make reject TRACE=<ref> NODE=flag_exceptions REVIEWER="<name>" REASON="<text>" EXPECTED='<json>'
	$(BIN)/python -m feedback.ingest --trace "$(TRACE)" --node "$(NODE)" --reviewer "$(REVIEWER)" \
		--reason "$(REASON)" $(if $(EXPECTED),--expected '$(EXPECTED)')

scorecard: ## render out/scorecard.html: make scorecard RUN=<run_id>
	$(BIN)/python -m report.scorecard --run $(RUN)

baseline: ## write evals/baselines/production.json from a run: make baseline RUN=<run_id>
	$(BIN)/python -m evals.gate_cli baseline --run $(RUN)

names: ## fail if any .names-denylist pattern appears in tracked files (local only)
	bash scripts/check_names.sh

lint: ## ruff
	$(BIN)/ruff check .

typecheck: ## mypy --strict
	$(BIN)/mypy auditlab data evals report feedback

test: ## pytest, no network
	$(BIN)/pytest

check: lint typecheck test ## everything ci runs except names
