# eval-harness — build spec

Status: v2, Nov 16 2026 (adds §9.7–9.9, §10.1, §11.1, §16). Owner: Tanmai Komatireddy.
Cursor: read `.cursor/rules/` and the section your task names. This file wins over guesses.

## 1. Goal

A new model (or prompt) drops at 9am. By lunch, one command says whether it is safe to
ship for revenue testing, and which step regressed, on which cases, with the trace.

Deliverables, in priority order:
1. `make demo` — offline, no keys: runs the suite across a model matrix, gates each
   candidate against production, writes `out/scorecard.html`.
2. A reviewer rejection turns into a new critical eval case that the next run catches.
3. A one-page scorecard a CTO could put in front of a board.

Out of scope: real client data, any claim of audit-standard compliance, a UI beyond the
static scorecard, integration with any real product, agent complexity beyond 5 nodes.

## 2. Repository layout

```
eval-harness/
  auditlab/
    config.py              # env + toml loading, REPO_ROOT, paths
    money.py               # integer cents / USD micros, formatting
    telemetry.py           # OTel JSONL exporter + optional LangSmith
    guard.py               # workpaper prose guard
    docs/loader.py         # HTML pages -> [Page], sha256, normalize
    domain/models.py       # Pydantic models, §4.2 and §5.2
    domain/enums.py        # ExceptionType, Node, Verdict
    models/base.py         # ModelClient protocol, ModelRequest/Response
    models/registry.py     # builds clients from config/models.toml
    models/cache.py        # content-addressed response cache
    models/stub.py         # simulated oracle + fault profiles
    models/replay.py       # serves cache only; error on miss
    models/openai_client.py, anthropic_client.py, gemini_client.py   # optional extras
    agent/state.py         # AgentState
    agent/graph.py         # LangGraph StateGraph wiring
    agent/nodes/{select_samples,extract_evidence,tie_out,flag_exceptions,draft_workpaper}.py
    agent/prompts/v1/*.md
    cli.py                 # `python -m auditlab.cli <cmd>`
  data/
    generate.py            # seeded engagement generator
    engagements/<case_id>/{population.json, docs/*.html, answer_key.json}
  evals/
    suites/core.json       # list of case ids + tags
    suites/from_feedback/*.json
    scorers/{select,extract,tieout,exceptions,workpaper_judge}.py
    scorers/judge_prompts/v1.md
    labels/workpaper_human_labels.json   # hand-labelled by the author (§9.6)
    agreement.py           # judge vs human agreement
    runner.py
    gate.py
    baselines/production.json
  feedback/
    rejections/*.json
    ingest.py
  report/
    scorecard.py
    templates/scorecard.html.j2
  config/models.toml, gate.toml
  runs/                    # gitignored except runs/demo/: replay cache, the recording runs'
                           # summary.json, and the demo-real summary + gate output
  scripts/check_names.sh
  tests/
  Makefile, pyproject.toml, README.md, AGENTS.md, SPEC.md, .env.example
  .github/workflows/ci.yml
```

No Postgres: run artifacts are files.

## 3. Stack and dependencies

Python 3.12, `uv`, `pydantic>=2.9`, `langgraph`, `jinja2`, `beautifulsoup4`,
`opentelemetry-api`, `opentelemetry-sdk`, `python-dotenv`, `tomli-w` (if writing toml).
Extras: `[openai]`, `[anthropic]`, `[gemini]`, `[langsmith]`, `[langfuse]`, `[otlp]`
(`opentelemetry-exporter-otlp`). Dev: `pytest`, `ruff`, `mypy`.
Commit `uv.lock`. Nothing else without asking.

## 4. Synthetic engagement

### 4.1 Scenario
Invented client: **Harbor Lane Supply Co.**, a distributor. Fiscal year ends 2025-12-31.
Audit area: revenue, existence and cutoff assertions. All names, addresses and documents
are invented. One engagement per case; each case is a seeded variant.

### 4.2 Population (`population.json`)
- ~120 sales invoices: `invoice_id, customer_id, invoice_date, ship_date, amount_cents, terms`.
- GL revenue entries: `je_id, invoice_id, posted_date, amount_cents, account`.
- Bank deposits: `deposit_id, date, amount_cents, payer, invoice_refs[]`.
- Shipping docs: `bol_id, invoice_id, ship_date, carrier`.
- Credit memo log: `cm_id, invoice_id, date, amount_cents`.
Supporting documents per sampled item are rendered as HTML (`docs/<doc_id>.html`) with
`<section class="page" data-page-no="N">`: invoice, bill of lading, bank statement page.

### 4.3 Sampling rule (given to the agent as text, and coded in the answer key)
Required sample = union of:
1. Key items: every invoice with `amount_cents >= 2_500_000` ($25,000).
2. Cutoff window: every invoice dated 2025-12-27 through 2026-01-05.
3. Every invoice with an entry in the credit memo log.

### 4.4 Planted exceptions
Each case plants 2–4, always inside the required sample:

| Type | Planted as | Critical |
|---|---|---|
| `DUPLICATE_INVOICE` | Two invoices, same customer, amount, and date; both posted to GL | yes |
| `CUTOFF_ERROR` | Invoice dated 2025-12-30, BOL ship date 2026-01-03, revenue in FY2025 | yes |
| `AMOUNT_MISMATCH` | GL amount differs from invoice by a planted delta | yes |
| `MISSING_EVIDENCE` | Sampled invoice with no BOL and no matching deposit | yes |
| `BILL_AND_HOLD` (judgment) | Customer-requested hold, goods segregated, title passed | no |
| `PARTIAL_SHIPMENT` (judgment) | BOL covers part of the invoice quantity | no |

Judgment items carry `allowed_outcomes` (for example `["exception", "no_exception_with_rationale"]`)
and a rubric of what the rationale must mention.

### 4.5 Answer key (`answer_key.json`)
`required_sample_ids[]`; `evidence[]` (item_id, field, value_cents or value_text, doc_id,
page_no, quote); `expected_tieouts[]` (item_id, invoice vs GL vs deposit, expected
match/delta_cents); `expected_exceptions[]` (item_id, type, critical); `judgment_items[]`
(item_id, allowed_outcomes, rubric). Generated, never hand-edited.

### 4.6 Suite
`evals/suites/core.json`: 20 cases = 4 seeds × 5 exception mixes. Each case has tags
(`cutoff`, `duplicate`, `judgment`, …) for filtering.

## 5. Agent

### 5.1 Graph
`select_samples` → `extract_evidence` → `tie_out` → `flag_exceptions` → `draft_workpaper`.
Linear LangGraph `StateGraph`. Compiled once; invoked per (case, run_index, model config).

### 5.2 State
```
AgentState:
  case_id: str; run_index: int; model_config_id: str
  population: Population                  # input
  sampling_rule: str                      # input, §4.3 text
  samples: SampleSelection | None         # select_samples: item_ids[], rationale
  evidence: EvidenceSet | None            # extract_evidence: fields[] with doc_id/page/quote, dropped[]
  tieouts: TieOutSet | None               # tie_out: per item, invoice/GL/deposit, delta_cents
  exceptions: ExceptionSet | None         # flag_exceptions: item_id, type, rationale
  workpaper: Workpaper | None             # draft_workpaper: prose with [[ev:id]], guard_violations[]
  step_records: list[StepRecord]          # one per node: latency_ms, tokens, cost_micros, trace_ref
```

### 5.3 Node contracts
- `select_samples` (model): gets population summary + sampling rule; returns item ids + one-line rationale per rule.
- `extract_evidence` (model): for each sampled item, reads its HTML docs; quote-or-drop.
- `tie_out` (code): compares invoice vs GL vs deposit vs BOL; integer cents only.
- `flag_exceptions` (model): gets tie-outs + evidence; returns typed exceptions with rationale; judgment items must pick one of the allowed outcomes.
- `draft_workpaper` (model): procedures performed, samples, results, exceptions, conclusion; amounts only as `[[ev:id]]` markers; guard runs after.

## 6. Model-swap layer

### 6.1 Interface
`ModelClient.complete(ModelRequest) -> ModelResponse`. Request: node, prompt_version,
system, user payload (JSON), output schema, run_index, seed. Response: parsed output,
tokens_in, tokens_out, latency_ms, provider_model_id, cached: bool.

### 6.2 Config (`config/models.toml`)
```toml
[production]
config_id = "prod"            # the baseline the gate compares against

[[model]]
config_id = "prod"
provider = "stub"            # or openai / anthropic / gemini / replay
model_id = "stub-prod"
prompt_version = "v1"
simulated = true

[[price]]                     # required for every real provider model_id
model_id = "<provider model id>"
input_usd_per_mtok = 0.0
output_usd_per_mtok = 0.0
source_url = "<pricing page>"
as_of = "YYYY-MM-DD"
```
Real model ids and prices are filled in by the human from the provider's pricing page.

### 6.3 Stub fault profiles (all `simulated = true`)
| config_id | Behaviour |
|---|---|
| `stub-prod` | Oracle; tiny seeded noise in workpaper wording only. |
| `stub-cand-good` | Oracle; cheaper simulated token counts. Should SHIP. |

Each stub has a simulated `[[price]]` (`source_url = "simulated"`; prod, cutoff and noisy
priced above cand-good) so cost per workpaper differs across the matrix.
| `stub-cand-cutoff` | At `select_samples`, ignores rule 2: drops every id that only the cutoff window selects (in the window, not a key item, no credit memo), so `required_recall` falls on every case, not only cutoff mixes. Downstream nodes are correct on what they get; planted `CUTOFF_ERROR` is missed. Should NOT SHIP, and the gate must point at `select_samples`. |
| `stub-cand-noisy` | Correct on deterministic items; on judgment items picks outcomes with seeded randomness per run_index. Every judgment item allows both outcomes, so this shows up in outcome agreement (below `min_outcome_agreement`), not in pass^N. |

### 6.4 Replay
`replay` serves only from `runs/demo/cache/`; a cache miss raises. A call that failed while
recording (timeout, prompt too large, invalid output) has no response to serve, so
`python -m evals.backfill_failures --run <id>` writes a failure entry under that call's exact
cache key, rebuilt from the run's cached upstream responses, with the recorded
`failure_origin` and message. Replay re-raises it, so the replayed run scores the failure
exactly as recorded (an infra timeout stays infra). If a key cannot be rebuilt exactly,
that run is re-recorded instead. A live cached client never serves a failure entry: it
calls the model again and replaces the entry on success. `make demo` uses
replay for any real-model config present in the committed cache, stubs otherwise.

## 7. Tracing and LLM observability
OpenTelemetry is the source of truth. OTel span per node: `gen_ai.system`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
`gen_ai.response.model`, latency, cost micros, `audit.case_id`, `audit.run_index`,
`audit.node`, `audit.model_config_id`, `audit.prompt_version`. Root span per agent run;
a judge call is a child span of the node it grades.

Sinks, each independent and each off unless its env vars are set (§8):
| Sink | Default | How |
|---|---|---|
| Local JSONL | on | `runs/<run_id>/spans.jsonl`; powers the offline demo and scorecard links |
| LangSmith | off | LangGraph's native tracing when `LANGSMITH_TRACING=true` |
| Langfuse | off | Langfuse SDK / LangGraph callback handler when `LANGFUSE_*` keys are set |
| Any OTLP backend | off | OTLP exporter when `OTEL_EXPORTER_OTLP_ENDPOINT` is set (Datadog, Honeycomb, a self-hosted collector) |

Scores travel with traces. When LangSmith or Langfuse is on, the runner attaches every
`StepScore` to its trace (LangSmith feedback / Langfuse score, keyed by metric name),
so a reviewer can filter traces by failing metric in either tool. Cursor: follow the
installed SDK's documentation for these calls; do not guess API names.

`trace_ref` = the URL in the first enabled remote sink (LangSmith, then Langfuse), else
`local:<run_id>/<span_id>`. The record keeps every enabled sink's ref, so the scorecard
can link to each. Tests never enable a remote sink.

## 8. Environment
`.env.example`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`LANGSMITH_TRACING=false`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=eval-harness`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`,
`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`,
`AUDITLAB_JUDGE_CONFIG=stub-judge`. All empty by default. Nothing required for `make demo`.

## 9. Scorers

| Node | Metric | Definition | Kind | Critical |
|---|---|---|---|---|
| select_samples | `required_recall` | share of required ids selected | process | yes |
| select_samples | `precision` | share of selected ids that are required | process | no |
| extract_evidence | `citation_validity` | share of fields whose quote is an exact substring | process | yes |
| extract_evidence | `field_exact` | share of answer-key evidence matched exactly | process | yes |
| tie_out | `tieout_accuracy` | share of expected tie-outs with correct match/delta | process | yes |
| flag_exceptions | `planted_recall` | share of planted critical exceptions found with correct type | outcome | yes (must be 1.0) |
| flag_exceptions | `false_positive_rate` | flagged exceptions not in key / flagged | outcome | no |
| flag_exceptions | `judgment_allowed` | judgment items whose outcome is in allowed_outcomes and whose rationale meets rubric | outcome | no |
| draft_workpaper | `guard_pass` | no guard violations | outcome | yes |
| draft_workpaper | `judge_<criterion>` | LLM judge per rubric criterion (§9.5) | outcome | no |

`kind` separates "did the work come out right" (outcome) from "was the procedure
followed" (process), as two scores that never mix (§16, Universal Verifier).

9.5 Workpaper judge rubric criteria: `procedures_described`, `samples_identified`,
`exceptions_explained`, `conclusion_supported`, `no_unsupported_claims`.
Offline, `stub-judge` returns a deterministic rubric check (labelled simulated).

9.6 Judge agreement: `evals/labels/workpaper_human_labels.json` holds ~15 workpapers the
author labels by hand per criterion. `evals/agreement.py` reports raw agreement and
Cohen's kappa per criterion; criteria with kappa < 0.6 are flagged "judge not trusted".
One judge, no panel (§16, correlated judge errors). If a second judge is ever tried,
report the correlation of the two judges' errors against the human labels before
combining them.

9.7 Judge confidence and escalation (§16, accept-when-confident pattern). The judge
returns `confidence` (0–1) per criterion. Below `judge.escalate_below` in
`config/gate.toml`: re-grade with `judge.strong_config` if one is set, else write the
item to `feedback/queue/<id>.json` for a reviewer. The threshold is tuned on the §9.6
human labels, never guessed. The runner reports, per judge setup: agreement with human
labels, share escalated, and judge cost per workpaper. A reviewer's decision on a queued
item goes through the §12 ingest and becomes an eval case.

9.8 Failure origin. Every failed `StepScore` carries `failure_origin`:
`agent` (the model's output was wrong, including a response that is not valid JSON or
fails the node's output schema), `infra` (provider HTTP error, timeout, rate limit,
connection failure), or `data` (the case itself is broken, for example a missing doc). Only `agent` failures count toward accuracy. `infra` and
`data` are reported separately and never make a model look worse.

9.9 Final-output-only shadow grade (§16, multi-channel evidence). For every run, also
compute the grade a reviewer would give from the deliverable alone: guard pass plus all
`judge_*` criteria pass, without the answer key. `missed_by_final_only` = runs that fail
any critical step metric but pass the shadow grade. This number is the case for
step-level scoring, measured on this suite.

Each `StepScore`: case_id, run_index, model_config_id, node, metric, kind, value, passed,
critical, failure_origin, detail, trace_ref.

## 10. Runner
`python -m evals.runner --suite core --models prod,stub-cand-good,stub-cand-cutoff,stub-cand-noisy --n 5`
Per model config: accuracy per metric (mean across case × run), pass^N per case,
outcome agreement, p50/p95 latency per workpaper, mean cost per workpaper (micros).
Writes `runs/<run_id>/{manifest.json, results.jsonl, summary.json, spans.jsonl}` and
appends to `runs/history.jsonl`. Cache by §6 key; `--no-cache` to force fresh calls.
Use N=3 or more for every config; a single run never counts as a pass (pass^N).

10.1 Saturation check (§16). A metric is `saturated` when every model config in the
run scores ≥ 0.98 on it. The summary lists saturated metrics; the scorecard greys them
out with a note that the suite no longer separates models on that metric. Feedback-
derived cases (§12) are the planned way to add headroom.

## 11. Gate (`config/gate.toml`)
```toml
default_margin = 0.02
[critical]
"select_samples.required_recall" = 0.0     # margin per metric; 0.0 = no drop allowed
"extract_evidence.citation_validity" = 0.0
"extract_evidence.field_exact" = 0.02
"tie_out.tieout_accuracy" = 0.0
"flag_exceptions.planted_recall" = 0.0
"draft_workpaper.guard_pass" = 0.0
[consistency]
min_pass_pow_n = 0.9
min_outcome_agreement = 0.9
[stats]
bootstrap_resamples = 2000
bootstrap_seed = 7
confidence = 0.95
max_infra_failure_rate = 0.05
[judge]
escalate_below = 0.7        # placeholder; tune on human labels (§9.7)
strong_config = ""          # optional stronger judge config_id
```
Verdicts: `SHIP`, `DO_NOT_SHIP`, `NEEDS_REVIEW` (non-critical drops, consistency below
threshold, a critical drop the statistics can't confirm, or infra failure rate above
`max_infra_failure_rate`).

11.1 Paired comparison (§16, paired with/without design). Candidate and baseline run
the same cases, so the gate compares per-case differences, not two averages. For each
critical metric: per-case delta = candidate − baseline (mean over that case's runs);
seeded bootstrap CI on the mean delta.
- `DO_NOT_SHIP` if the mean delta is below −margin **and** the CI upper bound is below 0.
- `NEEDS_REVIEW` if the mean delta is below −margin but the CI includes 0 (print
  "drop not confirmed at this suite size: n cases").
- Any planted critical exception missed in any agent-origin run is `DO_NOT_SHIP`
  regardless of statistics. Output example:
```
stub-cand-cutoff  DO_NOT_SHIP
  flag_exceptions.planted_recall  1.00 -> 0.80  cases: s2-mixC, s3-mixA ...
  root step: select_samples.required_recall  1.00 -> 0.91  (cutoff-window items not selected)
  traces: local:<run_id>/<span_id> ...
```
Root-step attribution: when a downstream critical metric fails, walk upstream and report
the earliest node whose critical metric also dropped on the same cases.
The baseline defaults to the `[production]` config in the same run; `--production <config_id>`
on the runner picks another config of that run (`make demo-real` uses the recorded
`gpt-6.2-sol`, so no real model is gated against a simulated stub). `make baseline RUN=<id>`
writes that run's production config as per-case metric means (plus its failing trace refs
and consistency numbers) to `evals/baselines/production.json`, for gating a later run with
`make gate RUN=<id> BASELINE=evals/baselines/production.json`; the gate never edits it.
Non-critical drops beyond `default_margin` move the verdict to `NEEDS_REVIEW` only when the
CI confirms them; unconfirmed ones are listed as notes. Consistency: pass^N share below
`min_pass_pow_n` or outcome agreement below `min_outcome_agreement` is `NEEDS_REVIEW`.

## 12. Feedback loop
`python -m feedback.ingest --trace <trace_ref> --node flag_exceptions --reviewer "R. Ortiz (synthetic)" --reason "..." --expected '{"item_id": "...", "type": "CUTOFF_ERROR"}'`
1. Writes `feedback/rejections/<id>.json`.
2. Builds a case in `evals/suites/from_feedback/<id>.json`: source case_id, the item,
   corrected expectation, provenance, `critical = true`, tag `from_feedback`.
3. Next runner call includes `from_feedback` cases by default (`--no-feedback` skips them);
   scorecard counts them. A feedback case reruns its source engagement and is scored on one
   critical metric, `flag_exceptions.feedback_expectation_met`. Reviewer-confirmed cases are
   ground truth: a miss in any agent-origin run is `DO_NOT_SHIP` ("reviewer-confirmed
   expectation missed in k of N runs"), like a missed planted exception.
Validation: the trace must be a score's trace_ref in some run, for that node; `--expected` must
be present and schema-valid; the item must exist in the source engagement. Only
`flag_exceptions` rejections are supported so far. `feedback/rejections/` and
`evals/suites/from_feedback/` are gitignored (local), so a fresh clone starts with none.
Demo (`make demo-feedback`, after `make demo`): a synthetic reviewer rejects a
`stub-cand-noisy` run that decided a bill-and-hold item as `no_exception_with_rationale`,
corrected outcome `exception`; re-run; noisy goes NEEDS_REVIEW → DO_NOT_SHIP, prod passes.

## 13. Scorecard (`out/scorecard.html`)
Top: one verdict row per candidate vs production (SHIP / DO_NOT_SHIP / NEEDS_REVIEW).
Then: step × model heatmap (critical metrics), split into process and outcome; a
"missed by final-only grading" count per model (§9.9); cost and latency per workpaper;
consistency (pass^N); infra/data failure counts kept apart from accuracy; saturated
metrics greyed out (§10.1); trend line per model across `runs/history.jsonl`; judge
agreement and escalation table (§9.6–9.7); count of feedback-derived cases. SIMULATED badge on stub runs. Footer: run id,
git sha, suite hash, generated_at.

## 14. Make targets
`install`, `data`, `demo` (data → runner → gate → scorecard, offline), `eval MODELS=… N=…`,
`gate RUN=…`, `reject …`, `scorecard RUN=…`, `baseline RUN=…`, `names`, `lint`,
`typecheck`, `test`, `check`.

## 15. README outline (third person, no invented metrics)
1. The problem (one paragraph, the 9am → lunch framing).
2. Quickstart: `make install && make demo`, open `out/scorecard.html`.
3. What the demo shows: model swap, regression caught at its source step, rejection →
   eval case, scorecard. Real-model numbers only if committed under `runs/demo/`.
4. Architecture diagram + the five nodes + scorer table.
5. Design choices: code for arithmetic, models for judgment; quote-or-drop; pass^N;
   step-level over final-only grading; outcome vs process; paired CI in the gate; one
   calibrated judge with escalation, not a panel. Link each choice to its paper (§16)
   in one line each; the author reads each abstract before citing it.
6. Limits and next steps.
7. Not affiliated with any company named in the problem statement. Synthetic data only.

## 16. Research this design draws on

Summaries come from Elvis Saravia's "My Top 10 Eval Papers of 2026" newsletter. Read
each abstract before citing it in the README or video.

| Paper | Used in | How |
|---|---|---|
| [Claw-Eval](https://arxiv.org/abs/2604.06132) | §9.9, §10 | Final-output-only grading misses failures; repeated runs |
| [The Art of Building Verifiers for Computer Use Agents](https://arxiv.org/abs/2604.06240) | §9 `kind`, §9.8 | Outcome vs process scores; agent vs external faults |
| [SkillsBench](https://arxiv.org/abs/2602.12670) | §11.1 | Paired comparisons on the same tasks |
| [When AI Benchmarks Plateau](https://arxiv.org/abs/2602.16763) | §10.1, §12 | Saturation check; expert-curated cases keep headroom |
| [JEV-as-a-Judge](https://arxiv.org/abs/2609.26550) | §9.7 | Accept when confident, escalate when unsure (pattern only) |
| [Nine Judges, Two Effective Votes](https://arxiv.org/abs/2605.29800) | §9.6 | One calibrated judge, no panel |
| [Agent-as-a-Judge](https://arxiv.org/abs/2601.05111) | §15 next steps | Tool-using judge as a later step |
| [Terminal-Bench](https://arxiv.org/abs/2601.11868) | §4.5 | Task + reference solution + state checks, like our answer keys |
| [PostTrainBench](https://arxiv.org/abs/2603.08640) | rules | Cheating checks, like the answer-key leak test |
