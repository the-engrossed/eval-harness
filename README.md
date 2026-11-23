# eval-harness

A release gate for an audit revenue-testing agent, on synthetic data. It runs a suite of
engagements across a matrix of models, scores every step of the agent, and says whether a
candidate model is safe to ship against production, which step regressed, and on which cases.

## The problem

A new model or prompt drops at 9am. By lunch, the team needs to know whether it can replace
the one in production for revenue testing. Final-output grading cannot answer that: a
workpaper can read well while the sample behind it skipped half the cutoff window. This
harness grades each step against a generated answer key, repeats every case, compares
candidate and production case by case, and turns a reviewer's rejection into a new critical
eval case that the next run must pass.

## Quickstart

Python 3.12 and `uv`. No keys, no network, no Docker.

```
make install
make demo           # 4 simulated models x 20 cases x N=5 -> gate -> out/scorecard.html
make demo-feedback  # after make demo: a synthetic reviewer rejection becomes a critical case
make demo-real      # replay the recorded real models (gpt-6.2-sol vs qwen3:8b) from runs/demo/cache
```

`make demo-real` serves every model and judge response from the committed cache; a cache
miss raises, and the API key is blanked for the run. `make help` lists the other targets.

## What the simulated demo shows

All numbers in this section are **SIMULATED**: they come from seeded stub models and a
deterministic stub judge (`make demo`, suite `core`, 20 cases, N=5), and the scorecard
badges them as such.

| Candidate | Verdict | Why |
|---|---|---|
| `stub-cand-good` | SHIP | Same scores as production; simulated cost per workpaper 36,049 vs 180,292 USD micros |
| `stub-cand-cutoff` | DO_NOT_SHIP | Root step `select_samples.required_recall` 1.00 → 0.56 (95% CI of the paired delta [−0.466, −0.407], 20 cases) |
| `stub-cand-noisy` | NEEDS_REVIEW | Outcome agreement across runs 0.76, below the 0.90 threshold |

- **Regression caught at its source step.** The cutoff candidate drops cutoff-window
  invoices at sampling. Downstream, `field_exact` falls to 0.55, `tieout_accuracy` to 0.56
  and `planted_recall` to 0.73 (a planted exception missed in 60 runs); the gate attributes
  all four drops to `select_samples`.
- **Final-only grading misses it.** All 100 cutoff-candidate runs fail a critical step yet
  pass the deliverable-only shadow grade (`missed_by_final_only` = 100).
- **Rejection → eval case.** `make demo-feedback` rejects a noisy run's bill-and-hold call
  on `INV-1110` (case `s1-mixB`). The rerun includes the new critical case: the noisy
  candidate meets the reviewer's expectation in 2 of 5 runs (0.40) and moves to
  DO_NOT_SHIP; production meets it in 5 of 5.
- **Saturation.** Ten metrics score ≥ 0.98 for every stub, so the scorecard greys them out:
  this suite no longer separates the stubs on them.

## Real models

The recorded comparison: **gpt-6.2-sol as production, a self-hosted qwen3:8b as the
candidate**, graded by a real LLM judge (gpt-6-luna). Suite `core-real`: 5 cases
(`s1-mixA`–`E`) plus one reviewer-derived case, N=3, 18 runs per model. Every number
below is from `runs/demo/` (the recording runs' `summary.json`, and the `demo-real`
replay's `summary.json` and `gate.txt`). The simulated production stub is not part of
this gate.

**Verdict: qwen3:8b DO_NOT_SHIP** against gpt-6.2-sol.

| Metric | gpt-6.2-sol | qwen3:8b |
|---|---|---|
| `select_samples.required_recall` (critical) | 1.00 | 0.82 (s1-mixB, D, E; CI [−0.348, −0.032]) |
| `select_samples.precision` | 1.00 | 0.25 |
| `extract_evidence.citation_validity`, `field_exact` (critical) | 1.00 | 0.00 (s1-mixA, s1-mixE) |
| Planted critical exception missed | 0 runs | 6 agent-origin runs |
| pass^N share | 1.00 | 0.00 |
| Infra failure rate | 0.00 | 0.67 |

- **Root step: `select_samples`.** qwen3:8b over-selects (precision 0.25) and still misses
  required items (recall 0.82). The larger sample is what breaks the next step.
- **Prompt too large.** For s1-mixA and s1-mixE, the `extract_evidence` prompt is an
  estimated 42,503 and 34,841 tokens, over the 32,768-token window. The harness checks this
  before calling the model and records an agent failure (6 runs) instead of sending a
  prompt the server would silently truncate.
- **Timeouts.** On s1-mixB, C, D and the reviewer case, `extract_evidence` timed out at
  300 s in all 12 runs. Timeouts are infra failures: excluded from accuracy, counted in the
  infra rate (0.67, above the 0.05 limit). qwen3:8b produced no workpaper, so tie-out,
  guard, judge and feedback metrics are unscored for it.
- **gpt-6.2-sol per workpaper:** $0.205 model cost plus $0.0019 for the judge; latency p50
  150 s, p95 194 s. Prices are in `config/models.toml` with source URL and date. Its
  non-critical shortfalls: `false_positive_rate` 0.15 and `judge_conclusion_supported` 0.73.
- **qwen3:8b's repeats are identical.** Each request sends a fixed seed (7) and the local
  server is deterministic, so its three runs per case match exactly. Its outcome agreement
  of 1.00 says nothing about run-to-run variance.

## Architecture

```
population.json + docs/*.html          answer_key.json (scorers only)
            |                                     |
 select_samples --> extract_evidence --> tie_out --> flag_exceptions --> draft_workpaper
   (model)            (model)           (code)        (model)              (model + guard)
            |                                     |
   ModelClient: stub | chat-completions client | replay  (content-addressed response cache)
            |                                     |
   step scores + traces (JSONL spans) --> runner (N runs per case) --> gate --> scorecard
                                                    ^
                          reviewer rejection --> feedback case (critical)
```

| Node | Metric | Kind | Critical |
|---|---|---|---|
| select_samples | `required_recall`, `precision` | process | recall yes |
| extract_evidence | `citation_validity`, `field_exact` | process | yes |
| tie_out | `tieout_accuracy` | process | yes |
| flag_exceptions | `planted_recall` (must be 1.0), `false_positive_rate`, `judgment_allowed` | outcome | planted_recall yes |
| flag_exceptions | `feedback_expectation_met` (reviewer cases) | outcome | yes |
| draft_workpaper | `guard_pass`; `judge_<criterion>` × 5 | outcome | guard yes |

Every failed score carries a failure origin: `agent` (wrong or invalid output, prompt too
large), `infra` (HTTP error, timeout) or `data` (broken case). Only `agent` failures count
toward accuracy. Full contract: `SPEC.md`.

## Design decisions

- **Code for arithmetic, models for judgment.** `tie_out` is deterministic integer-cent
  Python; models select, extract, classify and draft.
- **Quote-or-drop.** Every extracted field cites a doc, page and exact quote; a quote that
  is not a substring of the page fails `citation_validity`.
- **No numbers in prose.** The workpaper references amounts only as `[[ev:id]]` markers;
  a deterministic guard rejects any stray digit.
- **pass^N, not one run.** Each case runs N times; a case passes only if every run passes.
- **Step-level over final-only grading.** The shadow grade measures what final-only
  grading would miss on this suite (100 of 100 simulated cutoff runs).
- **Outcome vs process** are separate scores that never mix.
- **Paired CI in the gate.** Candidate and production run the same cases; the gate
  bootstraps the per-case delta and blocks only on a confirmed drop. Unconfirmed drops
  are NEEDS_REVIEW. A missed planted exception blocks regardless.
- **Infra never makes a model look worse.** Timeouts and judge errors are reported as an
  infra rate with its own threshold, apart from accuracy.
- **Measured context check.** The pre-call size estimate uses a bytes-per-token ratio
  measured per model on the same request (3.10 and 2.33), not a guess.
- **Replay is exact.** Responses are cached by provider, model, prompt version, node,
  input, run index and request variant; a call that failed while recording is replayed as
  the same failure, so the offline demo reproduces the recorded scores.
- **One judge.** A single judge per run; a panel's errors tend to be correlated.

## Deliberately not built

- Human labels for the judge and its kappa (SPEC §9.6), and confidence-based escalation
  (§9.7). The judge returns a confidence per criterion; nothing acts on it yet.
- A tool-using judge.
- Clients beyond one chat-completions-style client; no per-vendor SDKs.
- Attaching step scores to remote traces; spans are written locally, remote sinks are optional.
- Rejections for nodes other than `flag_exceptions`.
- A UI beyond the static scorecard, real client data, or any claim of audit-standard
  compliance.

Limits: the real comparison is 5 cases on one seed and one hardware setup; qwen3:8b's
repeats are deterministic; recording does not yet write failure entries itself
(`python -m evals.backfill_failures` does); the reviewer-derived case is local, so a fresh
clone gets it only after `make demo-feedback`. More notes: `docs/what-didnt-work.md`.

## Research this draws on

| Paper | Used for |
|---|---|
| [Claw-Eval](https://arxiv.org/abs/2604.06132) | Final-output-only grading misses failures; repeated runs |
| [The Art of Building Verifiers for Computer Use Agents](https://arxiv.org/abs/2604.06240) | Outcome vs process scores; agent vs external faults |
| [SkillsBench](https://arxiv.org/abs/2602.12670) | Paired comparisons on the same tasks |
| [When AI Benchmarks Plateau](https://arxiv.org/abs/2602.16763) | Saturation check; expert-curated cases keep headroom |
| [JEV-as-a-Judge](https://arxiv.org/abs/2609.26550) | Accept when confident, escalate when unsure (planned) |
| [Nine Judges, Two Effective Votes](https://arxiv.org/abs/2605.29800) | One calibrated judge, no panel |
| [Agent-as-a-Judge](https://arxiv.org/abs/2601.05111) | Tool-using judge as a later step |
| [Terminal-Bench](https://arxiv.org/abs/2601.11868) | Task + reference solution + state checks, like the answer keys |
| [PostTrainBench](https://arxiv.org/abs/2603.08640) | Cheating checks, like the answer-key access tests |

Not affiliated with any company named in the problem statement. Synthetic data only: the
client, documents and reviewer are invented.
