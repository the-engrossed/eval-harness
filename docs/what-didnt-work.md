# What didn't work

Notes behind the README's limits. Real-model numbers come from the smoke runs (2026-11-18,
case `s1-mixA`, run_index 0) and the `core-real` recordings committed under `runs/demo/`
(2026-11-23). Local models ran on one laptop (48 GB unified memory) behind a local
chat-completions server with a 32,768-token context and thinking off
(`reasoning_effort: "none"`) unless stated.

## Vision-capable local models reject structured output

`qwen3.6:27b` and `gemma4:31b` answer every JSON-schema request with
`HTTP 501: structured output is unavailable`, on both of the server's chat endpoints. The
text-only `qwen3:8b` and `qwen3:32b` accept the same request. The harness needs
schema-constrained output, so these two models could not be used.

## qwen3:32b: correct but too slow for extract_evidence

- Fits: 28 GB loaded, fully on GPU, 32k context, no swap-outs.
- About 13 output tokens/s (qwen3:8b: about 49).
- `select_samples`: 53 s, 5,449 tokens in, 336 out; 15 items, required_recall 0.87
  (missed key items INV-1028 and INV-1034), precision 0.87.
- `extract_evidence` timed out at 300 s, and again at 1,200 s in a measurement-only rerun.
  A probe capped at 1,500 output tokens showed well-formed, correct evidence at about 750
  tokens per item, so 15 items need roughly 19 minutes. Not a loop, just slow.
- Dropped from the recording: about 23 minutes per run puts 5 cases × N=3 near 6 hours.

## qwen3:8b: thinking on is slow, thinking off over-selects

- Thinking on (the model default): 8 min 51 s per case; `select_samples` alone wrote 8,291
  output tokens.
- Thinking off: `select_samples` takes 26 s but, in the smoke run, picked 114 of 117
  invoices (precision 0.13). In the `core-real` recording its precision is 0.25 and
  required_recall 0.82.
- The over-selection is what breaks the next step: the `extract_evidence` prompt grows
  with the sample.

## The server truncates long prompts without an error

In the smoke run the 114-item `extract_evidence` prompt was 37,166 tokens. The server
dropped the middle of it, reported success, and the only trace was a log warning; the call
then timed out. A 28,654-token prompt with a marker at the very start came back intact, so
the window itself was fine.

Fix: a pre-call check. If the estimated prompt exceeds the configured window, the step
records an agent failure ("prompt too large: N tokens") and no request is sent. In
`core-real` it fired on s1-mixA (42,503 estimated tokens) and s1-mixE (34,841), 3 runs each.

## A bytes/4 token estimate undercounts

The first estimate (UTF-8 bytes / 4) undercounted the provider-reported prompt tokens by
1.29× for gpt-6.2-sol and 1.72× for qwen3:8b, so the pre-check would never have fired for
qwen. Each config now carries a ratio measured on the same `s1-mixA` `select_samples`
request: 12,681 bytes / 4,086 tokens = 3.10, and 12,681 / 5,449 = 2.33. The judge config
borrows gpt-6.2-sol's ratio; it has not been measured on its own.

## Timeouts remain for qwen3:8b

Where the prompt fits (s1-mixB, C, D and the reviewer case), `extract_evidence` still timed
out at 300 s in all 12 runs. These are infra failures, so they are excluded from accuracy
and show up as an infra rate of 0.67. The large selections are the likely cause, but a
timeout does not prove it, so the harness does not count them against the model.

## Repeats are not independent with a fixed seed

Every request sends seed 7, and the local server is deterministic, so qwen3:8b's three
runs per case are identical. N=3 adds no information about its run-to-run variance, and its
outcome agreement of 1.00 is trivially true.

## Recording gaps

- One connection error at `flag_exceptions` (s1-mixA, run 1) in the first gpt-6.2-sol
  recording. Re-running the recording served everything else from the cache and refetched
  only that call; the committed run has no infra failures.
- Failed calls have no response to cache, so replay first missed on them. Failure entries
  are now backfilled from the recording run (`python -m evals.backfill_failures`), and
  replay re-raises the recorded error; recording does not yet write them itself.
- The recording runs include the simulated production stub. Gating a real model against
  it is meaningless, so `make demo-real` picks gpt-6.2-sol as production (`--production`).
