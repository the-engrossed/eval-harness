"""`make demo-feedback`, step 1: a SYNTHETIC reviewer rejects a stub-cand-noisy bill-and-hold call.

Finds the latest runs/demo-* run, picks a stub-cand-noisy run whose bill-and-hold item came out
no_exception_with_rationale, and ingests the rejection with corrected outcome `exception`.
The reviewer is invented and labelled as such. Re-running is a no-op once the case exists.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from auditlab.config import ENGAGEMENTS_DIR, RUNS_DIR
from auditlab.domain.enums import ExceptionType, JudgmentOutcome, Node
from evals.feedback_cases import load_feedback_cases
from evals.results import RunResult
from feedback.ingest import CASES_DIR, REJECTIONS_DIR, RejectionError, ingest

CANDIDATE = "stub-cand-noisy"
REVIEWER = "R. Ortiz (synthetic reviewer)"
REASON = (
    "Bill-and-hold criteria not met: the customer's hold letter alone does not show the goods "
    "were segregated and title had passed, so this is an exception."
)
REJECTED = (
    f":{ExceptionType.BILL_AND_HOLD.value}:{JudgmentOutcome.NO_EXCEPTION_WITH_RATIONALE.value}"
)


def latest_demo_run(runs_dir: Path) -> Path:
    # make demo runs only: demo-real-* runs (real models) sort after them and have no stubs.
    runs = sorted(
        p
        for p in runs_dir.glob("demo-*")
        if not p.name.startswith("demo-real-") and (p / "results.jsonl").is_file()
    )
    if not runs:
        raise RejectionError(f"no demo run under {runs_dir}; run `make demo` first")
    return runs[-1]


def pick(run_dir: Path) -> tuple[str, str]:
    """(trace_ref, item_id) of the first noisy run with a rejected bill-and-hold call."""
    for line in (run_dir / "results.jsonl").read_text("utf-8").splitlines():
        r = RunResult.model_validate_json(line)
        if r.model_config_id != CANDIDATE or r.exception_set is None:
            continue
        hit = next((e for e in r.exception_set if e.endswith(REJECTED)), None)
        if hit is None:
            continue
        trace = next(s.trace_ref for s in r.scores if s.node is Node.FLAG_EXCEPTIONS)
        return trace, hit.split(":", 1)[0]
    raise RejectionError(
        f"no {CANDIDATE} run in {run_dir.name} has a rejectable bill-and-hold call"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m feedback.demo")
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = p.parse_args(argv)
    try:
        run_dir = latest_demo_run(args.runs_dir)
        trace, item_id = pick(run_dir)
        expected = {
            "item_id": item_id,
            "type": ExceptionType.BILL_AND_HOLD.value,
            "outcome": JudgmentOutcome.EXCEPTION.value,
        }
        for existing in load_feedback_cases(CASES_DIR):
            same = existing.expected.model_dump(mode="json") == expected
            if same and existing.provenance.reviewer == REVIEWER:
                print(f"feedback case {existing.case_id} already exists; not adding another")
                return 0
        print(f"reviewer {REVIEWER} rejects {trace} ({run_dir.name}): {item_id} -> exception")
        case = ingest(
            trace=trace,
            node=Node.FLAG_EXCEPTIONS.value,
            reviewer=REVIEWER,
            reason=REASON,
            expected=json.dumps(expected),
            created_at=dt.datetime.now(dt.UTC),
            runs_dir=args.runs_dir,
            engagements_dir=ENGAGEMENTS_DIR,
            rejections_dir=REJECTIONS_DIR,
            cases_dir=CASES_DIR,
        )
    except RejectionError as exc:
        if str(exc).startswith("already ingested"):
            print(f"feedback case exists: {exc}")
            return 0
        print(f"demo-feedback: {exc}", file=sys.stderr)
        return 2
    print(f"feedback case {case.case_id} written to {CASES_DIR / (case.case_id + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
