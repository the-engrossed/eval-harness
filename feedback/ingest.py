"""Reviewer rejection → critical eval case (SPEC.md §12).

`python -m feedback.ingest --trace <ref> --node flag_exceptions --reviewer "<name>"
 --reason "<text>" --expected '{"item_id": "...", "type": "BILL_AND_HOLD", "outcome": "exception"}'`

A rejection needs the corrected outcome: a reason alone is not ground truth and is refused.
The trace must be a score's trace_ref in some runs/<run_id>/results.jsonl for that node.
Writes feedback/rejections/<id>.json and evals/suites/from_feedback/<id>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from auditlab.config import ENGAGEMENTS_DIR, REPO_ROOT, RUNS_DIR
from auditlab.domain.enums import Node
from auditlab.domain.population import Population
from evals.feedback_cases import FEEDBACK_PREFIX, Expectation, FeedbackCase, Provenance
from evals.results import RunResult

REJECTIONS_DIR = REPO_ROOT / "feedback" / "rejections"
CASES_DIR = REPO_ROOT / "evals" / "suites" / "from_feedback"
SUPPORTED_NODES = (Node.FLAG_EXCEPTIONS,)


class RejectionError(ValueError):
    """The rejection cannot become an eval case; the message says why."""


def parse_expected(raw: str | None) -> Expectation:
    if raw is None or not raw.strip():
        raise RejectionError(
            "--expected is required: a reason alone is not ground truth; give the corrected "
            'outcome, e.g. {"item_id": "INV-1110", "type": "BILL_AND_HOLD", "outcome": "exception"}'
        )
    try:
        return Expectation.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RejectionError(f"--expected is not a valid expectation: {exc}") from exc


def parse_node(raw: str) -> Node:
    try:
        node = Node(raw)
    except ValueError as exc:
        raise RejectionError(f"unknown node {raw!r}; nodes: {[n.value for n in Node]}") from exc
    if node not in SUPPORTED_NODES:
        raise RejectionError(f"only {[n.value for n in SUPPORTED_NODES]} rejections become cases")
    return node


def find_trace(runs_dir: Path, ref: str, node: Node) -> tuple[str, RunResult]:
    """The run id and the run result holding a `node` score with exactly this trace_ref."""
    if ref.startswith("local:") and "/" in ref:
        candidates = [runs_dir / ref.removeprefix("local:").split("/", 1)[0]]
    else:
        candidates = sorted(p for p in runs_dir.glob("*") if p.is_dir())
    for run_dir in candidates:
        path = run_dir / "results.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text("utf-8").splitlines():
            if ref not in line:
                continue
            result = RunResult.model_validate_json(line)
            nodes = {s.node for s in result.scores if s.trace_ref == ref}
            if node in nodes:
                return run_dir.name, result
            if nodes:
                got = sorted(n.value for n in nodes)
                raise RejectionError(f"trace {ref} is a {got} trace, not {node.value}")
    raise RejectionError(f"trace {ref} not found in any run under {runs_dir}")


def ingest(
    *,
    trace: str,
    node: str,
    reviewer: str,
    reason: str,
    expected: str | None,
    created_at: dt.datetime,
    runs_dir: Path = RUNS_DIR,
    engagements_dir: Path = ENGAGEMENTS_DIR,
    rejections_dir: Path = REJECTIONS_DIR,
    cases_dir: Path = CASES_DIR,
) -> FeedbackCase:
    want = parse_expected(expected)
    step = parse_node(node)
    if not reviewer.strip() or not reason.strip():
        raise RejectionError("--reviewer and --reason are required")
    run_id, source = find_trace(runs_dir, trace, step)
    if source.case_id.startswith(FEEDBACK_PREFIX):
        raise RejectionError(
            f"trace {trace} is from feedback case {source.case_id}; reject the source run"
        )
    population = Population.model_validate_json(
        (engagements_dir / source.case_id / "population.json").read_text("utf-8")
    )
    if want.item_id not in {i.invoice_id for i in population.invoices}:
        raise RejectionError(f"{want.item_id} is not an invoice in {source.case_id}")
    digest = hashlib.sha256(f"{trace}|{step.value}|{want.model_dump_json()}".encode()).hexdigest()
    case_id = f"{FEEDBACK_PREFIX}{digest[:10]}"
    case_path = cases_dir / f"{case_id}.json"
    if case_path.exists():
        raise RejectionError(f"already ingested as {case_path}")
    case = FeedbackCase(
        case_id=case_id,
        source_case_id=source.case_id,
        source_run_id=run_id,
        source_config_id=source.model_config_id,
        source_run_index=source.run_index,
        node=step,
        item_id=want.item_id,
        expected=want,
        provenance=Provenance(
            trace_ref=trace, reviewer=reviewer, reason=reason, created_at=created_at
        ),
    )
    rejection = {
        "id": case_id,
        "trace_ref": trace,
        "node": step.value,
        "reviewer": reviewer,
        "reason": reason,
        "expected": want.model_dump(mode="json"),
        "created_at": created_at.isoformat(),
        "source": {
            "run_id": run_id,
            "config_id": source.model_config_id,
            "case_id": source.case_id,
            "run_index": source.run_index,
        },
        "case_file": f"{case_id}.json",  # under evals/suites/from_feedback/
    }
    rejections_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    (rejections_dir / f"{case_id}.json").write_text(json.dumps(rejection, indent=2) + "\n", "utf-8")
    case_path.write_text(case.model_dump_json(indent=2) + "\n", "utf-8")
    return case


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m feedback.ingest")
    p.add_argument("--trace", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--expected", help="the corrected outcome as JSON (required)")
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--engagements-dir", type=Path, default=ENGAGEMENTS_DIR)
    p.add_argument("--rejections-dir", type=Path, default=REJECTIONS_DIR)
    p.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    args = p.parse_args(argv)
    try:
        case = ingest(
            trace=args.trace,
            node=args.node,
            reviewer=args.reviewer,
            reason=args.reason,
            expected=args.expected,
            created_at=dt.datetime.now(dt.UTC),
            runs_dir=args.runs_dir,
            engagements_dir=args.engagements_dir,
            rejections_dir=args.rejections_dir,
            cases_dir=args.cases_dir,
        )
    except RejectionError as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 2
    print(f"feedback case {case.case_id}: {args.cases_dir / (case.case_id + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
