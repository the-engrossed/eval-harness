"""Reviewer rejection → critical eval case (SPEC.md §12), end to end on stub runs."""

import datetime as dt
import json
from pathlib import Path

import pytest
from matrix_run import CASE_IDS, run_matrix

from auditlab.config import CONFIG_DIR
from auditlab.domain.enums import Verdict
from evals.feedback_cases import Expectation, FeedbackCase, load_feedback_cases
from evals.gate import evaluate_gate
from evals.gate_cli import run_gate
from evals.gate_models import Check, GateInput, load_gate_config
from evals.results import RunResult
from evals.runner import main as runner_main
from feedback.demo import pick
from feedback.ingest import RejectionError, ingest
from report.scorecard import build_context, render

GATE_TOML = CONFIG_DIR / "gate.toml"
FEEDBACK = "flag_exceptions.feedback_expectation_met"
AT = dt.datetime(2026, 11, 18, 12, 0, tzinfo=dt.UTC)
REVIEWER = "Test Reviewer (synthetic)"
REASON = "The hold letter alone does not meet bill-and-hold criteria."


def expected_json(item_id: str) -> str:
    return json.dumps({"item_id": item_id, "type": "BILL_AND_HOLD", "outcome": "exception"})


def do_ingest(matrix_run: Path, out: Path, trace: str, expected: str | None, **kw: str):
    args = {
        "trace": trace,
        "node": "flag_exceptions",
        "reviewer": REVIEWER,
        "reason": REASON,
        "expected": expected,
        "created_at": AT,
        "runs_dir": matrix_run.parent,
        "engagements_dir": matrix_run.parent.parent / "engagements",
        "rejections_dir": out / "rejections",
        "cases_dir": out / "from_feedback",
    } | kw
    return ingest(**args)


@pytest.fixture(scope="module")
def rejected(matrix_run: Path) -> tuple[str, str]:
    """(trace_ref, item_id) of a noisy run that decided bill-and-hold as no exception."""
    return pick(matrix_run)


@pytest.fixture(scope="module")
def ingested(matrix_run: Path, rejected, tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("fb")
    trace, item = rejected
    case = do_ingest(matrix_run, out, trace, expected_json(item))
    return out, case


@pytest.fixture(scope="module")
def after(matrix_run: Path, ingested, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """prod + noisy rerun with the feedback case included, in its own runs dir."""
    _, case = ingested
    root = matrix_run.parent.parent
    runs = tmp_path_factory.mktemp("after-runs")
    return run_matrix(root, "after-feedback", ["prod", "stub-cand-noisy"], [case], runs_dir=runs)


def test_a_reason_without_the_corrected_outcome_is_rejected(matrix_run, rejected, tmp_path):
    for missing in (None, "", "  "):
        with pytest.raises(RejectionError, match="a reason alone is not ground truth"):
            do_ingest(matrix_run, tmp_path, rejected[0], missing)
    with pytest.raises(RejectionError, match="not a valid expectation"):
        do_ingest(matrix_run, tmp_path, rejected[0], "{not json")
    with pytest.raises(RejectionError, match="outcome is required"):
        do_ingest(matrix_run, tmp_path, rejected[0], '{"item_id": "X", "type": "BILL_AND_HOLD"}')
    assert not (tmp_path / "from_feedback").exists()


def test_unknown_trace_node_or_item_is_rejected(matrix_run, rejected, tmp_path):
    trace, item = rejected
    with pytest.raises(RejectionError, match="not found in any run"):
        do_ingest(matrix_run, tmp_path, "local:gate-e2e/0000000000000000", expected_json(item))
    with pytest.raises(RejectionError, match="not found in any run"):
        do_ingest(matrix_run, tmp_path, "local:no-such-run/abc", expected_json(item))
    with pytest.raises(RejectionError, match="unknown node"):
        do_ingest(matrix_run, tmp_path, trace, expected_json(item), node="review")
    with pytest.raises(RejectionError, match="only"):
        do_ingest(matrix_run, tmp_path, trace, expected_json(item), node="select_samples")
    with pytest.raises(RejectionError, match="not an invoice"):
        do_ingest(matrix_run, tmp_path, trace, expected_json("INV-9999"))
    select_trace = _select_trace(matrix_run)
    with pytest.raises(RejectionError, match="not flag_exceptions"):
        do_ingest(matrix_run, tmp_path, select_trace, expected_json(item))


def _select_trace(matrix_run: Path) -> str:
    line = (matrix_run / "results.jsonl").read_text().splitlines()[0]
    r = RunResult.model_validate_json(line)
    return next(s.trace_ref for s in r.scores if s.node.value == "select_samples")


def test_case_file_has_full_provenance(ingested, rejected):
    out, case = ingested
    trace, item = rejected
    stored = json.loads((out / "from_feedback" / f"{case.case_id}.json").read_text())
    assert stored["source_case_id"] in CASE_IDS
    assert stored["source_run_id"] == "gate-e2e" and stored["source_config_id"] == "stub-cand-noisy"
    assert stored["item_id"] == item and stored["node"] == "flag_exceptions"
    assert stored["expected"] == {"item_id": item, "type": "BILL_AND_HOLD", "outcome": "exception"}
    assert stored["provenance"] == {
        "trace_ref": trace,
        "reviewer": REVIEWER,
        "reason": REASON,
        "created_at": "2026-11-18T12:00:00Z",
    }
    assert stored["critical"] is True and stored["tags"] == ["from_feedback"]
    rejection = json.loads((out / "rejections" / f"{case.case_id}.json").read_text())
    assert rejection["reason"] == REASON and rejection["source"]["case_id"] == case.source_case_id
    assert load_feedback_cases(out / "from_feedback") == [case]


def test_the_same_rejection_twice_is_refused(matrix_run, ingested, rejected):
    out, _ = ingested
    with pytest.raises(RejectionError, match="already ingested"):
        do_ingest(matrix_run, out, rejected[0], expected_json(rejected[1]))


def test_next_eval_includes_the_case_and_noisy_becomes_do_not_ship(matrix_run, ingested, after):
    _, case = ingested
    before = {g.candidate_config_id: g for g in run_gate(matrix_run, GATE_TOML, None)}
    assert before["stub-cand-noisy"].verdict is Verdict.NEEDS_REVIEW
    summary = json.loads((after / "summary.json").read_text())
    assert case.case_id in summary["case_ids"]
    prod, noisy = summary["configs"]
    assert prod["accuracy"][FEEDBACK]["value"] == 1.0
    assert noisy["accuracy"][FEEDBACK]["value"] < 1.0
    [result] = run_gate(after, GATE_TOML, None)
    assert result.candidate_config_id == "stub-cand-noisy"
    assert result.verdict is Verdict.DO_NOT_SHIP
    [miss] = [f for f in result.findings if f.check is Check.FEEDBACK_MISS]
    assert miss.message.startswith("reviewer-confirmed expectation missed in ")
    assert " of 5 runs" in miss.message and miss.case_ids == [case.case_id]
    assert miss.root is not None and miss.root.metric == FEEDBACK
    assert miss.trace_refs and all(t.startswith("local:after-feedback/") for t in miss.trace_refs)


def test_prod_keeps_passing_the_feedback_case(after, ingested):
    _, case = ingested
    rows = [json.loads(line) for line in (after / "results.jsonl").read_text().splitlines()]
    prod = [r for r in rows if r["model_config_id"] == "prod" and r["case_id"] == case.case_id]
    assert len(prod) == 5
    assert all([s["metric"] for s in r["scores"]] == ["feedback_expectation_met"] for r in prod)
    assert all(r["scores"][0]["passed"] for r in prod)


def test_one_missed_run_is_a_hard_fail_even_without_a_mean_drop() -> None:
    def gi(config_id: str, **kw: object) -> GateInput:
        fields: dict[str, object] = {
            "source_run_id": "r",
            "config_id": config_id,
            "simulated": True,
            "suite_hash": "h",
            "case_means": {FEEDBACK: {"fb-0123456789": 1.0}},
            "failures": {},
            "planted_misses": [],
            "pass_pow_n_share": 1.0,
            "outcome_agreement": 1.0,
            "infra_failure_rate": 0.0,
        }
        return GateInput.model_validate(fields | kw)

    miss = {"case_id": "fb-0123456789", "run_index": 2, "trace_ref": "local:r/x", "detail": "d"}
    cand = gi("cand", feedback_misses=[miss], feedback_runs=5)
    result = evaluate_gate(cand, gi("base"), load_gate_config(GATE_TOML))
    assert result.verdict is Verdict.DO_NOT_SHIP
    [f] = result.findings
    assert f.check is Check.FEEDBACK_MISS
    assert f.message == "reviewer-confirmed expectation missed in 1 of 5 runs"


def test_expectation_schema() -> None:
    Expectation(item_id="I", type="CUTOFF_ERROR")
    with pytest.raises(ValueError, match="outcome is required"):
        Expectation(item_id="I", type="PARTIAL_SHIPMENT")
    with pytest.raises(ValueError, match="omit outcome"):
        Expectation(item_id="I", type="CUTOFF_ERROR", outcome="exception")


def test_runner_includes_feedback_by_default_and_no_feedback_skips(
    matrix_run: Path, ingested, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    out, case = ingested
    monkeypatch.delenv("AUDITLAB_JUDGE_CONFIG", raising=False)
    suites = tmp_path / "suites"
    suites.mkdir()
    cases = [{"case_id": c, "seed": 1, "mix": c[-1], "tags": ["t"]} for c in CASE_IDS]
    (suites / "three.json").write_text(json.dumps({"name": "three", "cases": cases}))
    base = [
        "--suite", "three", "--suites-dir", str(suites),
        "--engagements-dir", str(matrix_run.parent.parent / "engagements"),
        "--runs-dir", str(tmp_path / "runs"), "--no-cache",
        "--models", "prod", "--n", "2",
        "--feedback-dir", str(out / "from_feedback"),
    ]  # fmt: skip
    assert runner_main([*base, "--run-id", "with"]) == 0
    assert runner_main([*base, "--run-id", "without", "--no-feedback"]) == 0
    with_ = json.loads((tmp_path / "runs" / "with" / "summary.json").read_text())
    without = json.loads((tmp_path / "runs" / "without" / "summary.json").read_text())
    assert case.case_id in with_["case_ids"] and FEEDBACK in with_["configs"][0]["accuracy"]
    assert case.case_id not in without["case_ids"]
    assert FEEDBACK not in without["configs"][0]["accuracy"]
    assert with_["suite_hash"] != without["suite_hash"]


def test_scorecard_counts_the_case_and_shows_its_row(after, ingested):
    out, _ = ingested
    history = after.parent / "history.jsonl"
    html = render(build_context(after, history, out / "from_feedback", GATE_TOML, AT))
    assert "<strong>1</strong> case from reviewer rejections" in html
    assert "Reviewer corrections honoured" in html and FEEDBACK in html
    assert 'data-verdict="stub-cand-noisy">DO_NOT_SHIP<' in html


def test_feedback_case_model_requires_its_invariants() -> None:
    with pytest.raises(ValueError):
        FeedbackCase.model_validate({"case_id": "not-fb", "critical": False})


def test_latest_demo_run_ignores_demo_real_runs(tmp_path: Path) -> None:
    from feedback.demo import latest_demo_run

    for name in ("demo-20261118T100000Z", "demo-20261118T110000Z", "demo-real-20261123T000000Z"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "results.jsonl").write_text("", encoding="utf-8")
    assert latest_demo_run(tmp_path).name == "demo-20261118T110000Z"
