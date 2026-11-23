from pathlib import Path

import pytest
from cases import CASE_D
from scoring_fixtures import (
    CRITICAL_STEP,
    KEY,
    PAGES,
    by_metric,
    fields,
    oracle,
    record,
    ref,
    with_workpaper,
)

from auditlab.agent.nodes.extract_evidence import enforce_quotes
from auditlab.docs.store import DocStore
from auditlab.domain.enums import FailureOrigin, JudgmentOutcome, MetricKind, Node
from auditlab.domain.models import (
    DroppedField,
    EvidenceSet,
    ExceptionSet,
    FlaggedException,
    JudgmentDecision,
    StepError,
    TieOutSet,
)
from auditlab.domain.scores import StepScore
from data.sampling import cutoff_window
from evals.scorers import score_run
from evals.scorers.stub_judge import StubJudge


def _score_run(state=None, key=KEY):  # type: ignore[no-untyped-def]
    s = state or oracle()
    if (
        s.workpaper is None
        and all(r.error is None for r in s.step_records)
        and any(r.node is Node.DRAFT_WORKPAPER for r in s.step_records)
    ):
        s = with_workpaper(s)
    return score_run(s, key, PAGES, StubJudge())


def test_a_correct_run_scores_every_critical_metric_one() -> None:
    scores = by_metric(_score_run().scores)
    assert set(scores) >= CRITICAL_STEP
    for metric in CRITICAL_STEP:
        assert scores[metric].critical and scores[metric].value == 1.0 and scores[metric].passed
    assert scores["precision"].value == 1.0 and not scores["precision"].critical
    assert scores["false_positive_rate"].value == 0.0 and scores["false_positive_rate"].passed
    assert scores["judgment_allowed"].passed
    for s in scores.values():
        if s.node is not Node.DRAFT_WORKPAPER:
            assert s.failure_origin is None and not s.skipped


def test_every_score_has_a_trace_ref_and_a_kind() -> None:
    for s in _score_run().scores:
        assert s.trace_ref
        assert isinstance(s.kind, MetricKind)
    kinds = {s.metric: s.kind for s in _score_run().scores}
    assert kinds["required_recall"] is MetricKind.PROCESS
    assert kinds["planted_recall"] is MetricKind.OUTCOME


def test_missing_the_cutoff_window_lowers_required_recall_and_planted_recall() -> None:
    window = set(cutoff_window(CASE_D.population))
    assert window & {x.item_id for x in KEY.expected_exceptions}
    keep = set(KEY.required_sample_ids) - window
    scores = by_metric(_score_run(oracle(keep=keep)).scores)
    for metric in ("required_recall", "planted_recall"):
        assert scores[metric].value < 1.0 and not scores[metric].passed
        assert scores[metric].failure_origin is FailureOrigin.AGENT
    assert "INV-1117" in scores["planted_recall"].detail
    assert scores["precision"].value == 1.0


def test_a_judgment_outcome_outside_allowed_outcomes_fails() -> None:
    [item] = KEY.judgment_items
    narrow = KEY.model_copy(
        update={
            "judgment_items": [
                item.model_copy(update={"allowed_outcomes": [JudgmentOutcome.EXCEPTION]})
            ]
        }
    )
    state = oracle(narrow)
    assert state.exceptions is not None
    wrong = JudgmentDecision(
        item_id=item.item_id,
        type=item.type,
        outcome=JudgmentOutcome.NO_EXCEPTION_WITH_RATIONALE,
        rationale="r",
    )
    state.exceptions = ExceptionSet(exceptions=state.exceptions.exceptions, judgments=[wrong])
    score = by_metric(_score_run(state, narrow).scores)["judgment_allowed"]
    assert score.value == 0.0 and not score.passed
    assert score.failure_origin is FailureOrigin.AGENT
    assert item.item_id in score.detail

    state.exceptions = ExceptionSet(exceptions=state.exceptions.exceptions, judgments=[])
    assert not by_metric(_score_run(state, narrow).scores)["judgment_allowed"].passed


def test_a_bad_quote_dropped_by_the_node_still_lowers_citation_validity(
    engagements: Path,
) -> None:
    docs = DocStore(engagements / CASE_D.case_id)
    good = fields(KEY)
    first = good[0]
    bad = first.model_copy(update={"quote": "a sentence that is on no page"})
    proposed = EvidenceSet(fields=[bad, *good[1:]], dropped=[])
    kept = enforce_quotes(proposed, KEY.required_sample_ids, docs)
    assert [d.reason for d in kept.dropped] == ["quote_not_found"]

    state = oracle()
    state.evidence = kept
    score = by_metric(_score_run(state).scores)["citation_validity"]
    assert score.value == (len(good) - 1) / len(good)
    assert not score.passed and score.failure_origin is FailureOrigin.AGENT


def test_selection_drops_do_not_count_against_citation_validity() -> None:
    state = oracle()
    assert state.evidence is not None
    state.evidence = EvidenceSet(
        fields=state.evidence.fields,
        dropped=[
            DroppedField(item_id="INV-9999", field="all", reason="unknown_item"),
            DroppedField(item_id="INV-1001", field="ship_date", reason="not_sampled"),
            DroppedField(item_id="INV-1001", field="ship_date", reason="no supporting fields"),
        ],
    )
    assert by_metric(_score_run(state).scores)["citation_validity"].value == 1.0


def test_a_kept_quote_is_rechecked_against_the_page() -> None:
    state = oracle()
    assert state.evidence is not None
    [first, *rest] = state.evidence.fields
    moved = first.model_copy(update={"page_no": first.page_no + 5})
    state.evidence = EvidenceSet(fields=[moved, *rest], dropped=[])
    scores = by_metric(_score_run(state).scores)
    assert scores["citation_validity"].value < 1.0
    assert scores["field_exact"].value < 1.0


def test_a_wrong_value_lowers_field_exact_only() -> None:
    state = oracle()
    assert state.evidence is not None
    fields_ = state.evidence.fields
    i = next(n for n, f in enumerate(fields_) if f.value_cents is not None)
    wrong = fields_[i].model_copy(update={"value_cents": (fields_[i].value_cents or 0) + 1})
    state.evidence = EvidenceSet(fields=[*fields_[:i], wrong, *fields_[i + 1 :]], dropped=[])
    scores = by_metric(_score_run(state).scores)
    assert scores["field_exact"].value < 1.0
    assert wrong.item_id in scores["field_exact"].detail
    assert scores["citation_validity"].value == 1.0


def test_a_wrong_delta_lowers_tieout_accuracy() -> None:
    state = oracle()
    assert state.tieouts is not None
    [first, *rest] = state.tieouts.tieouts
    assert first.gl_cents is not None
    off = first.model_copy(
        update={"gl_cents": first.gl_cents + 100, "gl_delta_cents": 100, "matched": False}
    )
    state.tieouts = TieOutSet(tieouts=[off, *rest])
    score = by_metric(_score_run(state).scores)["tieout_accuracy"]
    assert score.value < 1.0 and first.item_id in score.detail


def test_an_extra_flag_raises_false_positive_rate() -> None:
    state = oracle()
    assert state.exceptions is not None
    clean = next(
        i for i in KEY.required_sample_ids if i not in {x.item_id for x in KEY.expected_exceptions}
    )
    extra = FlaggedException(item_id=clean, type=KEY.expected_exceptions[0].type, rationale="r")
    flagged = [*state.exceptions.exceptions, extra]
    state.exceptions = ExceptionSet(exceptions=flagged, judgments=state.exceptions.judgments)
    score = by_metric(_score_run(state).scores)["false_positive_rate"]
    assert score.value == 1 / len(flagged) and not score.passed and not score.critical


def _failed_at_extract(origin: FailureOrigin) -> object:
    state = oracle()
    state.evidence = state.tieouts = state.exceptions = None
    error = StepError(origin=origin, message="HTTP 500 from provider")
    state.step_records = [record(Node.SELECT_SAMPLES), record(Node.EXTRACT_EVIDENCE, error)]
    return state


def test_a_provider_error_at_extract_is_infra_and_later_nodes_are_skipped() -> None:
    scores = _score_run(_failed_at_extract(FailureOrigin.INFRA)).scores  # type: ignore[arg-type]
    select = [s for s in scores if s.node is Node.SELECT_SAMPLES]
    assert select and all(s.passed for s in select)
    extract = [s for s in scores if s.node is Node.EXTRACT_EVIDENCE]
    assert {s.metric for s in extract} == {"citation_validity", "field_exact"}
    for s in extract:
        assert (s.value, s.passed, s.skipped) == (0.0, False, False)
        assert s.failure_origin is FailureOrigin.INFRA
        assert s.trace_ref == ref(Node.EXTRACT_EVIDENCE)
        assert "HTTP 500" in s.detail
    later = [
        s for s in scores if s.node in (Node.TIE_OUT, Node.FLAG_EXCEPTIONS, Node.DRAFT_WORKPAPER)
    ]
    for s in later:
        assert s.skipped and not s.passed and s.failure_origin is None
        assert s.detail == "upstream_failed"
        assert s.trace_ref == ref(Node.EXTRACT_EVIDENCE)
    assert not any(s.counts_toward_accuracy for s in extract + later)
    assert all(s.counts_toward_accuracy for s in select)


def test_a_model_output_error_is_an_agent_failure_that_counts() -> None:
    scores = _score_run(_failed_at_extract(FailureOrigin.AGENT)).scores  # type: ignore[arg-type]
    extract = [s for s in scores if s.node is Node.EXTRACT_EVIDENCE]
    assert all(s.failure_origin is FailureOrigin.AGENT for s in extract)
    assert all(s.counts_toward_accuracy for s in extract)


def test_a_node_with_no_record_and_no_upstream_failure_is_a_broken_run() -> None:
    state = oracle()
    state.step_records = state.step_records[:2]
    with pytest.raises(ValueError, match="tie_out"):
        _score_run(state)


def _score(**overrides: object) -> StepScore:
    fields_: dict[str, object] = {
        "case_id": "c",
        "run_index": 0,
        "model_config_id": "m",
        "node": Node.TIE_OUT,
        "metric": "tieout_accuracy",
        "kind": MetricKind.PROCESS,
        "value": 1.0,
        "passed": True,
        "critical": True,
        "failure_origin": None,
        "detail": "ok",
        "trace_ref": "local:t/1",
    }
    return StepScore.model_validate({**fields_, **overrides})


def test_step_score_rules() -> None:
    assert _score().counts_toward_accuracy
    with pytest.raises(ValueError, match="failure_origin"):
        _score(passed=False, value=0.0)
    with pytest.raises(ValueError, match="failure_origin"):
        _score(failure_origin=FailureOrigin.AGENT)
    with pytest.raises(ValueError, match="skipped"):
        _score(skipped=True)
    with pytest.raises(ValueError, match="trace_ref"):
        _score(trace_ref="")
    with pytest.raises(ValueError):
        _score(value=1.5)
    for origin in (FailureOrigin.INFRA, FailureOrigin.DATA):
        assert not _score(passed=False, value=0.0, failure_origin=origin).counts_toward_accuracy
    agent = _score(passed=False, value=0.0, failure_origin=FailureOrigin.AGENT)
    assert agent.counts_toward_accuracy
    skipped = _score(passed=False, value=0.0, skipped=True, detail="upstream_failed")
    assert not skipped.counts_toward_accuracy
