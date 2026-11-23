"""flag_exceptions.feedback_expectation_met (critical): a reviewer's corrected expectation."""

from __future__ import annotations

from auditlab.agent.state import AgentState
from auditlab.domain.enums import MetricKind, Node
from auditlab.domain.scores import StepScore
from evals.feedback_cases import FeedbackCase
from evals.scorers import RunGrade
from evals.scorers.common import Metric, make_score, node_status, output_of, unscored

FEEDBACK_EXPECTATION_MET = Metric("feedback_expectation_met", MetricKind.OUTCOME, critical=True)


def score_feedback(state: AgentState, case: FeedbackCase) -> StepScore:
    """Scored on the source run's state; the score is recorded under the feedback case id."""
    status = node_status(state, Node.FLAG_EXCEPTIONS)
    if (early := unscored(state, status, [FEEDBACK_EXPECTATION_MET])) is not None:
        score = early[0]
    else:
        flagged = output_of(state.exceptions, status)
        want = case.expected
        if want.outcome is not None:
            got = [j for j in flagged.judgments if (j.item_id, j.type) == (want.item_id, want.type)]
            met = any(j.outcome is want.outcome for j in got)
            seen = ", ".join(j.outcome.value for j in got) or "no judgment for the item"
            what = f"{want.item_id} {want.type.value} -> {want.outcome.value}"
        else:
            got_types = [e.type for e in flagged.exceptions if e.item_id == want.item_id]
            met = want.type in got_types
            seen = ", ".join(t.value for t in got_types) or "item not flagged"
            what = f"{want.item_id} flagged {want.type.value}"
        detail = f"reviewer expects {what}; got {seen}"
        score = make_score(state, status, FEEDBACK_EXPECTATION_MET, 1.0 if met else 0.0, detail)
    return score.model_copy(update={"case_id": case.case_id})


def feedback_grade(state: AgentState, case: FeedbackCase, full: RunGrade) -> RunGrade:
    """A feedback run keeps only its expectation score; the shadow grade still comes from the
    full grade, so a reviewer-caught miss the deliverable hid counts as missed by final-only."""
    score = score_feedback(state, case)
    failed = not score.passed and not score.skipped
    return RunGrade(
        scores=[score],
        shadow_pass=full.shadow_pass,
        missed_by_final_only=full.shadow_pass and failed,
        judge=full.judge,
    )
