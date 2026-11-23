"""draft_workpaper: guard_pass (critical) and judge_<criterion> scores (SPEC.md §9.5, §9.9)."""

from __future__ import annotations

from auditlab.agent.state import AgentState
from auditlab.domain.enums import FailureOrigin, MetricKind, Node
from auditlab.domain.scores import StepScore
from auditlab.models.errors import (
    ModelOutputError,
    PromptTooLargeError,
    ProviderError,
    RecordedFailure,
)
from evals.scorers.common import Metric, make_score, node_status, output_of, unscored
from evals.scorers.judge import CRITERIA, Judge, JudgeInput, JudgeResult

GUARD_PASS = Metric("guard_pass", MetricKind.OUTCOME, critical=True)
JUDGE_METRICS = [Metric(f"judge_{c.value}", MetricKind.OUTCOME, critical=False) for c in CRITERIA]


JUDGE_ERRORS = (ProviderError, ModelOutputError, PromptTooLargeError, RecordedFailure)


def score_workpaper(
    state: AgentState, judge: Judge
) -> tuple[list[StepScore], JudgeResult | None, str | None]:
    """Guard and judge scores, the judge result, and the judge's error message if it failed.

    Any judge HTTP, output or size error is the grader's fault, not the agent's: the
    judge_* scores are marked infra (never counted toward accuracy), the run goes on, and
    the runner counts the run in the infra failure rate.
    """
    metrics = [GUARD_PASS, *JUDGE_METRICS]
    status = node_status(state, Node.DRAFT_WORKPAPER)
    if (early := unscored(state, status, metrics)) is not None:
        return early, None, None
    workpaper = output_of(state.workpaper, status)
    if workpaper.guard_violations:
        codes = ", ".join(sorted({v.code.value for v in workpaper.guard_violations}))
        guard = make_score(state, status, GUARD_PASS, 0.0, f"guard violations: {codes}")
    else:
        guard = make_score(state, status, GUARD_PASS, 1.0, "no guard violations")

    try:
        result = judge.grade(JudgeInput.from_state(state))
    except JUDGE_ERRORS as exc:
        message = str(exc)
        failed = [
            make_score(state, status, m, 0.0, f"judge failed (infra): {message}").model_copy(
                update={"failure_origin": FailureOrigin.INFRA}
            )
            for m in JUDGE_METRICS
        ]
        return [guard, *failed], None, message
    by_c = {v.criterion: v for v in result.verdict.verdicts}
    judge_scores = []
    for criterion, metric in zip(CRITERIA, JUDGE_METRICS, strict=True):
        v = by_c[criterion]
        label = "simulated" if result.simulated else "live"
        detail = f"{v.reason} (confidence={v.confidence:.2f}, judge={result.judge_id}, {label})"
        judge_scores.append(make_score(state, status, metric, 1.0 if v.passed else 0.0, detail))
    return [guard, *judge_scores], result, None
