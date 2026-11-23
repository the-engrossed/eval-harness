"""flag_exceptions: planted_recall (critical), false_positive_rate, judgment_allowed.

Judgment items are scored against allowed_outcomes only; rubric scoring comes later.
Metrics whose answer-key side is empty (no planted critical exceptions, no judgment
items) are not emitted.
"""

from __future__ import annotations

from auditlab.agent.state import AgentState
from auditlab.domain.answer_key import AnswerKey
from auditlab.domain.enums import MetricKind, Node
from auditlab.domain.models import ExceptionSet
from auditlab.domain.scores import StepScore
from evals.scorers.common import (
    Metric,
    NodeStatus,
    ids_text,
    make_score,
    node_status,
    output_of,
    share,
    unscored,
)

PLANTED_RECALL = Metric("planted_recall", MetricKind.OUTCOME, critical=True)
FALSE_POSITIVE_RATE = Metric(
    "false_positive_rate", MetricKind.OUTCOME, critical=False, lower_is_better=True
)
JUDGMENT_ALLOWED = Metric("judgment_allowed", MetricKind.OUTCOME, critical=False)


def _metrics(key: AnswerKey) -> list[Metric]:
    metrics = []
    if any(x.critical for x in key.expected_exceptions):
        metrics.append(PLANTED_RECALL)
    metrics.append(FALSE_POSITIVE_RATE)
    if key.judgment_items:
        metrics.append(JUDGMENT_ALLOWED)
    return metrics


def score_exceptions(state: AgentState, key: AnswerKey) -> list[StepScore]:
    status = node_status(state, Node.FLAG_EXCEPTIONS)
    metrics = _metrics(key)
    if (early := unscored(state, status, metrics)) is not None:
        return early
    found = output_of(state.exceptions, status)
    scorers = {
        PLANTED_RECALL.name: _planted_recall,
        FALSE_POSITIVE_RATE.name: _false_positive_rate,
        JUDGMENT_ALLOWED.name: _judgment_allowed,
    }
    return [scorers[m.name](state, status, key, found) for m in metrics]


def _planted_recall(
    state: AgentState, status: NodeStatus, key: AnswerKey, found: ExceptionSet
) -> StepScore:
    flagged = {(x.item_id, x.type) for x in found.exceptions}
    planted = [x for x in key.expected_exceptions if x.critical]
    missed = [f"{x.item_id}:{x.type.value}" for x in planted if (x.item_id, x.type) not in flagged]
    value = share(len(planted) - len(missed), len(planted))
    detail = f"missed: {ids_text(missed)}" if missed else "all planted exceptions found"
    return make_score(state, status, PLANTED_RECALL, value, detail)


def _false_positive_rate(
    state: AgentState, status: NodeStatus, key: AnswerKey, found: ExceptionSet
) -> StepScore:
    if not found.exceptions:
        return make_score(state, status, FALSE_POSITIVE_RATE, 0.0, "0 flagged")
    expected = {(x.item_id, x.type) for x in key.expected_exceptions}
    extra = [
        f"{x.item_id}:{x.type.value}"
        for x in found.exceptions
        if (x.item_id, x.type) not in expected
    ]
    value = share(len(extra), len(found.exceptions))
    detail = f"not in key: {ids_text(extra)}" if extra else "no false positives"
    return make_score(state, status, FALSE_POSITIVE_RATE, value, detail)


def _judgment_allowed(
    state: AgentState, status: NodeStatus, key: AnswerKey, found: ExceptionSet
) -> StepScore:
    decisions = {(d.item_id, d.type): d.outcome for d in found.judgments}
    bad = [
        j.item_id
        for j in key.judgment_items
        if decisions.get((j.item_id, j.type)) not in j.allowed_outcomes
    ]
    value = share(len(key.judgment_items) - len(bad), len(key.judgment_items))
    detail = (
        f"missing or outside allowed_outcomes: {ids_text(bad)}"
        if bad
        else "all judgment outcomes allowed"
    )
    return make_score(state, status, JUDGMENT_ALLOWED, value, detail)
