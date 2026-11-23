"""tie_out: tieout_accuracy (critical) — matched and both deltas equal the expected tie-out."""

from __future__ import annotations

from auditlab.agent.state import AgentState
from auditlab.domain.answer_key import AnswerKey
from auditlab.domain.enums import MetricKind, Node
from auditlab.domain.models import TieOut
from auditlab.domain.scores import StepScore
from evals.scorers.common import (
    Metric,
    ids_text,
    make_score,
    node_status,
    output_of,
    share,
    unscored,
)

TIEOUT_ACCURACY = Metric("tieout_accuracy", MetricKind.PROCESS, critical=True)


def _verdict(t: TieOut) -> tuple[bool, int | None, int | None]:
    return t.matched, t.gl_delta_cents, t.deposit_delta_cents


def score_tieout(state: AgentState, key: AnswerKey) -> list[StepScore]:
    if not key.expected_tieouts:
        return []
    status = node_status(state, Node.TIE_OUT)
    if (early := unscored(state, status, [TIEOUT_ACCURACY])) is not None:
        return early
    got = {t.item_id: _verdict(t) for t in output_of(state.tieouts, status).tieouts}
    wrong = [t.item_id for t in key.expected_tieouts if got.get(t.item_id) != _verdict(t)]
    correct = len(key.expected_tieouts) - len(wrong)
    detail = f"wrong or missing tie-outs: {ids_text(wrong)}" if wrong else "all tie-outs correct"
    value = share(correct, len(key.expected_tieouts))
    return [make_score(state, status, TIEOUT_ACCURACY, value, detail)]
