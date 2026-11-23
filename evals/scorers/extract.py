"""extract_evidence: citation_validity, field_exact (both critical).

citation_validity is measured over everything the model proposed, not only what the node
kept: valid quotes / (kept fields + fields dropped as quote_not_found). Kept quotes are
re-checked against the page text. Drops for selection reasons (not_sampled, unknown_item)
are selection errors and are not counted here.
"""

from __future__ import annotations

from auditlab.agent.nodes.extract_evidence import QUOTE_NOT_FOUND
from auditlab.agent.state import AgentState
from auditlab.domain.answer_key import AnswerKey
from auditlab.domain.enums import MetricKind, Node
from auditlab.domain.models import EvidenceField, EvidenceSet
from auditlab.domain.scores import StepScore
from evals.scorers.common import (
    Metric,
    Pages,
    ids_text,
    make_score,
    node_status,
    output_of,
    share,
    unscored,
)

CITATION_VALIDITY = Metric("citation_validity", MetricKind.PROCESS, critical=True)
FIELD_EXACT = Metric("field_exact", MetricKind.PROCESS, critical=True)


def _quote_on_page(field: EvidenceField, pages: Pages) -> bool:
    text = pages.get(field.doc_id, {}).get(field.page_no)
    return text is not None and field.quote in text


def _citation(state: AgentState, evidence: EvidenceSet, pages: Pages) -> tuple[float, str]:
    bad = [f"{f.item_id}:{f.field}" for f in evidence.fields if not _quote_on_page(f, pages)]
    dropped = [f"{d.item_id}:{d.field}" for d in evidence.dropped if d.reason == QUOTE_NOT_FOUND]
    proposed = len(evidence.fields) + len(dropped)
    if proposed == 0:
        return 0.0, "the model proposed no evidence fields"
    invalid = bad + dropped
    valid = proposed - len(invalid)
    detail = f"quotes not on their page: {ids_text(invalid)}" if invalid else "all quotes valid"
    return share(valid, proposed), detail


def score_extract(state: AgentState, key: AnswerKey, pages: Pages) -> list[StepScore]:
    status = node_status(state, Node.EXTRACT_EVIDENCE)
    if (early := unscored(state, status, [CITATION_VALIDITY, FIELD_EXACT])) is not None:
        return early
    evidence = output_of(state.evidence, status)
    citation, citation_detail = _citation(state, evidence, pages)

    got = {
        (f.item_id, f.field, f.value_cents, f.value_text, f.doc_id, f.page_no)
        for f in evidence.fields
    }
    wrong = [
        f"{e.item_id}:{e.field}"
        for e in key.evidence
        if (e.item_id, e.field, e.value_cents, e.value_text, e.doc_id, e.page_no) not in got
    ]
    matched = len(key.evidence) - len(wrong)
    return [
        make_score(state, status, CITATION_VALIDITY, citation, citation_detail),
        make_score(
            state,
            status,
            FIELD_EXACT,
            share(matched, len(key.evidence)),
            f"missing or wrong fields: {ids_text(wrong)}" if wrong else "all expected fields exact",
        ),
    ]
