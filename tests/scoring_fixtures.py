"""Shared hand-built AgentState fixtures for scorer tests. No model calls."""

import random

from cases import CASE_D

from auditlab.agent.state import AgentState
from auditlab.docs.loader import pages_from_html
from auditlab.domain.answer_key import AnswerKey
from auditlab.domain.enums import Node
from auditlab.domain.models import (
    EvidenceField,
    EvidenceSet,
    ExceptionSet,
    FlaggedException,
    JudgmentDecision,
    SampleSelection,
    StepError,
    StepRecord,
    TieOutSet,
    Workpaper,
)
from auditlab.domain.scores import StepScore
from auditlab.models.stub import _workpaper

KEY = CASE_D.answer_key
PAGES = {
    doc_id: {p.page_no: p.text for p in pages_from_html(html)}
    for doc_id, html in CASE_D.docs.items()
}
CRITICAL_STEP = {
    "required_recall",
    "citation_validity",
    "field_exact",
    "tieout_accuracy",
    "planted_recall",
}


def ref(node: Node) -> str:
    return f"local:t/{list(Node).index(node) + 1:016x}"


def record(node: Node, error: StepError | None = None) -> StepRecord:
    r = ref(node)
    return StepRecord(
        node=node,
        latency_ms=1,
        tokens_in=0,
        tokens_out=0,
        cost_micros=0,
        trace_ref=r,
        trace_refs={"local": r},
        error=error,
    )


def fields(key: AnswerKey, keep: set[str] | None = None) -> list[EvidenceField]:
    return [
        EvidenceField(evidence_id=f"ev-{e.item_id}-{e.field}", **e.model_dump())
        for e in key.evidence
        if keep is None or e.item_id in keep
    ]


def oracle(key: AnswerKey = KEY, keep: set[str] | None = None) -> AgentState:
    """What a correct (stub-prod) run leaves behind, optionally limited to `keep` items."""
    ids = [i for i in key.required_sample_ids if keep is None or i in keep]
    inside = set(ids)
    return AgentState(
        case_id=CASE_D.case_id,
        run_index=0,
        model_config_id="prod",
        population=CASE_D.population,
        sampling_rule="rule",
        samples=SampleSelection(item_ids=ids, rationale=["key items"]),
        evidence=EvidenceSet(fields=fields(key, inside), dropped=[]),
        tieouts=TieOutSet(tieouts=[t for t in key.expected_tieouts if t.item_id in inside]),
        exceptions=ExceptionSet(
            exceptions=[
                FlaggedException(item_id=x.item_id, type=x.type, rationale="found")
                for x in key.expected_exceptions
                if x.item_id in inside
            ],
            judgments=[
                JudgmentDecision(
                    item_id=j.item_id, type=j.type, outcome=j.allowed_outcomes[0], rationale="r"
                )
                for j in key.judgment_items
                if j.item_id in inside
            ],
        ),
        step_records=[record(n) for n in Node],
    )


def stub_workpaper(state: AgentState) -> Workpaper:
    """Draft a stub-prod workpaper from the state's own outputs (no answer key)."""
    assert state.samples is not None and state.evidence is not None
    assert state.exceptions is not None and state.tieouts is not None
    return _workpaper(
        random.Random(7),
        {
            "evidence": state.evidence.model_dump(mode="json"),
            "exceptions": state.exceptions.model_dump(mode="json"),
        },
    )


def with_workpaper(state: AgentState, workpaper: Workpaper | None = None) -> AgentState:
    return state.model_copy(update={"workpaper": workpaper or stub_workpaper(state)})


def by_metric(scores: list[StepScore]) -> dict[str, StepScore]:
    out = {s.metric: s for s in scores}
    assert len(out) == len(scores)
    return out
