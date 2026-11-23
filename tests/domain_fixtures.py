"""Valid synthetic domain objects. All names are invented."""

import datetime as dt
from typing import Any

from pydantic import BaseModel

from auditlab.domain.answer_key import (
    AnswerKey,
    ExpectedEvidence,
    ExpectedException,
    JudgmentItem,
)
from auditlab.domain.enums import ExceptionType, JudgmentOutcome, Node
from auditlab.domain.models import (
    DroppedField,
    EvidenceField,
    EvidenceSet,
    ExceptionSet,
    FlaggedException,
    GuardFinding,
    JudgmentDecision,
    SampleSelection,
    StepRecord,
    TieOut,
    TieOutSet,
    Workpaper,
)
from auditlab.domain.population import (
    CreditMemo,
    Deposit,
    GLEntry,
    Invoice,
    Population,
    ShippingDoc,
)
from auditlab.guard import GuardCode


def invoice_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "invoice_id": "INV-1001",
        "customer_id": "C-014",
        "invoice_date": "2025-12-30",
        "ship_date": "2026-01-03",
        "amount_cents": 2_750_000,
        "terms": "net 30",
    }
    data.update(overrides)
    return data


def population() -> Population:
    return Population(
        invoices=[
            Invoice.model_validate(invoice_data()),
            Invoice.model_validate(
                invoice_data(invoice_id="INV-1002", invoice_date="2025-11-04", amount_cents=48_200)
            ),
        ],
        gl_entries=[
            GLEntry(
                je_id="JE-5001",
                invoice_id="INV-1001",
                posted_date=dt.date(2025, 12, 30),
                amount_cents=2_750_000,
                account="4000 Revenue",
            ),
            GLEntry(
                je_id="JE-5002",
                invoice_id="INV-1002",
                posted_date=dt.date(2025, 11, 4),
                amount_cents=48_200,
                account="4000 Revenue",
            ),
        ],
        deposits=[
            Deposit(
                deposit_id="DEP-301",
                date=dt.date(2026, 1, 29),
                amount_cents=2_750_000,
                payer="Tidewell Hardware",
                invoice_refs=["INV-1001"],
            )
        ],
        shipping_docs=[
            ShippingDoc(
                bol_id="BOL-77",
                invoice_id="INV-1001",
                ship_date=dt.date(2026, 1, 3),
                carrier="Northgate Freight",
            )
        ],
        credit_memos=[
            CreditMemo(
                cm_id="CM-9", invoice_id="INV-1002", date=dt.date(2025, 12, 1), amount_cents=4_800
            )
        ],
    )


def tieout(**overrides: Any) -> TieOut:
    data: dict[str, Any] = {
        "item_id": "INV-1001",
        "invoice_cents": 2_750_000,
        "gl_cents": 2_750_000,
        "deposit_cents": 2_750_000,
        "evidence_complete": True,
        "gl_delta_cents": 0,
        "deposit_delta_cents": 0,
        "matched": True,
    }
    data.update(overrides)
    return TieOut.model_validate(data)


def cited(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "item_id": "INV-1001",
        "field": "invoice_total",
        "value_cents": 2_750_000,
        "doc_id": "doc-inv-1001",
        "page_no": 1,
        "quote": "Total due: $27,500.00",
    }
    data.update(overrides)
    return data


def answer_key() -> AnswerKey:
    return AnswerKey(
        required_sample_ids=["INV-1001", "INV-1002"],
        evidence=[ExpectedEvidence.model_validate(cited())],
        expected_tieouts=[tieout()],
        expected_exceptions=[
            ExpectedException(item_id="INV-1001", type=ExceptionType.CUTOFF_ERROR, critical=True)
        ],
        judgment_items=[
            JudgmentItem(
                item_id="INV-1002",
                type=ExceptionType.BILL_AND_HOLD,
                allowed_outcomes=[
                    JudgmentOutcome.EXCEPTION,
                    JudgmentOutcome.NO_EXCEPTION_WITH_RATIONALE,
                ],
                rubric=["mentions the customer hold request", "mentions title transfer"],
            )
        ],
    )


def step_record(**overrides: Any) -> StepRecord:
    data: dict[str, Any] = {
        "node": Node.TIE_OUT,
        "latency_ms": 3,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_micros": 0,
        "trace_ref": "local:run-1/00f067aa0ba902b7",
        "trace_refs": {"local": "local:run-1/00f067aa0ba902b7"},
    }
    data.update(overrides)
    return StepRecord.model_validate(data)


def agent_outputs() -> list[BaseModel]:
    return [
        SampleSelection(
            item_ids=["INV-1001", "INV-1002"],
            rationale=["rule 1: key items", "rule 2: cutoff window", "rule 3: credit memos"],
        ),
        EvidenceSet(
            fields=[EvidenceField.model_validate(cited(evidence_id="ev-inv-1001-total"))],
            dropped=[DroppedField(item_id="INV-1002", field="ship_date", reason="no BOL page")],
        ),
        TieOutSet(tieouts=[tieout()]),
        ExceptionSet(
            exceptions=[
                FlaggedException(
                    item_id="INV-1001",
                    type=ExceptionType.CUTOFF_ERROR,
                    rationale="shipped after year end, revenue booked in the year",
                )
            ],
            judgments=[
                JudgmentDecision(
                    item_id="INV-1002",
                    type=ExceptionType.BILL_AND_HOLD,
                    outcome=JudgmentOutcome.NO_EXCEPTION_WITH_RATIONALE,
                    rationale="customer asked for the hold; goods segregated; title passed",
                )
            ],
        ),
        Workpaper(
            prose="Invoice total [[ev:ev-inv-1001-total]] agreed to the ledger.",
            guard_violations=[GuardFinding(code=GuardCode.BAD_MARKER, message="example")],
        ),
        step_record(),
    ]
