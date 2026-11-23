from typing import Any

import pytest
from domain_fixtures import (
    answer_key,
    cited,
    invoice_data,
    population,
    step_record,
    tieout,
)
from pydantic import ValidationError

from auditlab.domain.answer_key import AnswerKey, ExpectedException
from auditlab.domain.enums import ExceptionType, JudgmentOutcome
from auditlab.domain.models import EvidenceField, FlaggedException, JudgmentDecision
from auditlab.domain.population import Invoice, Population


def test_exception_type_outside_enum_is_rejected() -> None:
    with pytest.raises(ValidationError, match="type"):
        ExpectedException.model_validate({"item_id": "INV-1", "type": "FRAUD", "critical": True})
    with pytest.raises(ValidationError, match="type"):
        FlaggedException.model_validate({"item_id": "INV-1", "type": "FRAUD", "rationale": "x"})


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        Invoice.model_validate(invoice_data(discount_cents=10))


def _population_dump(**overrides: Any) -> dict[str, Any]:
    data = population().model_dump(mode="json")
    data.update(overrides)
    return data


def test_duplicate_ids_are_rejected() -> None:
    data = _population_dump()
    data["invoices"].append(data["invoices"][0])
    with pytest.raises(ValidationError, match="duplicate invoice_id: INV-1001"):
        Population.model_validate(data)


def test_dangling_invoice_reference_is_rejected() -> None:
    data = _population_dump()
    data["shipping_docs"][0]["invoice_id"] = "INV-9999"
    with pytest.raises(ValidationError, match="unknown invoice_id: INV-9999"):
        Population.model_validate(data)
    data = _population_dump()
    data["deposits"][0]["invoice_refs"] = ["INV-9999"]
    with pytest.raises(ValidationError, match="unknown invoice_id: INV-9999"):
        Population.model_validate(data)


@pytest.mark.parametrize(
    "overrides",
    [{"value_text": "net 30"}, {"value_cents": None}],
    ids=["both", "neither"],
)
def test_cited_value_needs_exactly_one_value(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="exactly one of value_cents or value_text"):
        EvidenceField.model_validate(cited(evidence_id="ev-1", **overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"gl_cents": 2_700_000},
        {"deposit_cents": None},
        {"gl_delta_cents": None},
    ],
)
def test_tieout_must_agree_with_its_own_arithmetic(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="tie-out"):
        tieout(**overrides)


def test_tieout_with_delta_and_no_match_is_valid() -> None:
    t = tieout(gl_cents=2_700_000, gl_delta_cents=-50_000, matched=False)
    assert t.gl_delta_cents == -50_000


def test_missing_bill_of_lading_does_not_affect_matched() -> None:
    t = tieout(evidence_complete=False)
    assert t.matched is True
    assert t.evidence_complete is False


def test_tieout_requires_evidence_complete() -> None:
    data = tieout().model_dump()
    del data["evidence_complete"]
    with pytest.raises(ValidationError, match="evidence_complete"):
        type(tieout()).model_validate(data)


def test_bol_present_is_no_longer_a_field() -> None:
    with pytest.raises(ValidationError, match="bol_present"):
        tieout(bol_present=True)


def test_critical_flag_must_match_exception_type() -> None:
    with pytest.raises(ValidationError, match="critical"):
        ExpectedException(item_id="INV-1", type=ExceptionType.CUTOFF_ERROR, critical=False)
    with pytest.raises(ValidationError, match="critical"):
        ExpectedException(item_id="INV-1", type=ExceptionType.BILL_AND_HOLD, critical=True)


def test_answer_key_items_must_be_in_the_required_sample() -> None:
    data = answer_key().model_dump(mode="json")
    data["required_sample_ids"] = ["INV-1002"]
    with pytest.raises(ValidationError, match="not in required_sample_ids: INV-1001"):
        AnswerKey.model_validate(data)


def test_judgment_and_exception_types_do_not_cross() -> None:
    with pytest.raises(ValidationError, match="judgment type"):
        JudgmentDecision(
            item_id="INV-1",
            type=ExceptionType.CUTOFF_ERROR,
            outcome=JudgmentOutcome.EXCEPTION,
            rationale="x",
        )
    with pytest.raises(ValidationError, match="judgment type"):
        FlaggedException(item_id="INV-1", type=ExceptionType.PARTIAL_SHIPMENT, rationale="x")


def test_step_record_trace_ref_must_be_one_of_its_sink_refs() -> None:
    with pytest.raises(ValidationError, match="trace_ref"):
        step_record(trace_ref="local:run-1/ffffffffffffffff")
