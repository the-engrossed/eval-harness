from pathlib import Path

import pytest
from cases import CASE_D

from auditlab.docs.store import DocStore, UnknownItemError
from auditlab.domain.enums import ExceptionType as ET


def store(engagements: Path) -> DocStore:
    return DocStore(engagements / CASE_D.case_id)


def item_of(kind: ET) -> str:
    key = CASE_D.answer_key
    items = [e.item_id for e in key.expected_exceptions if e.type is kind]
    items += [j.item_id for j in key.judgment_items if j.type is kind]
    return items[0]


def test_a_clean_item_returns_invoice_and_bill_of_lading(engagements: Path) -> None:
    clean = next(
        i
        for i in CASE_D.answer_key.required_sample_ids
        if i not in {e.item_id for e in CASE_D.answer_key.expected_exceptions}
        and i not in {j.item_id for j in CASE_D.answer_key.judgment_items}
    )
    assert sorted(store(engagements).item_docs(clean)) == [f"{clean}-bol", f"{clean}-invoice"]


def test_missing_evidence_item_returns_just_its_invoice(engagements: Path) -> None:
    item = item_of(ET.MISSING_EVIDENCE)
    assert list(store(engagements).item_docs(item)) == [f"{item}-invoice"]


def test_bill_and_hold_returns_its_hold_papers(engagements: Path) -> None:
    item = item_of(ET.BILL_AND_HOLD)
    assert sorted(store(engagements).item_docs(item)) == [
        f"{item}-hold-letter",
        f"{item}-invoice",
        f"{item}-warehouse-note",
    ]


@pytest.mark.parametrize("bad", ["INV-9999", "../population", "INV-1001/../x", ""])
def test_ids_without_an_invoice_document_raise_unknown_item(engagements: Path, bad: str) -> None:
    with pytest.raises(UnknownItemError):
        store(engagements).item_docs(bad)


def test_bank_pages_are_looked_up_by_month(engagements: Path) -> None:
    deposit = CASE_D.population.deposits[0]
    month = (deposit.date.year, deposit.date.month)
    pages = store(engagements).bank_pages([month, (2031, 1)])
    [page] = pages.values()
    assert f"Period: {month[0]}-{month[1]:02d}-01 to" in page.text
    assert deposit.invoice_refs[0] in page.text
