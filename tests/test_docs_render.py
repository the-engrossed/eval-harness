import re

import pytest
from cases import CASES, each_case

from auditlab.docs.loader import pages_from_html
from auditlab.domain.enums import ExceptionType as ET
from data.evidence import assert_quotes
from data.generate import Case

BANNED = re.compile(
    r"duplicate[\s_-]*invoice|cut[\s_-]*off[\s_-]*error|amount[\s_-]*mismatch"
    r"|missing[\s_-]*evidence|bill[\s_-]*(?:and|&|&amp;)[\s_-]*hold|partial[\s_-]*shipment"
    r"|\bexceptions?\b",
    re.IGNORECASE,
)
INVOICE_FIELDS = {"customer_id", "invoice_date", "invoice_quantity", "invoice_amount"}
BOL_FIELDS = {"ship_date", "quantity_shipped"}
DEPOSIT_FIELDS = {"deposit_date", "deposit_amount"}
HOLD_FIELDS = {"hold_until_date", "segregated_quantity"}


def pages(html: str) -> dict[int, str]:
    return {p.page_no: p.text for p in pages_from_html(html)}


def cents(money: str) -> int:
    sign = -1 if money.startswith("-") else 1
    dollars, rem = money.lstrip("-$").replace(",", "").split(".")
    return sign * (int(dollars) * 100 + int(rem))


def fields_by_item(case: Case) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for ev in case.answer_key.evidence:
        value = ev.value_cents if ev.value_cents is not None else ev.value_text
        out.setdefault(ev.item_id, {})[ev.field] = value
    return out


@each_case
def test_every_quote_is_an_exact_substring_of_its_page(case: Case) -> None:
    cache: dict[str, dict[int, str]] = {}
    for ev in case.answer_key.evidence:
        doc_pages = cache.setdefault(ev.doc_id, pages(case.docs[ev.doc_id]))
        assert ev.quote in doc_pages[ev.page_no], (ev.doc_id, ev.page_no, ev.quote)


def test_a_tampered_quote_is_rejected() -> None:
    case = CASES[0]
    ev = case.answer_key.evidence[0]
    bad = ev.model_copy(update={"quote": ev.quote + " (altered)"})
    with pytest.raises(ValueError, match="not found"):
        assert_quotes(case.docs, [bad])


@each_case
def test_every_invoice_has_documents_whether_or_not_it_is_required(case: Case) -> None:
    pop = case.population
    assert {f"{i.invoice_id}-invoice" for i in pop.invoices} <= set(case.docs)
    bol_docs = {d for d in case.docs if d.endswith("-bol")}
    assert bol_docs == {f"{b.invoice_id}-bol" for b in pop.shipping_docs}


@each_case
def test_bank_statement_has_a_page_per_deposit_month_listing_every_deposit(case: Case) -> None:
    deposits = case.population.deposits
    months = sorted({(d.date.year, d.date.month) for d in deposits})
    statement = pages(case.docs["bank-statement"])
    assert len(statement) == len(months)
    for d in deposits:
        page_no = months.index((d.date.year, d.date.month)) + 1
        assert f"| {d.invoice_refs[0]} |" in statement[page_no]


@each_case
def test_evidence_covers_exactly_the_required_items_with_the_right_fields(case: Case) -> None:
    pop = case.population
    by_item = fields_by_item(case)
    assert set(by_item) == set(case.answer_key.required_sample_ids)
    invoices = {i.invoice_id: i for i in pop.invoices}
    bols = {b.invoice_id for b in pop.shipping_docs}
    deposits = {d.invoice_refs[0]: d for d in pop.deposits}
    holds = {j.item_id for j in case.answer_key.judgment_items if j.type is ET.BILL_AND_HOLD}
    for item, fields in by_item.items():
        expected = set(INVOICE_FIELDS)
        expected |= BOL_FIELDS if item in bols else set()
        expected |= DEPOSIT_FIELDS if item in deposits else set()
        expected |= HOLD_FIELDS if item in holds else set()
        assert set(fields) == expected, item
        assert fields["invoice_amount"] == invoices[item].amount_cents
        if item in deposits:
            assert fields["deposit_amount"] == deposits[item].amount_cents


@each_case
def test_bill_and_hold_has_a_hold_letter_and_a_warehouse_note(case: Case) -> None:
    for j in case.answer_key.judgment_items:
        if j.type is ET.BILL_AND_HOLD:
            assert f"{j.item_id}-hold-letter" in case.docs
            assert f"{j.item_id}-warehouse-note" in case.docs


@each_case
def test_partial_shipment_ships_fewer_units_than_invoiced(case: Case) -> None:
    by_item = fields_by_item(case)
    for j in case.answer_key.judgment_items:
        if j.type is ET.PARTIAL_SHIPMENT:
            fields = by_item[j.item_id]
            assert int(str(fields["quantity_shipped"])) < int(str(fields["invoice_quantity"]))


@each_case
def test_invoice_line_quantity_times_unit_price_is_the_amount(case: Case) -> None:
    for ev in case.answer_key.evidence:
        if ev.field == "invoice_quantity":
            _desc, qty, unit, amount = ev.quote.split(" | ")
            assert int(qty) * cents(unit) == cents(amount)


def test_banned_pattern_catches_every_exception_type_name() -> None:
    for kind in ET:
        assert BANNED.search(kind.value)
        assert BANNED.search(kind.value.replace("_", " ").title())


def test_no_document_names_an_exception_type_or_says_exception() -> None:
    hits = [
        (case.case_id, doc_id, m.group(0))
        for case in CASES
        for doc_id, html in case.docs.items()
        if (m := BANNED.search(html))
    ]
    assert hits == []
