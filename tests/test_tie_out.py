import datetime as dt

from auditlab.agent.nodes.tie_out import tie_out_items
from auditlab.domain.models import EvidenceField, EvidenceSet
from auditlab.domain.population import GLEntry, Invoice, Population

D = dt.date(2025, 6, 2)


def _population(gl_cents: int = 10_000) -> Population:
    return Population(
        invoices=[
            Invoice(
                invoice_id="INV-1",
                customer_id="C-001",
                invoice_date=D,
                ship_date=D,
                amount_cents=10_000,
                terms="net 30",
            )
        ],
        gl_entries=[
            GLEntry(
                je_id="JE-1",
                invoice_id="INV-1",
                posted_date=D,
                amount_cents=gl_cents,
                account="4000",
            )
        ],
        deposits=[],
        shipping_docs=[],
        credit_memos=[],
    )


def _field(
    field: str, doc_id: str, *, cents: int | None = None, text: str | None = None
) -> EvidenceField:
    return EvidenceField(
        evidence_id=f"ev-INV-1-{field}",
        item_id="INV-1",
        field=field,
        value_cents=cents,
        value_text=text,
        doc_id=doc_id,
        page_no=1,
        quote="q",
    )


DEPOSIT = _field("deposit_amount", "bank-statement", cents=10_000)
BOL = _field("ship_date", "INV-1-bol", text="2025-06-02")


def test_everything_agrees() -> None:
    [t] = tie_out_items(
        _population(), ["INV-1"], EvidenceSet(fields=[DEPOSIT, BOL], dropped=[])
    ).tieouts
    assert (t.gl_delta_cents, t.deposit_delta_cents, t.matched, t.evidence_complete) == (
        0,
        0,
        True,
        True,
    )


def test_gl_difference_is_a_signed_integer_delta() -> None:
    evidence = EvidenceSet(fields=[DEPOSIT, BOL], dropped=[])
    [t] = tie_out_items(_population(gl_cents=9_750), ["INV-1"], evidence).tieouts
    assert t.gl_delta_cents == -250
    assert type(t.gl_delta_cents) is int
    assert t.matched is False


def test_missing_deposit_evidence_means_no_match() -> None:
    [t] = tie_out_items(_population(), ["INV-1"], EvidenceSet(fields=[BOL], dropped=[])).tieouts
    assert (t.deposit_cents, t.deposit_delta_cents, t.matched) == (None, None, False)


def test_missing_bill_of_lading_leaves_matched_alone() -> None:
    [t] = tie_out_items(_population(), ["INV-1"], EvidenceSet(fields=[DEPOSIT], dropped=[])).tieouts
    assert t.matched is True
    assert t.evidence_complete is False


def test_ids_not_in_the_population_are_skipped() -> None:
    out = tie_out_items(_population(), ["INV-1", "INV-404"], EvidenceSet(fields=[], dropped=[]))
    assert [t.item_id for t in out.tieouts] == ["INV-1"]
