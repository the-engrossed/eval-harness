"""Answer-key evidence for required items, and the quote check that runs before any write."""

from __future__ import annotations

from auditlab.docs.loader import pages_from_html
from auditlab.domain.answer_key import ExpectedEvidence
from auditlab.domain.population import Population
from data import doc_text as T
from data.base_population import Draft
from data.render import BANK_DOC_ID, bank_page_no


def _ev(
    item: str,
    field: str,
    doc_id: str,
    page_no: int,
    quote: str,
    *,
    cents: int | None = None,
    text: str | None = None,
) -> ExpectedEvidence:
    return ExpectedEvidence(
        item_id=item,
        field=field,
        value_cents=cents,
        value_text=text,
        doc_id=doc_id,
        page_no=page_no,
        quote=quote,
    )


def _for_item(iid: str, d: Draft, pop: Population) -> list[ExpectedEvidence]:
    inv = f"{iid}-invoice"
    row = T.invoice_row(d.description, d.quantity, d.unit_price_cents, d.amount_cents)
    out = [
        _ev(iid, "customer_id", inv, 1, T.customer_line(d.customer_id), text=d.customer_id),
        _ev(
            iid,
            "invoice_date",
            inv,
            1,
            T.invoice_date_line(d.invoice_date),
            text=d.invoice_date.isoformat(),
        ),
        _ev(iid, "invoice_quantity", inv, 1, row, text=str(d.quantity)),
        _ev(iid, "invoice_amount", inv, 2, T.total_line(d.amount_cents), cents=d.amount_cents),
    ]
    for bol in (b for b in pop.shipping_docs if b.invoice_id == iid):
        doc = f"{iid}-bol"
        date = bol.ship_date
        out.append(_ev(iid, "ship_date", doc, 1, T.ship_date_line(date), text=date.isoformat()))
        qty = T.shipped_quantity_line(d.shipped_quantity)
        out.append(_ev(iid, "quantity_shipped", doc, 1, qty, text=str(d.shipped_quantity)))
    for dep in (x for x in pop.deposits if iid in x.invoice_refs):
        page = bank_page_no(pop, dep)
        line = T.deposit_row(dep.date, dep.payer, iid, dep.amount_cents)
        out.append(_ev(iid, "deposit_date", BANK_DOC_ID, page, line, text=dep.date.isoformat()))
        out.append(_ev(iid, "deposit_amount", BANK_DOC_ID, page, line, cents=dep.amount_cents))
    if d.hold is not None:
        until = d.hold.hold_until
        out.append(
            _ev(
                iid,
                "hold_until_date",
                f"{iid}-hold-letter",
                1,
                T.hold_until_line(until),
                text=until.isoformat(),
            )
        )
        seg = T.segregated_line(d.quantity)
        out.append(
            _ev(iid, "segregated_quantity", f"{iid}-warehouse-note", 1, seg, text=str(d.quantity))
        )
    return out


def build_evidence(
    required: list[str], drafts: dict[str, Draft], pop: Population
) -> list[ExpectedEvidence]:
    return [ev for iid in required for ev in _for_item(iid, drafts[iid], pop)]


def assert_quotes(docs: dict[str, str], evidence: list[ExpectedEvidence]) -> None:
    """Every quote must be an exact substring of its normalised page. Raises on the first miss."""
    cache: dict[str, dict[int, str]] = {}
    for ev in evidence:
        if ev.doc_id not in docs:
            raise ValueError(f"evidence for {ev.item_id}/{ev.field} cites missing doc {ev.doc_id}")
        if ev.doc_id not in cache:
            cache[ev.doc_id] = {p.page_no: p.text for p in pages_from_html(docs[ev.doc_id])}
        page = cache[ev.doc_id].get(ev.page_no)
        if page is None:
            raise ValueError(f"{ev.doc_id} has no page {ev.page_no} (cited by {ev.item_id})")
        if ev.quote not in page:
            raise ValueError(
                f"{ev.doc_id} page {ev.page_no}: quote for {ev.item_id}/{ev.field} "
                f"not found: {ev.quote!r}"
            )
