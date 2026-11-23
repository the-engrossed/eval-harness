"""Render HTML evidence documents for every invoice in a case."""

from __future__ import annotations

import calendar
import datetime as dt
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from auditlab.domain.population import Deposit, Population, ShippingDoc
from auditlab.money import format_cents
from data import doc_text as T
from data.base_population import CUSTOMERS, Draft

TEMPLATES = Path(__file__).resolve().parent / "templates"
_ENV = Environment(
    loader=FileSystemLoader(TEMPLATES),
    autoescape=True,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)
COMMON = {
    "seller": T.SELLER,
    "seller_address": T.SELLER_ADDRESS,
    "bank": T.BANK,
    "account": T.ACCOUNT,
}
BANK_DOC_ID = "bank-statement"


def deposit_months(pop: Population) -> list[tuple[int, int]]:
    return sorted({(d.date.year, d.date.month) for d in pop.deposits})


def bank_page_no(pop: Population, deposit: Deposit) -> int:
    return deposit_months(pop).index((deposit.date.year, deposit.date.month)) + 1


def _invoice(iid: str, d: Draft) -> str:
    return _ENV.get_template("invoice.html.j2").render(
        **COMMON,
        invoice_id=iid,
        payer=CUSTOMERS[d.customer_id],
        customer_line=T.customer_line(d.customer_id),
        invoice_date_line=T.invoice_date_line(d.invoice_date),
        terms=d.terms,
        description=d.description,
        quantity=d.quantity,
        unit_price=format_cents(d.unit_price_cents),
        amount=format_cents(d.amount_cents),
        total_line=T.total_line(d.amount_cents),
    )


def _bol(iid: str, d: Draft, bol: ShippingDoc) -> str:
    return _ENV.get_template("bol.html.j2").render(
        **COMMON,
        bol_id=bol.bol_id,
        invoice_id=iid,
        payer=CUSTOMERS[d.customer_id],
        carrier=bol.carrier,
        description=d.description,
        ship_date_line=T.ship_date_line(bol.ship_date),
        shipped_quantity_line=T.shipped_quantity_line(d.shipped_quantity),
    )


def _bank_statement(pop: Population) -> str:
    pages = []
    for year, month in deposit_months(pop):
        rows = sorted(
            (d for d in pop.deposits if (d.date.year, d.date.month) == (year, month)),
            key=lambda d: (d.date, d.deposit_id),
        )
        pages.append(
            {
                "start": dt.date(year, month, 1).isoformat(),
                "end": dt.date(year, month, calendar.monthrange(year, month)[1]).isoformat(),
                "rows": [
                    {
                        "date": d.date.isoformat(),
                        "payer": d.payer,
                        "reference": d.invoice_refs[0],
                        "amount": format_cents(d.amount_cents),
                    }
                    for d in rows
                ],
            }
        )
    return _ENV.get_template("bank_statement.html.j2").render(**COMMON, pages=pages)


def _hold_docs(iid: str, d: Draft) -> dict[str, str]:
    if d.hold is None:
        return {}
    payer = CUSTOMERS[d.customer_id]
    letter = _ENV.get_template("hold_letter.html.j2").render(
        **COMMON,
        invoice_id=iid,
        payer=payer,
        request_date=d.hold.request_date.isoformat(),
        quantity=d.quantity,
        description=d.description,
        hold_until_line=T.hold_until_line(d.hold.hold_until),
    )
    note = _ENV.get_template("warehouse_note.html.j2").render(
        **COMMON,
        invoice_id=iid,
        payer=payer,
        description=d.description,
        segregated_line=T.segregated_line(d.quantity),
        bay=d.hold.bay,
    )
    return {f"{iid}-hold-letter": letter, f"{iid}-warehouse-note": note}


def render_docs(drafts: dict[str, Draft], pop: Population) -> dict[str, str]:
    """doc_id -> HTML for every invoice, every BOL, the bank statement and any hold papers."""
    bols = {b.invoice_id: b for b in pop.shipping_docs}
    docs: dict[str, str] = {}
    for iid in sorted(drafts):
        d = drafts[iid]
        docs[f"{iid}-invoice"] = _invoice(iid, d)
        if iid in bols:
            docs[f"{iid}-bol"] = _bol(iid, d, bols[iid])
        docs.update(_hold_docs(iid, d))
    if pop.deposits:
        docs[BANK_DOC_ID] = _bank_statement(pop)
    return docs
