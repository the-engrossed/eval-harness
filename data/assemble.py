"""Turn ordered drafts into the §4.2 population records, assigning ids."""

from __future__ import annotations

from auditlab.domain.population import (
    CreditMemo,
    Deposit,
    GLEntry,
    Invoice,
    Population,
    ShippingDoc,
)
from data.base_population import CUSTOMERS, Draft

REVENUE_ACCOUNT = "4000 Sales Revenue"


def assemble(drafts: list[Draft]) -> tuple[Population, list[str]]:
    ids = [f"INV-{1001 + n}" for n in range(len(drafts))]
    invoices: list[Invoice] = []
    gl: list[GLEntry] = []
    bols: list[ShippingDoc] = []
    deposits: list[Deposit] = []
    memos: list[CreditMemo] = []
    for n, (iid, d) in enumerate(zip(ids, drafts, strict=True)):
        invoices.append(
            Invoice(
                invoice_id=iid,
                customer_id=d.customer_id,
                invoice_date=d.invoice_date,
                ship_date=d.ship_date,
                amount_cents=d.amount_cents,
                terms=d.terms,
            )
        )
        gl.append(
            GLEntry(
                je_id=f"JE-{5001 + n}",
                invoice_id=iid,
                posted_date=d.gl_posted,
                amount_cents=d.gl_amount_cents,
                account=REVENUE_ACCOUNT,
            )
        )
        if d.has_bol:
            bol_id = f"BOL-{7001 + len(bols)}"
            bols.append(
                ShippingDoc(bol_id=bol_id, invoice_id=iid, ship_date=d.ship_date, carrier=d.carrier)
            )
        if d.has_deposit:
            if d.deposit_date is None:
                raise ValueError(f"draft for {iid} has a deposit but no deposit date")
            deposits.append(
                Deposit(
                    deposit_id=f"DEP-{3001 + len(deposits)}",
                    date=d.deposit_date,
                    amount_cents=d.amount_cents,
                    payer=CUSTOMERS[d.customer_id],
                    invoice_refs=[iid],
                )
            )
        if d.credit_memo is not None:
            memo_date, memo_cents = d.credit_memo
            memos.append(
                CreditMemo(
                    cm_id=f"CM-{901 + len(memos)}",
                    invoice_id=iid,
                    date=memo_date,
                    amount_cents=memo_cents,
                )
            )
    population = Population(
        invoices=invoices,
        gl_entries=gl,
        deposits=deposits,
        shipping_docs=bols,
        credit_memos=memos,
    )
    return population, ids
