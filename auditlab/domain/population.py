"""Engagement population (SPEC.md §4.2). The agent sees this; it never sees the answer key."""

from __future__ import annotations

import datetime as dt
from typing import Self

from pydantic import model_validator

from auditlab.domain.base import Amount, DomainModel, Id, Text, require_unique


class Invoice(DomainModel):
    invoice_id: Id
    customer_id: Id
    invoice_date: dt.date
    ship_date: dt.date
    amount_cents: Amount
    terms: Text


class GLEntry(DomainModel):
    je_id: Id
    invoice_id: Id
    posted_date: dt.date
    amount_cents: Amount
    account: Text


class Deposit(DomainModel):
    deposit_id: Id
    date: dt.date
    amount_cents: Amount
    payer: Text
    invoice_refs: list[Id]


class ShippingDoc(DomainModel):
    bol_id: Id
    invoice_id: Id
    ship_date: dt.date
    carrier: Text


class CreditMemo(DomainModel):
    cm_id: Id
    invoice_id: Id
    date: dt.date
    amount_cents: Amount


class Population(DomainModel):
    invoices: list[Invoice]
    gl_entries: list[GLEntry]
    deposits: list[Deposit]
    shipping_docs: list[ShippingDoc]
    credit_memos: list[CreditMemo]

    @model_validator(mode="after")
    def _ids_unique_and_references_resolve(self) -> Self:
        invoice_ids = require_unique("invoice_id", [i.invoice_id for i in self.invoices])
        require_unique("je_id", [e.je_id for e in self.gl_entries])
        require_unique("deposit_id", [d.deposit_id for d in self.deposits])
        require_unique("bol_id", [b.bol_id for b in self.shipping_docs])
        require_unique("cm_id", [c.cm_id for c in self.credit_memos])
        refs = [
            *(e.invoice_id for e in self.gl_entries),
            *(b.invoice_id for b in self.shipping_docs),
            *(c.invoice_id for c in self.credit_memos),
            *(ref for d in self.deposits for ref in d.invoice_refs),
        ]
        for ref in refs:
            if ref not in invoice_ids:
                raise ValueError(f"unknown invoice_id: {ref}")
        return self
