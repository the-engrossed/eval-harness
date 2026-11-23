"""Expected tie-outs for the answer key. Integer cents, deterministic Python."""

from __future__ import annotations

from auditlab.domain.models import TieOut
from auditlab.domain.population import Population


def expected_tieout(pop: Population, invoice_id: str) -> TieOut:
    [invoice] = [i for i in pop.invoices if i.invoice_id == invoice_id]
    gl = [e.amount_cents for e in pop.gl_entries if e.invoice_id == invoice_id]
    deposits = [d for d in pop.deposits if invoice_id in d.invoice_refs]
    for deposit in deposits:
        if len(deposit.invoice_refs) != 1:
            raise ValueError(
                f"deposit {deposit.deposit_id} covers several invoices; "
                "per-invoice tie-out needs one invoice per deposit"
            )
    gl_cents = sum(gl) if gl else None
    deposit_cents = sum(d.amount_cents for d in deposits) if deposits else None
    gl_delta = None if gl_cents is None else gl_cents - invoice.amount_cents
    deposit_delta = None if deposit_cents is None else deposit_cents - invoice.amount_cents
    return TieOut(
        item_id=invoice_id,
        invoice_cents=invoice.amount_cents,
        gl_cents=gl_cents,
        deposit_cents=deposit_cents,
        gl_delta_cents=gl_delta,
        deposit_delta_cents=deposit_delta,
        matched=gl_delta == 0 and deposit_delta == 0,
        evidence_complete=any(b.invoice_id == invoice_id for b in pop.shipping_docs),
    )
