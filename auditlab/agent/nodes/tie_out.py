"""tie_out (code): invoice vs GL vs deposit, integer cents. Never calls a model.

Invoice and GL amounts come from the population; the deposit amount and whether a bill of
lading was found come from extracted evidence, so an extraction miss shows up here.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from auditlab.agent.deps import deps_from
from auditlab.agent.nodes.common import run_code, upstream_failed
from auditlab.agent.state import AgentState
from auditlab.domain.enums import Node
from auditlab.domain.models import EvidenceSet, TieOut, TieOutSet
from auditlab.domain.population import Population

DEPOSIT_FIELD = "deposit_amount"
BOL_FIELDS = frozenset({"ship_date", "quantity_shipped"})


def tie_out_items(pop: Population, item_ids: list[str], evidence: EvidenceSet) -> TieOutSet:
    invoices = {i.invoice_id: i for i in pop.invoices}
    tieouts = []
    for iid in dict.fromkeys(item_ids):
        invoice = invoices.get(iid)
        if invoice is None:
            continue
        gl = [e.amount_cents for e in pop.gl_entries if e.invoice_id == iid]
        deposits = [
            f.value_cents
            for f in evidence.fields
            if f.item_id == iid and f.field == DEPOSIT_FIELD and f.value_cents is not None
        ]
        gl_cents = sum(gl) if gl else None
        deposit_cents = sum(deposits) if deposits else None
        gl_delta = None if gl_cents is None else gl_cents - invoice.amount_cents
        deposit_delta = None if deposit_cents is None else deposit_cents - invoice.amount_cents
        tieouts.append(
            TieOut(
                item_id=iid,
                invoice_cents=invoice.amount_cents,
                gl_cents=gl_cents,
                deposit_cents=deposit_cents,
                gl_delta_cents=gl_delta,
                deposit_delta_cents=deposit_delta,
                matched=gl_delta == 0 and deposit_delta == 0,
                evidence_complete=any(
                    f.item_id == iid and f.field in BOL_FIELDS for f in evidence.fields
                ),
            )
        )
    return TieOutSet(tieouts=tieouts)


def tie_out(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    if upstream_failed(state):
        return {}
    if state.samples is None or state.evidence is None:
        raise RuntimeError("tie_out ran before select_samples and extract_evidence")
    samples, evidence = state.samples, state.evidence
    output, step = run_code(
        state,
        deps_from(config),
        Node.TIE_OUT,
        lambda: tie_out_items(state.population, samples.item_ids, evidence),
    )
    return {"tieouts": output, "step_records": [step]}
