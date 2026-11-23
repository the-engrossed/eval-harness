"""extract_evidence (model): reads sampled items' documents; quote-or-drop.

Bank pages are chosen from each invoice's own date and terms, never from the deposit
records, so the pages sent don't reveal which items have no deposit.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from langchain_core.runnables import RunnableConfig

from auditlab.agent.deps import deps_from
from auditlab.agent.nodes.common import call_model, upstream_failed
from auditlab.agent.state import AgentState
from auditlab.docs.store import BANK_DOC_ID, DocStore, UnknownItemError
from auditlab.domain.enums import Node
from auditlab.domain.models import DroppedField, EvidenceSet
from auditlab.domain.population import Invoice, Population

PAYMENT_GRACE_DAYS = 15
QUOTE_NOT_FOUND = "quote_not_found"
_TERMS = re.compile(r"net (\d+)")


def bank_months(invoice: Invoice) -> list[tuple[int, int]]:
    match = _TERMS.fullmatch(invoice.terms)
    if match is None:
        raise ValueError(f"{invoice.invoice_id}: cannot read payment terms {invoice.terms!r}")
    last = invoice.invoice_date + dt.timedelta(days=int(match.group(1)) + PAYMENT_GRACE_DAYS)
    year, month = invoice.invoice_date.year, invoice.invoice_date.month
    months = []
    while (year, month) <= (last.year, last.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def extract_payload(
    pop: Population, item_ids: list[str], docs: DocStore
) -> tuple[dict[str, Any], list[DroppedField]]:
    invoices = {i.invoice_id: i for i in pop.invoices}
    known: list[str] = []
    dropped: list[DroppedField] = []
    documents: dict[str, dict[str, str]] = {}
    months: set[tuple[int, int]] = set()
    for iid in dict.fromkeys(item_ids):
        try:
            item_docs = docs.item_docs(iid)
        except UnknownItemError:
            item_docs = {}
        if not item_docs or iid not in invoices:
            dropped.append(DroppedField(item_id=iid, field="all", reason="unknown_item"))
            continue
        known.append(iid)
        for doc_id, doc in item_docs.items():
            documents[doc_id] = {str(p.page_no): p.text for p in doc.pages}
        months.update(bank_months(invoices[iid]))
    bank = docs.bank_pages(sorted(months))
    if bank:
        documents[BANK_DOC_ID] = {str(n): bank[n].text for n in sorted(bank)}
    return {"item_ids": known, "documents": documents}, dropped


def enforce_quotes(evidence: EvidenceSet, known: list[str], docs: DocStore) -> EvidenceSet:
    fields = []
    dropped = list(evidence.dropped)
    for f in evidence.fields:
        text = docs.page_text(f.doc_id, f.page_no)
        if f.item_id not in known:
            dropped.append(DroppedField(item_id=f.item_id, field=f.field, reason="not_sampled"))
        elif text is None or f.quote not in text:
            dropped.append(DroppedField(item_id=f.item_id, field=f.field, reason=QUOTE_NOT_FOUND))
        else:
            fields.append(f)
    return EvidenceSet(fields=fields, dropped=dropped)


def extract_evidence(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    if upstream_failed(state):
        return {}
    if state.samples is None:
        raise RuntimeError("extract_evidence ran before select_samples")
    deps = deps_from(config)
    payload, unknown = extract_payload(state.population, state.samples.item_ids, deps.docs)
    output, step = call_model(state, deps, Node.EXTRACT_EVIDENCE, payload, EvidenceSet)
    if output is None:
        return {"step_records": [step]}
    checked = enforce_quotes(output, payload["item_ids"], deps.docs)
    evidence = EvidenceSet(fields=checked.fields, dropped=[*unknown, *checked.dropped])
    return {"evidence": evidence, "step_records": [step]}
