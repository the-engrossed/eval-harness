"""The SPEC.md §4.3 sampling rule, as code. The agent gets the rule as text, never this module."""

from __future__ import annotations

import datetime as dt

from auditlab.domain.population import Population

KEY_ITEM_THRESHOLD_CENTS = 2_500_000
CUTOFF_START = dt.date(2025, 12, 27)
CUTOFF_END = dt.date(2026, 1, 5)


def key_items(pop: Population) -> set[str]:
    return {i.invoice_id for i in pop.invoices if i.amount_cents >= KEY_ITEM_THRESHOLD_CENTS}


def cutoff_window(pop: Population) -> set[str]:
    return {i.invoice_id for i in pop.invoices if CUTOFF_START <= i.invoice_date <= CUTOFF_END}


def credit_memo_items(pop: Population) -> set[str]:
    return {c.invoice_id for c in pop.credit_memos}


def required_sample_ids(pop: Population) -> list[str]:
    return sorted(key_items(pop) | cutoff_window(pop) | credit_memo_items(pop))
