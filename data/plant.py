"""Planted exceptions (SPEC.md §4.4).

Each planted item enters the required sample through exactly one rule: cutoff errors only
through the cutoff window, everything else through key items or credit memos and never
inside the window. A fault that ignores one rule then misses only that rule's exceptions.
"""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Callable
from dataclasses import replace

from auditlab.domain.enums import ExceptionType as ET
from data.base_population import (
    KEY_CENTS,
    NORMAL_CENTS,
    Draft,
    Hold,
    Signature,
    add_credit_memo,
    new_draft,
)

OUTSIDE_WINDOW = (dt.date(2025, 2, 1), dt.date(2025, 12, 20))
LATE_DECEMBER = (dt.date(2025, 12, 1), dt.date(2025, 12, 20))
CUTOFF_INVOICE_DATE = dt.date(2025, 12, 30)
CUTOFF_SHIP_DATE = dt.date(2026, 1, 3)

RUBRICS: dict[ET, tuple[str, ...]] = {
    ET.BILL_AND_HOLD: (
        "states that the customer requested the hold",
        "states that the goods were segregated from other inventory",
        "states whether title and risk passed to the customer before year end",
    ),
    ET.PARTIAL_SHIPMENT: (
        "compares the quantity on the bill of lading to the invoiced quantity",
        "states how much of the invoice the shipped quantity supports",
        "states whether revenue for the unshipped quantity was recognised in the year",
    ),
}

Planter = Callable[[random.Random, set[Signature]], list[Draft]]


def _duplicate(rng: random.Random, taken: set[Signature]) -> list[Draft]:
    """One shipment, one payment, billed twice: the copy is posted but has no BOL or deposit."""
    original = new_draft(rng, taken, dates=OUTSIDE_WINDOW, cents=KEY_CENTS)
    copy = replace(
        original,
        has_bol=False,
        has_deposit=False,
        deposit_date=None,
        planted=ET.DUPLICATE_INVOICE,
        rank=1,
    )
    return [original, copy]


def _cutoff(rng: random.Random, taken: set[Signature]) -> list[Draft]:
    window = (CUTOFF_INVOICE_DATE, CUTOFF_INVOICE_DATE)
    draft = new_draft(rng, taken, dates=window, cents=NORMAL_CENTS)
    draft.ship_date = CUTOFF_SHIP_DATE
    draft.planted = ET.CUTOFF_ERROR
    return [draft]


def _amount_mismatch(rng: random.Random, taken: set[Signature]) -> list[Draft]:
    if rng.random() < 0.5:
        draft = new_draft(rng, taken, dates=OUTSIDE_WINDOW, cents=KEY_CENTS)
    else:
        draft = new_draft(rng, taken, dates=OUTSIDE_WINDOW, cents=NORMAL_CENTS)
        add_credit_memo(rng, draft)
    delta = rng.randint(100, min(50_000, draft.amount_cents // 2))
    draft.gl_amount_cents = draft.amount_cents + rng.choice((-1, 1)) * delta
    draft.planted = ET.AMOUNT_MISMATCH
    return [draft]


def _missing_evidence(rng: random.Random, taken: set[Signature]) -> list[Draft]:
    draft = new_draft(rng, taken, dates=OUTSIDE_WINDOW, cents=KEY_CENTS)
    draft.has_bol = False
    draft.has_deposit = False
    draft.deposit_date = None
    draft.planted = ET.MISSING_EVIDENCE
    return [draft]


def _bill_and_hold(rng: random.Random, taken: set[Signature]) -> list[Draft]:
    """Goods held at the customer's request: paid and posted, not shipped."""
    draft = new_draft(rng, taken, dates=LATE_DECEMBER, cents=KEY_CENTS)
    draft.has_bol = False
    draft.hold = Hold(
        request_date=draft.invoice_date - dt.timedelta(days=rng.randint(3, 10)),
        hold_until=dt.date(2026, 1, rng.randint(10, 31)),
        bay=f"{rng.choice('ABCD')}-{rng.randint(1, 12)}",
    )
    draft.planted = ET.BILL_AND_HOLD
    return [draft]


def _partial_shipment(rng: random.Random, taken: set[Signature]) -> list[Draft]:
    """Only the documents show it: the BOL quantity is below the invoice quantity."""
    draft = new_draft(rng, taken, dates=OUTSIDE_WINDOW, cents=KEY_CENTS)
    draft.shipped_quantity = rng.randint(1, draft.quantity - 1)
    draft.planted = ET.PARTIAL_SHIPMENT
    return [draft]


PLANTERS: dict[ET, Planter] = {
    ET.DUPLICATE_INVOICE: _duplicate,
    ET.CUTOFF_ERROR: _cutoff,
    ET.AMOUNT_MISMATCH: _amount_mismatch,
    ET.MISSING_EVIDENCE: _missing_evidence,
    ET.BILL_AND_HOLD: _bill_and_hold,
    ET.PARTIAL_SHIPMENT: _partial_shipment,
}


def plant(rng: random.Random, kind: ET, taken: set[Signature]) -> list[Draft]:
    return PLANTERS[kind](rng, taken)
