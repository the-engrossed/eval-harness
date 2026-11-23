"""Seeded background population: clean invoices only. Every name is invented."""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass

from auditlab.domain.enums import ExceptionType
from data.sampling import CUTOFF_END, CUTOFF_START, KEY_ITEM_THRESHOLD_CENTS

FIRST_DAY = dt.date(2025, 1, 2)
TERMS = {"net 15": 15, "net 30": 30, "net 45": 45}
CARRIERS = ("Lanternway Freight", "Cobalt Line Haulage", "Pinecrest Carriers")
_STEMS = ("Tidewell", "Ashgrove", "Kestrel", "Millbrook", "Saltmarsh", "Copperfield")
_KINDS = ("Hardware", "Builders", "Outfitters", "Trading", "Home Supply")
CUSTOMERS: dict[str, str] = {
    f"C-{n:03d}": f"{stem} {kind}"
    for n, (stem, kind) in enumerate(((s, k) for s in _STEMS for k in _KINDS), start=1)
}

PRODUCTS = (
    "Steel pallet racking upright, 12 ft",
    "Wire mesh decking panel",
    "Heavy-duty shelving unit",
    "Industrial hand truck",
    "Stretch wrap roll, 18 in",
    "Loading dock bumper",
    "Plastic storage bin, large",
    "Safety bollard, 42 in",
)
QUANTITY_RANGE = (2, 400)

RANDOM_BACKGROUND = 105
NORMAL_CENTS = (15_000, KEY_ITEM_THRESHOLD_CENTS - 100_000)
KEY_CENTS = (KEY_ITEM_THRESHOLD_CENTS, 9_500_000)

Signature = tuple[str, int, dt.date]


@dataclass(frozen=True)
class Hold:
    request_date: dt.date
    hold_until: dt.date
    bay: str


@dataclass
class Draft:
    """One invoice and its supporting records, before ids are assigned.

    Quantities, descriptions and hold details appear only in the rendered documents.
    """

    customer_id: str
    invoice_date: dt.date
    ship_date: dt.date
    description: str
    quantity: int
    unit_price_cents: int
    shipped_quantity: int
    terms: str
    carrier: str
    tiebreak: float
    gl_amount_cents: int
    gl_posted: dt.date
    has_bol: bool = True
    has_deposit: bool = True
    deposit_date: dt.date | None = None
    credit_memo: tuple[dt.date, int] | None = None
    hold: Hold | None = None
    planted: ExceptionType | None = None
    rank: int = 0  # breaks exact ties so a duplicate always gets the later invoice id

    @property
    def amount_cents(self) -> int:
        return self.quantity * self.unit_price_cents

    @property
    def signature(self) -> Signature:
        return (self.customer_id, self.amount_cents, self.invoice_date)


def _quantity_and_price(rng: random.Random, cents: tuple[int, int]) -> tuple[int, int]:
    """A quantity and unit price whose product falls inside `cents`."""
    low, high = cents
    while True:
        quantity = rng.randint(*QUANTITY_RANGE)
        price_low, price_high = -(-low // quantity), high // quantity
        if price_low <= price_high:
            return quantity, rng.randint(price_low, price_high)


def day_between(rng: random.Random, start: dt.date, end: dt.date) -> dt.date:
    return start + dt.timedelta(days=rng.randint(0, (end - start).days))


def new_draft(
    rng: random.Random,
    taken: set[Signature],
    *,
    dates: tuple[dt.date, dt.date],
    cents: tuple[int, int],
) -> Draft:
    """A clean draft whose (customer, amount, date) signature is not already taken."""
    while True:
        customer = rng.choice(sorted(CUSTOMERS))
        invoice_date = day_between(rng, *dates)
        quantity, unit_price = _quantity_and_price(rng, cents)
        amount = quantity * unit_price
        if (customer, amount, invoice_date) not in taken:
            break
    taken.add((customer, amount, invoice_date))
    ship_date = invoice_date - dt.timedelta(days=rng.randint(0, 2))
    if ship_date.year != invoice_date.year:
        ship_date = invoice_date
    terms = rng.choice(sorted(TERMS))
    return Draft(
        customer_id=customer,
        invoice_date=invoice_date,
        ship_date=ship_date,
        description=rng.choice(PRODUCTS),
        quantity=quantity,
        unit_price_cents=unit_price,
        shipped_quantity=quantity,
        terms=terms,
        carrier=rng.choice(CARRIERS),
        tiebreak=rng.random(),
        gl_amount_cents=amount,
        gl_posted=invoice_date,
        deposit_date=invoice_date + dt.timedelta(days=TERMS[terms] + rng.randint(-5, 10)),
    )


def add_credit_memo(rng: random.Random, draft: Draft) -> None:
    amount = max(1, draft.amount_cents * rng.randint(5, 25) // 100)
    draft.credit_memo = (draft.invoice_date + dt.timedelta(days=rng.randint(5, 25)), amount)


def background(rng: random.Random, taken: set[Signature]) -> list[Draft]:
    """Clean invoices, with at least 3 key items, 3 in the cutoff window and 1 credit memo."""
    before_window = CUTOFF_START - dt.timedelta(days=1)
    whole_year = (FIRST_DAY, CUTOFF_END)
    drafts = [
        new_draft(rng, taken, dates=whole_year, cents=NORMAL_CENTS)
        for _ in range(RANDOM_BACKGROUND)
    ]
    drafts += [
        new_draft(rng, taken, dates=(FIRST_DAY, before_window), cents=KEY_CENTS)
        for _ in range(rng.randint(3, 5))
    ]
    drafts += [
        new_draft(rng, taken, dates=(CUTOFF_START, CUTOFF_END), cents=NORMAL_CENTS)
        for _ in range(rng.randint(3, 5))
    ]
    memo_drafts = [
        new_draft(rng, taken, dates=(FIRST_DAY, before_window), cents=NORMAL_CENTS)
        for _ in range(rng.randint(1, 3))
    ]
    for draft in memo_drafts:
        add_credit_memo(rng, draft)
    return drafts + memo_drafts
