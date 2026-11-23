"""Lines that appear in rendered documents and are quoted as answer-key evidence.

Each line must render as one text node so it survives page normalisation intact.
Nothing here may label an item or name an exception type.
"""

from __future__ import annotations

import datetime as dt

from auditlab.money import format_cents

SELLER = "Harbor Lane Supply Co."
SELLER_ADDRESS = "1180 Wharf Road, Port Halden"
BANK = "Fenmore Savings Bank"
ACCOUNT = "account ending 4417"


def customer_line(customer_id: str) -> str:
    return f"Customer ID: {customer_id}"


def invoice_date_line(date: dt.date) -> str:
    return f"Invoice date: {date.isoformat()}"


def invoice_row(description: str, quantity: int, unit_cents: int, amount_cents: int) -> str:
    """Matches the loader's table linearisation: cells joined by ' | '."""
    return f"{description} | {quantity} | {format_cents(unit_cents)} | {format_cents(amount_cents)}"


def total_line(amount_cents: int) -> str:
    return f"Total due: {format_cents(amount_cents)}"


def ship_date_line(date: dt.date) -> str:
    return f"Ship date: {date.isoformat()}"


def shipped_quantity_line(quantity: int) -> str:
    return f"Quantity shipped: {quantity}"


def deposit_row(date: dt.date, payer: str, invoice_id: str, amount_cents: int) -> str:
    return f"{date.isoformat()} | {payer} | {invoice_id} | {format_cents(amount_cents)}"


def hold_until_line(date: dt.date) -> str:
    return f"Please keep these goods at your warehouse until {date.isoformat()}."


def segregated_line(quantity: int) -> str:
    return f"Units segregated: {quantity}"
