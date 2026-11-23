import datetime as dt

from auditlab.domain.population import CreditMemo, Invoice, Population
from data.sampling import credit_memo_items, cutoff_window, key_items, required_sample_ids


def _inv(invoice_id: str, date: dt.date, amount_cents: int = 10_000) -> Invoice:
    return Invoice(
        invoice_id=invoice_id,
        customer_id="C-001",
        invoice_date=date,
        ship_date=date,
        amount_cents=amount_cents,
        terms="net 30",
    )


def _pop(invoices: list[Invoice], memos: list[CreditMemo] | None = None) -> Population:
    return Population(
        invoices=invoices,
        gl_entries=[],
        deposits=[],
        shipping_docs=[],
        credit_memos=memos or [],
    )


JUNE = dt.date(2025, 6, 1)


def test_key_item_threshold_is_inclusive() -> None:
    pop = _pop([_inv("A", JUNE, 2_499_999), _inv("B", JUNE, 2_500_000)])
    assert key_items(pop) == {"B"}


def test_cutoff_window_bounds_are_inclusive() -> None:
    pop = _pop(
        [
            _inv("before", dt.date(2025, 12, 26)),
            _inv("first", dt.date(2025, 12, 27)),
            _inv("last", dt.date(2026, 1, 5)),
            _inv("after", dt.date(2026, 1, 6)),
        ]
    )
    assert cutoff_window(pop) == {"first", "last"}


def test_credit_memo_items() -> None:
    memo = CreditMemo(cm_id="CM-1", invoice_id="B", date=JUNE, amount_cents=100)
    pop = _pop([_inv("A", JUNE), _inv("B", JUNE)], [memo])
    assert credit_memo_items(pop) == {"B"}


def test_required_sample_is_the_sorted_union_without_repeats() -> None:
    memo = CreditMemo(cm_id="CM-1", invoice_id="Z", date=JUNE, amount_cents=100)
    pop = _pop(
        [
            _inv("Z", dt.date(2025, 12, 30), 3_000_000),
            _inv("M", dt.date(2025, 12, 28)),
            _inv("K", JUNE, 2_600_000),
            _inv("Q", JUNE),
        ],
        [memo],
    )
    assert required_sample_ids(pop) == ["K", "M", "Z"]
