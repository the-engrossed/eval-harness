"""Workpaper prose guard.

G1: after removing well-formed [[ev:<id>]] markers, any remaining digit is UNBOUND_NUMERAL.
    Digits inside a well-formed marker id are not a violation.
G2: every [[...]] token must be a well-formed [[ev:<id>]] whose id is in evidence_ids.
"""

from auditlab.guard import GuardCode, validate_prose

EVIDENCE = frozenset({"ev-inv-0042-amount", "ev-po-0042-qty"})


def test_clean_marked_prose_passes() -> None:
    prose = (
        "Invoice total of [[ev:ev-inv-0042-amount]] agreed to the purchase order "
        "quantity [[ev:ev-po-0042-qty]] without exception."
    )
    assert validate_prose(prose, EVIDENCE) == []


def test_g1_bare_digit_is_a_violation() -> None:
    hits = validate_prose("Invoice total of $4,500.00 agreed.", EVIDENCE)
    assert [v.code for v in hits] == [GuardCode.UNBOUND_NUMERAL]


def test_g1_digit_next_to_a_valid_marker_is_a_violation() -> None:
    hits = validate_prose("Agreed [[ev:ev-inv-0042-amount]] in 2025.", EVIDENCE)
    assert [v.code for v in hits] == [GuardCode.UNBOUND_NUMERAL]


def test_g2_unknown_evidence_id_is_a_violation() -> None:
    hits = validate_prose("Agreed [[ev:ev-ghost]].", EVIDENCE)
    assert [v.code for v in hits] == [GuardCode.BAD_MARKER]


def test_g2_malformed_marker_is_a_violation() -> None:
    hits = validate_prose("Agreed [[fact:abc]] and [[ev:]].", EVIDENCE)
    assert [v.code for v in hits] == [GuardCode.BAD_MARKER, GuardCode.BAD_MARKER]


def test_g2_digits_inside_a_malformed_marker_also_count_as_bare_digits() -> None:
    hits = validate_prose("Agreed [[amount:4500]].", EVIDENCE)
    assert {v.code for v in hits} == {GuardCode.BAD_MARKER, GuardCode.UNBOUND_NUMERAL}


ITEMS = frozenset({"INV-1042", "INV-1043"})


def test_item_markers_resolve_against_sampled_ids() -> None:
    prose = "Invoice [[item:INV-1042]] totalled [[ev:ev-inv-0042-amount]] and agreed."
    assert validate_prose(prose, EVIDENCE, item_ids=ITEMS) == []


def test_bare_amount_and_bare_invoice_id_are_violations() -> None:
    for prose in ("Invoice total was $12,400.", "Invoice INV-1042 agreed."):
        hits = validate_prose(prose, EVIDENCE, item_ids=ITEMS)
        assert [v.code for v in hits] == [GuardCode.UNBOUND_NUMERAL], prose


def test_unresolved_item_and_evidence_markers_are_violations() -> None:
    hits = validate_prose("See [[item:INV-9]] and [[ev:ghost]].", EVIDENCE, item_ids=ITEMS)
    assert [v.code for v in hits] == [GuardCode.BAD_MARKER, GuardCode.BAD_MARKER]


def test_item_marker_is_not_accepted_as_evidence_or_vice_versa() -> None:
    hits = validate_prose(
        "See [[ev:INV-1042]] and [[item:ev-po-0042-qty]].", EVIDENCE, item_ids=ITEMS
    )
    assert [v.code for v in hits] == [GuardCode.BAD_MARKER, GuardCode.BAD_MARKER]
