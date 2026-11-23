import datetime as dt
import json
from collections import Counter
from pathlib import Path

from cases import CASES, each_case

from auditlab.domain.enums import ExceptionType
from auditlab.domain.population import Invoice
from data.generate import Case, write_case
from data.mixes import MIXES
from data.sampling import credit_memo_items, cutoff_window, key_items

ET = ExceptionType


def planted(case: Case) -> dict[str, ExceptionType]:
    key = case.answer_key
    return {e.item_id: e.type for e in key.expected_exceptions} | {
        j.item_id: j.type for j in key.judgment_items
    }


def signature(inv: Invoice) -> tuple[str, int, dt.date]:
    return (inv.customer_id, inv.amount_cents, inv.invoice_date)


def signature_groups(case: Case) -> list[list[Invoice]]:
    groups: dict[tuple[str, int, dt.date], list[Invoice]] = {}
    for inv in case.population.invoices:
        groups.setdefault(signature(inv), []).append(inv)
    return [sorted(g, key=lambda i: i.invoice_id) for g in groups.values() if len(g) > 1]


def invoices(case: Case) -> dict[str, Invoice]:
    return {i.invoice_id: i for i in case.population.invoices}


def has_bol(case: Case, item: str) -> bool:
    return any(b.invoice_id == item for b in case.population.shipping_docs)


def has_deposit(case: Case, item: str) -> bool:
    return any(item in d.invoice_refs for d in case.population.deposits)


def items_of(case: Case, kind: ExceptionType) -> list[str]:
    return [item for item, t in planted(case).items() if t is kind]


@each_case
def test_planted_count_and_types_follow_the_mix(case: Case) -> None:
    assert 2 <= len(planted(case)) <= 4
    assert Counter(planted(case).values()) == Counter(MIXES[case.mix])


@each_case
def test_every_planted_item_is_in_the_required_sample(case: Case) -> None:
    assert set(planted(case)) <= set(case.answer_key.required_sample_ids)


def test_mixes_cover_every_type_and_judgment_items_are_complete() -> None:
    assert {t for types in MIXES.values() for t in types} == set(ExceptionType)
    judgments = [j for case in CASES for j in case.answer_key.judgment_items]
    assert judgments
    assert all(j.allowed_outcomes and j.rubric for j in judgments)


@each_case
def test_duplicate_is_billed_twice_for_one_shipment_and_one_payment(case: Case) -> None:
    for item in items_of(case, ET.DUPLICATE_INVOICE):
        dup = invoices(case)[item]
        [group] = [g for g in signature_groups(case) if dup in g]
        assert len(group) == 2 and group[1] == dup
        original = group[0]
        posted = {e.invoice_id for e in case.population.gl_entries}
        assert {original.invoice_id, dup.invoice_id} <= posted
        assert not has_bol(case, item) and not has_deposit(case, item)
        assert has_bol(case, original.invoice_id) and has_deposit(case, original.invoice_id)


@each_case
def test_signature_match_is_expected_as_duplicate_not_missing_evidence(case: Case) -> None:
    for group in signature_groups(case):
        for later in group[1:]:
            assert planted(case).get(later.invoice_id) is ET.DUPLICATE_INVOICE


@each_case
def test_missing_evidence_never_matches_the_duplicate_signature(case: Case) -> None:
    in_groups = {inv.invoice_id for g in signature_groups(case) for inv in g}
    for item in items_of(case, ET.MISSING_EVIDENCE):
        assert item not in in_groups
        assert not has_bol(case, item) and not has_deposit(case, item)


@each_case
def test_cutoff_error_is_required_only_by_the_cutoff_window(case: Case) -> None:
    pop = case.population
    for item in items_of(case, ET.CUTOFF_ERROR):
        inv = invoices(case)[item]
        assert inv.invoice_date == dt.date(2025, 12, 30)
        assert inv.ship_date == dt.date(2026, 1, 3)
        [gl] = [e for e in pop.gl_entries if e.invoice_id == item]
        assert gl.posted_date.year == 2025
        assert item in cutoff_window(pop)
        assert item not in key_items(pop) | credit_memo_items(pop)


@each_case
def test_other_planted_items_sit_outside_the_cutoff_window(case: Case) -> None:
    others = {i for i, t in planted(case).items() if t is not ET.CUTOFF_ERROR}
    assert not others & cutoff_window(case.population)


@each_case
def test_amount_mismatch_has_a_nonzero_gl_delta(case: Case) -> None:
    tieouts = {t.item_id: t for t in case.answer_key.expected_tieouts}
    for item in items_of(case, ET.AMOUNT_MISMATCH):
        assert tieouts[item].gl_delta_cents not in (None, 0)


@each_case
def test_unplanted_invoices_are_clean(case: Case) -> None:
    tieouts = {t.item_id: t for t in case.answer_key.expected_tieouts}
    gl = {e.invoice_id: e for e in case.population.gl_entries}
    for inv in case.population.invoices:
        if inv.invoice_id in planted(case):
            continue
        assert inv.ship_date <= inv.invoice_date
        assert inv.ship_date.year == inv.invoice_date.year
        assert gl[inv.invoice_id].posted_date == inv.invoice_date
        assert gl[inv.invoice_id].amount_cents == inv.amount_cents
        assert has_bol(case, inv.invoice_id) and has_deposit(case, inv.invoice_id)
        if inv.invoice_id in tieouts:
            assert tieouts[inv.invoice_id].matched and tieouts[inv.invoice_id].evidence_complete


@each_case
def test_every_sampling_rule_also_selects_clean_invoices(case: Case) -> None:
    pop = case.population
    involved = set(planted(case)) | {i.invoice_id for g in signature_groups(case) for i in g}
    assert len(key_items(pop) - involved) >= 3
    assert len(cutoff_window(pop) - involved) >= 3
    assert len(credit_memo_items(pop) - involved) >= 1


@each_case
def test_population_size_and_tieout_coverage(case: Case) -> None:
    assert 100 <= len(case.population.invoices) <= 140
    tieout_ids = [t.item_id for t in case.answer_key.expected_tieouts]
    assert tieout_ids == case.answer_key.required_sample_ids


def _cents_values(node: object) -> list[object]:
    if isinstance(node, dict):
        found = [v for k, v in node.items() if k.endswith("_cents")]
        return found + [x for v in node.values() for x in _cents_values(v)]
    if isinstance(node, list):
        return [x for v in node for x in _cents_values(v)]
    return []


def test_amounts_are_int_cents_in_the_written_json(tmp_path: Path) -> None:
    for case in CASES:
        case_dir = write_case(case, tmp_path)
        for name in ("population.json", "answer_key.json"):
            values = _cents_values(json.loads((case_dir / name).read_text(encoding="utf-8")))
            assert values
            assert all(v is None or (type(v) is int) for v in values), (case.case_id, name)
