from pathlib import Path

import pytest
from cases import CASES

from auditlab.config import REPO_ROOT
from data.build_all import build_all, build_suite, suite_json
from data.mixes import MIXES


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_building_twice_gives_identical_bytes_and_clears_stale_files(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    build_all(a / "engagements", a / "core.json")
    build_all(b / "engagements", b / "core.json")
    assert snapshot(a) == snapshot(b)
    assert len(list((a / "engagements").iterdir())) == 20

    (a / "engagements" / "stale.txt").write_text("left over", encoding="utf-8")
    build_all(a / "engagements", a / "core.json")
    assert snapshot(a) == snapshot(b)


def test_refuses_to_clear_anything_but_an_engagements_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="engagements"):
        build_all(tmp_path / "other", tmp_path / "core.json")


def test_suite_has_twenty_cases_with_tags_from_their_mix() -> None:
    suite = build_suite(CASES)
    assert suite.name == "core"
    assert [c.case_id for c in suite.cases] == [
        f"s{seed}-mix{mix}" for seed in range(1, 5) for mix in MIXES
    ]
    for case in suite.cases:
        types = {t.value for t in MIXES[case.mix]}
        assert ("cutoff" in case.tags) == ("CUTOFF_ERROR" in types)
        assert ("duplicate" in case.tags) == ("DUPLICATE_INVOICE" in types)
        judgment = bool(types & {"BILL_AND_HOLD", "PARTIAL_SHIPMENT"})
        assert ("judgment" in case.tags) == judgment


def test_committed_suite_file_matches_the_generator() -> None:
    committed = (REPO_ROOT / "evals" / "suites" / "core.json").read_text(encoding="utf-8")
    assert committed == suite_json(build_suite(CASES))
