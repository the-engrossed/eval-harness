"""evals/suites/core-real.json: the real-model suite is s1, mixes A-E, taken from core.json."""

import json

from auditlab.config import REPO_ROOT
from auditlab.domain.suite import Suite

SUITES = REPO_ROOT / "evals" / "suites"


def test_core_real_is_the_s1_cases_of_core_unchanged() -> None:
    real = Suite.model_validate_json((SUITES / "core-real.json").read_text(encoding="utf-8"))
    core = json.loads((SUITES / "core.json").read_text(encoding="utf-8"))
    assert real.name == "core-real"
    assert [c.case_id for c in real.cases] == [f"s1-mix{m}" for m in "ABCDE"]
    by_id = {c["case_id"]: c for c in core["cases"]}
    for case in real.cases:
        assert case.model_dump(mode="json") == by_id[case.case_id]
