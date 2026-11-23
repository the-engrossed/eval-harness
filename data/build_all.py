"""`make data`: rebuild data/engagements/ (build output) and evals/suites/core.json."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

from auditlab.config import REPO_ROOT
from auditlab.domain.enums import JUDGMENT_TYPES
from auditlab.domain.suite import Suite, SuiteCase
from data.generate import ENGAGEMENTS_DIR, Case, build_case, write_case
from data.mixes import JUDGMENT_TAG, MIXES, TAGS

SEEDS = (1, 2, 3, 4)
SUITE_NAME = "core"
SUITE_PATH = REPO_ROOT / "evals" / "suites" / f"{SUITE_NAME}.json"


def build_suite(cases: Sequence[Case]) -> Suite:
    suite_cases = []
    for case in cases:
        kinds = MIXES[case.mix]
        tags = {TAGS[k] for k in kinds} | ({JUDGMENT_TAG} if set(kinds) & JUDGMENT_TYPES else set())
        suite_cases.append(
            SuiteCase(case_id=case.case_id, seed=case.seed, mix=case.mix, tags=sorted(tags))
        )
    return Suite(name=SUITE_NAME, cases=suite_cases)


def suite_json(suite: Suite) -> str:
    return suite.model_dump_json(indent=2) + "\n"


def build_all(out_root: Path, suite_path: Path) -> Suite:
    """Build every case in memory (quotes checked), then replace out_root and write the suite."""
    if out_root.name != "engagements":
        raise ValueError(f"refusing to clear {out_root}: only an 'engagements' folder is cleared")
    cases = [build_case(seed, mix) for seed in SEEDS for mix in MIXES]
    suite = build_suite(cases)
    if out_root.exists():
        shutil.rmtree(out_root)
    for case in cases:
        write_case(case, out_root)
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(suite_json(suite), encoding="utf-8")
    return suite


def main() -> int:
    suite = build_all(ENGAGEMENTS_DIR, SUITE_PATH)
    print(f"wrote {len(suite.cases)} cases to {ENGAGEMENTS_DIR} and suite {SUITE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
