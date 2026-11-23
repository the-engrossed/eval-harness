"""Seeded engagement generator: `python -m data.generate --seed N --mix A..E`.

Writes population.json, answer_key.json and docs/<doc_id>.html under <out>/s<seed>-mix<mix>/.
Every evidence quote is checked against its rendered page before anything is written.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from auditlab.config import ENGAGEMENTS_DIR
from auditlab.domain.answer_key import (
    AnswerKey,
    ExpectedEvidence,
    ExpectedException,
    JudgmentItem,
)
from auditlab.domain.enums import CRITICAL_TYPES, JUDGMENT_TYPES, JudgmentOutcome
from auditlab.domain.population import Population
from data.assemble import assemble
from data.base_population import Draft, Signature, background
from data.evidence import assert_quotes, build_evidence
from data.mixes import MIXES
from data.plant import RUBRICS, plant
from data.render import render_docs
from data.sampling import required_sample_ids
from data.tieouts import expected_tieout

__all__ = ["ENGAGEMENTS_DIR", "Case", "build_case", "write_case"]


@dataclass(frozen=True)
class Case:
    case_id: str
    seed: int
    mix: str
    population: Population
    answer_key: AnswerKey
    docs: dict[str, str]


def _answer_key(
    population: Population,
    drafts: dict[str, Draft],
    required: list[str],
    evidence: list[ExpectedEvidence],
) -> AnswerKey:
    planted = sorted((iid, d.planted) for iid, d in drafts.items() if d.planted is not None)
    outcomes = [JudgmentOutcome.EXCEPTION, JudgmentOutcome.NO_EXCEPTION_WITH_RATIONALE]
    return AnswerKey(
        required_sample_ids=required,
        evidence=evidence,
        expected_tieouts=[expected_tieout(population, iid) for iid in required],
        expected_exceptions=[
            ExpectedException(item_id=iid, type=kind, critical=kind in CRITICAL_TYPES)
            for iid, kind in planted
            if kind not in JUDGMENT_TYPES
        ],
        judgment_items=[
            JudgmentItem(
                item_id=iid, type=kind, allowed_outcomes=outcomes, rubric=list(RUBRICS[kind])
            )
            for iid, kind in planted
            if kind in JUDGMENT_TYPES
        ],
    )


def build_case(seed: int, mix: str) -> Case:
    if mix not in MIXES:
        raise ValueError(f"unknown mix: {mix}; expected one of {', '.join(MIXES)}")
    rng = random.Random(seed * 10 + list(MIXES).index(mix))
    taken: set[Signature] = set()
    drafts = background(rng, taken)
    for kind in MIXES[mix]:
        drafts += plant(rng, kind, taken)
    drafts.sort(key=lambda d: (d.invoice_date, d.tiebreak, d.rank))
    population, ids = assemble(drafts)
    by_id = dict(zip(ids, drafts, strict=True))
    required = required_sample_ids(population)
    docs = render_docs(by_id, population)
    evidence = build_evidence(required, by_id, population)
    assert_quotes(docs, evidence)
    return Case(
        case_id=f"s{seed}-mix{mix}",
        seed=seed,
        mix=mix,
        population=population,
        answer_key=_answer_key(population, by_id, required, evidence),
        docs=docs,
    )


def write_case(case: Case, out_root: Path) -> Path:
    case_dir = out_root / case.case_id
    docs_dir = case_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, model in (("population", case.population), ("answer_key", case.answer_key)):
        (case_dir / f"{name}.json").write_text(
            model.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    for doc_id, html in sorted(case.docs.items()):
        (docs_dir / f"{doc_id}.html").write_text(html, encoding="utf-8")
    return case_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mix", choices=sorted(MIXES), required=True)
    parser.add_argument("--out", type=Path, default=ENGAGEMENTS_DIR)
    args = parser.parse_args(argv)
    case_dir = write_case(build_case(args.seed, args.mix), args.out)
    print(f"wrote {case_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
