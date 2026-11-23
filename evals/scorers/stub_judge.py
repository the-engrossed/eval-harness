"""Deterministic stub-judge: rubric checks against the agent's own outputs (SPEC.md §9.5).

Never reads the answer key. Simulated = true. Confidence is 1.0 on a pass; on a fail it
is 1 − share, so a near-miss is low-confidence and a total miss is high-confidence.
"""

from __future__ import annotations

import re

from evals.scorers.judge import (
    CRITERIA,
    CriterionVerdict,
    JudgeCriterion,
    JudgeInput,
    JudgeResult,
    JudgeVerdict,
)

JUDGE_ID = "stub-judge@v1"
_ITEM = re.compile(r"\[\[item:([A-Za-z0-9_.-]+)\]\]")
_EV = re.compile(r"\[\[ev:([A-Za-z0-9_.-]+)\]\]")
_ITEM_WITH_EV = re.compile(r"\[\[item:([A-Za-z0-9_.-]+)\]\]\s*\(\[\[ev:([A-Za-z0-9_.-]+)\]\]\)")
PROCEDURES = (
    ("sampl", "sampling / sample selection"),
    ("bill of lading", "bill of lading"),
    ("bank", "bank statement / deposit"),
    ("general ledger", "general ledger"),
)


def _share(hits: int, total: int) -> float:
    return 1.0 if total == 0 else hits / total


def _verdict(criterion: JudgeCriterion, share: float, reason: str) -> CriterionVerdict:
    passed = share == 1.0
    return CriterionVerdict(
        criterion=criterion,
        passed=passed,
        confidence=1.0 if passed else 1.0 - share,
        reason=reason,
    )


def _procedures(inp: JudgeInput) -> CriterionVerdict:
    prose = inp.prose.lower()
    missing = [label for needle, label in PROCEDURES if needle not in prose]
    return _verdict(
        JudgeCriterion.PROCEDURES_DESCRIBED,
        _share(len(PROCEDURES) - len(missing), len(PROCEDURES)),
        "all four procedures named" if not missing else f"missing: {', '.join(missing)}",
    )


def _samples(inp: JudgeInput) -> CriterionVerdict:
    named = set(_ITEM.findall(inp.prose))
    missing = [i for i in inp.item_ids if i not in named]
    return _verdict(
        JudgeCriterion.SAMPLES_IDENTIFIED,
        _share(len(inp.item_ids) - len(missing), len(inp.item_ids)),
        "every sampled id named" if not missing else f"not named: {', '.join(missing[:10])}",
    )


def _type_words(value: str) -> str:
    return value.replace("_", " ").lower()


def _exceptions(inp: JudgeInput) -> CriterionVerdict:
    prose = inp.prose.lower()
    expected = [(x.item_id, _type_words(x.type.value)) for x in inp.exceptions.exceptions] + [
        (j.item_id, _type_words(j.outcome.value)) for j in inp.exceptions.judgments
    ]
    if not expected:
        return _verdict(JudgeCriterion.EXCEPTIONS_EXPLAINED, 1.0, "no exceptions to explain")
    missed = [
        f"{iid}:{label}"
        for iid, label in expected
        if f"[[item:{iid}]]".lower() not in prose or label not in prose
    ]
    return _verdict(
        JudgeCriterion.EXCEPTIONS_EXPLAINED,
        _share(len(expected) - len(missed), len(expected)),
        "every matter explained" if not missed else f"unexplained: {', '.join(missed[:10])}",
    )


def _conclusion(inp: JudgeInput) -> CriterionVerdict:
    prose = inp.prose
    if "conclusion:" not in prose.lower():
        return _verdict(JudgeCriterion.CONCLUSION_SUPPORTED, 0.0, "no Conclusion: sentence")
    flagged = bool(inp.exceptions.exceptions or inp.exceptions.judgments)
    if flagged and not re.search(r"subject to|except", prose, re.IGNORECASE):
        return _verdict(
            JudgeCriterion.CONCLUSION_SUPPORTED,
            0.0,
            "matters were flagged but the conclusion is not qualified",
        )
    return _verdict(JudgeCriterion.CONCLUSION_SUPPORTED, 1.0, "conclusion present and supported")


def _unsupported(inp: JudgeInput) -> CriterionVerdict:
    known_ev = {f.evidence_id: f.item_id for f in inp.evidence.fields}
    problems: list[str] = []
    for ev in _EV.findall(inp.prose):
        if ev not in known_ev:
            problems.append(f"unknown [[ev:{ev}]]")
    for item, ev in _ITEM_WITH_EV.findall(inp.prose):
        owner = known_ev.get(ev)
        if owner is not None and owner != item:
            problems.append(f"[[ev:{ev}]] belongs to {owner}, not {item}")
    flagged = {(x.item_id, _type_words(x.type.value)) for x in inp.exceptions.exceptions}
    flagged |= {(j.item_id, _type_words(j.outcome.value)) for j in inp.exceptions.judgments}
    for match in re.finditer(
        r"(duplicate invoice|cutoff error|amount mismatch|missing evidence|"
        r"exception|no exception with rationale)\s+on\s+\[\[item:([A-Za-z0-9_.-]+)\]\]",
        inp.prose,
        re.IGNORECASE,
    ):
        label, iid = match.group(1).lower(), match.group(2)
        if (iid, label) not in flagged:
            problems.append(f"claimed {label} on {iid} was not flagged")
    total = max(len(problems), 1) if problems else 1
    hits = 0 if problems else 1
    return _verdict(
        JudgeCriterion.NO_UNSUPPORTED_CLAIMS,
        _share(hits, total) if problems else 1.0,
        "no unsupported claims" if not problems else "; ".join(problems[:5]),
    )


_CHECKS = {
    JudgeCriterion.PROCEDURES_DESCRIBED: _procedures,
    JudgeCriterion.SAMPLES_IDENTIFIED: _samples,
    JudgeCriterion.EXCEPTIONS_EXPLAINED: _exceptions,
    JudgeCriterion.CONCLUSION_SUPPORTED: _conclusion,
    JudgeCriterion.NO_UNSUPPORTED_CLAIMS: _unsupported,
}


class StubJudge:
    simulated = True

    def grade(self, inp: JudgeInput) -> JudgeResult:
        verdicts = [_CHECKS[c](inp) for c in CRITERIA]
        return JudgeResult(
            verdict=JudgeVerdict(verdicts=verdicts),
            judge_id=JUDGE_ID,
            simulated=True,
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
        )
