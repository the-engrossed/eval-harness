"""The scorecard's one-sentence headline, built from gate verdicts and root-cause data only.

Wording comes from report/labels.py (keyed by metric, rule and measure, never by config id).
"""

from __future__ import annotations

from collections.abc import Sequence

from auditlab.domain.enums import Verdict
from evals.gate_models import GateResult
from report.labels import cause_lines

VERDICT_PHRASE = {Verdict.DO_NOT_SHIP: "is blocked", Verdict.NEEDS_REVIEW: "needs review"}


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def headline(results: Sequence[GateResult]) -> str:
    if not results:
        return "No candidates in this run."
    n = len(results)
    ships = sum(r.verdict is Verdict.SHIP for r in results)
    noun = "candidate" if n == 1 else "candidates"
    verb = "is" if ships == 1 else "are"
    parts = [f"{ships} of {n} {noun} {verb} safe to ship."]
    for r in results:
        if r.verdict is not Verdict.SHIP:
            why = "; ".join(_lower_first(c) for c in cause_lines(r, with_counts=False))
            parts.append(f"{r.candidate_config_id} {VERDICT_PHRASE[r.verdict]}: {why}.")
    return " ".join(parts)
