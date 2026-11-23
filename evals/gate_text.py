"""Root-cause text for a GateResult (SPEC.md §11), shared by the gate CLI and the scorecard.

Pure: built from verdicts, findings and root steps only.
"""

from __future__ import annotations

from auditlab.domain.enums import Verdict
from evals.gate_models import Check, GateResult

REASONS = {
    "outcome_agreement": ("inconsistent outcomes across runs", "outcome agreement"),
    "pass_pow_n_share": ("cases that fail in some runs", "pass^N share"),
    "infra_failure_rate": ("infra failures", "infra failure rate"),
}
NO_DROP = "no critical metric dropped"


def unrooted_reasons(result: GateResult) -> list[str]:
    """What moved a verdict that no root step explains: consistency, infra, non-critical."""
    out: list[str] = []
    for f in result.findings:
        if f.verdict is Verdict.SHIP:
            continue
        if f.measure in REASONS:
            reason, label = REASONS[f.measure]
            value = "no eligible case" if f.candidate_value is None else f"{f.candidate_value:.2f}"
            out.append(f"{reason} ({label} {value}, {NO_DROP})")
        elif f.check is Check.NONCRITICAL_DROP and f.metric is not None:
            b, c = f.baseline_value or 0.0, f.candidate_value or 0.0
            out.append(f"non-critical drop ({f.metric} {b:.2f} -> {c:.2f}, {NO_DROP})")
        else:
            out.append(f"{f.message} ({NO_DROP})")
    return out or [f"none attributed ({NO_DROP})"]


def root_causes(result: GateResult) -> dict[str, int]:
    """Root step -> number of critical findings it explains, in first-seen order."""
    counts: dict[str, int] = {}
    for f in result.findings:
        if f.root is not None:
            counts[f.root.metric] = counts.get(f.root.metric, 0) + 1
    return counts


def root_cause_lines(result: GateResult) -> list[str]:
    """One line per root step (or per unrooted reason); none for a SHIP verdict."""
    if result.verdict is Verdict.SHIP:
        return []
    roots = root_causes(result)
    if not roots:
        return unrooted_reasons(result)
    return [
        f"{metric} (explains {count} critical drop{'s' if count != 1 else ''})"
        for metric, count in roots.items()
    ]
