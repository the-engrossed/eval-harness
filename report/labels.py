"""Plain-English labels for the scorecard and its headline: the one lookup table.

Keyed by metric, sampling rule and consistency measure, never by config id.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from auditlab.domain.enums import Verdict
from evals.gate_models import Check, GateResult
from evals.gate_text import NO_DROP, root_causes


class MetricLabel(NamedTuple):
    label: str  # what the metric measures
    failure: str  # what went wrong when it is a root step; {items} filled for sample misses


METRICS = {
    "select_samples.required_recall": MetricLabel(
        "Required samples selected", "Sample selection left out {items}"
    ),
    "select_samples.precision": MetricLabel(
        "Selected samples required", "Sample selection added items the rule does not require"
    ),
    "extract_evidence.citation_validity": MetricLabel(
        "Evidence quotes valid", "Evidence extraction cited quotes that are not in the documents"
    ),
    "extract_evidence.field_exact": MetricLabel(
        "Evidence fields correct", "Evidence extraction got field values wrong"
    ),
    "tie_out.tieout_accuracy": MetricLabel("Amounts tie out", "Tie-out results were wrong"),
    "flag_exceptions.planted_recall": MetricLabel(
        "Planted exceptions found", "Exception flagging missed planted exceptions"
    ),
    "flag_exceptions.false_positive_rate": MetricLabel(
        "False exceptions flagged", "Exception flagging raised exceptions that are not real"
    ),
    "flag_exceptions.judgment_allowed": MetricLabel(
        "Judgment calls allowed", "Judgment calls fell outside the allowed outcomes"
    ),
    "flag_exceptions.feedback_expectation_met": MetricLabel(
        "Reviewer corrections honoured", "Exception flagging repeated a call a reviewer rejected"
    ),
    "draft_workpaper.guard_pass": MetricLabel(
        "Workpaper has no unsupported numbers", "The workpaper stated unsupported numbers"
    ),
}
RULE_ITEMS = {
    "key_item": "key items",
    "cutoff_window": "cutoff-window invoices",
    "credit_memo": "credit-memo invoices",
}
MEASURES = {
    "outcome_agreement": ("Outcomes change between runs", "outcome agreement"),
    "pass_pow_n_share": ("Some cases pass in some runs and fail in others", "pass^N share"),
    "infra_failure_rate": ("Too many runs hit infra failures", "infra failure rate"),
}
RULES_IN_DETAIL = re.compile(r"\(rules: ([a-z_, ]+)\)")


def metric_label(key: str) -> str:
    return METRICS[key].label if key in METRICS else key


def failure_phrase(key: str, detail: str = "") -> str:
    if key not in METRICS:
        return f"{key} dropped"
    found = RULES_IN_DETAIL.search(detail)
    rules = [r.strip() for r in found.group(1).split(",")] if found else []
    items = " and ".join(RULE_ITEMS.get(r, r) for r in rules) or "required samples"
    return METRICS[key].failure.format(items=items)


def cause_lines(result: GateResult, *, with_counts: bool = True) -> list[str]:
    """Readable root causes: one per root step, else one per verdict-moving reason."""
    if result.verdict is Verdict.SHIP:
        return []
    roots = root_causes(result)
    if roots:
        out = []
        for key, count in roots.items():
            detail = next(
                (f.root.detail for f in result.findings if f.root and f.root.metric == key), ""
            )
            phrase = failure_phrase(key, detail)
            plural = "s" if count != 1 else ""
            out.append(
                f"{phrase} (explains {count} critical drop{plural})" if with_counts else phrase
            )
        return out
    out = []
    for f in result.findings:
        if f.verdict is Verdict.SHIP:
            continue
        if f.measure in MEASURES:
            text, label = MEASURES[f.measure]
            value = "no eligible case" if f.candidate_value is None else f"{f.candidate_value:.2f}"
            out.append(f"{text} ({label} {value}, {NO_DROP})" if with_counts else text)
        elif f.check is Check.NONCRITICAL_DROP and f.metric is not None:
            out.append(
                f"{metric_label(f.metric)} dropped" + (f" ({NO_DROP})" if with_counts else "")
            )
    return list(dict.fromkeys(out)) or ["No single cause was attributed"]
