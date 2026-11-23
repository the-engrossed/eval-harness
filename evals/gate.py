"""Release gate (SPEC.md §11, §11.1). Pure: no I/O; evals/gate_cli.py reads and prints.

Per-case paired deltas (candidate − baseline, each the mean over that case's counted runs)
with a seeded bootstrap CI on the mean delta. A drop the CI cannot confirm is NEEDS_REVIEW
for a critical metric and a note for a non-critical one. Lower-is-better metrics are
compared as 1 − value, so a drop is always a negative delta.
"""

from __future__ import annotations

from collections.abc import Sequence

from auditlab.domain.enums import Node, Verdict
from evals.gate_input import FEEDBACK, PLANTED, gate_input
from evals.gate_models import (
    Check,
    Failure,
    Finding,
    GateConfig,
    GateInput,
    GateResult,
    RootStep,
)
from evals.gate_stats import bootstrap_ci, mean_over, paired_deltas
from evals.gate_thresholds import thresholds

__all__ = ["bootstrap_ci", "evaluate_gate", "gate_input"]
EPS = 1e-9
MAX_REFS = 10
SEVERITY = {Verdict.SHIP: 0, Verdict.NEEDS_REVIEW: 1, Verdict.DO_NOT_SHIP: 2}
NODE_ORDER = {n.value: i for i, n in enumerate(Node)}


def _refs(failures: Sequence[Failure], cases: set[str]) -> list[str]:
    refs = [f.trace_ref for f in failures if f.case_id in cases]
    return list(dict.fromkeys(refs))[:MAX_REFS]


def _root(
    metric: str,
    cases: set[str],
    dropped: dict[str, set[str]],
    cand: GateInput,
    base: GateInput,
    config: GateConfig,
) -> RootStep:
    """Earliest upstream critical metric that also dropped on every one of these cases."""
    upstream = sorted(
        (
            k
            for k in config.critical
            if NODE_ORDER[k.split(".")[0]] < NODE_ORDER[metric.split(".")[0]]
        ),
        key=lambda k: NODE_ORDER[k.split(".")[0]],
    )
    root = next((k for k in upstream if cases and cases <= dropped.get(k, set())), metric)
    paired = sorted(cand.case_means.get(root, {}).keys() & base.case_means.get(root, {}).keys())
    fails = cand.failures.get(root, []) or (cand.planted_misses if root == PLANTED else [])
    detail = next((f.detail for f in fails if f.case_id in cases), "no failing score detail")
    return RootStep(
        metric=root,
        baseline_value=mean_over(base.case_means[root], paired) if paired else 0.0,
        candidate_value=mean_over(cand.case_means[root], paired) if paired else 0.0,
        detail=detail if len(detail) <= 100 else detail[:97] + "...",
    )


def _drop(
    metric: str, margin: float, cand: GateInput, base: GateInput, config: GateConfig, check: Check
) -> Finding | None:
    deltas = paired_deltas(metric, cand, base)
    if not deltas:
        return None
    n = len(deltas)
    mean = sum(deltas.values()) / n
    if mean >= -margin - EPS:
        return None
    s = config.stats
    ci = bootstrap_ci(list(deltas.values()), s.bootstrap_resamples, s.bootstrap_seed, s.confidence)
    confirmed = ci[1] < 0
    stats = (
        f"mean delta {mean:+.3f}, {s.confidence:.0%} CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
        f"margin {margin:.2f}"
    )
    if confirmed:
        verdict = Verdict.DO_NOT_SHIP if check is Check.CRITICAL_DROP else Verdict.NEEDS_REVIEW
        message = f"drop confirmed over {n} paired cases: {stats}"
    else:
        verdict = Verdict.NEEDS_REVIEW if check is Check.CRITICAL_DROP else Verdict.SHIP
        message = f"drop not confirmed at this suite size: {n} cases ({stats})"
    cases = {k for k, d in deltas.items() if d < -EPS}
    paired = list(deltas)
    return Finding(
        check=check,
        verdict=verdict,
        message=message,
        metric=metric,
        baseline_value=mean_over(base.case_means[metric], paired),
        candidate_value=mean_over(cand.case_means[metric], paired),
        mean_delta=mean,
        ci=ci,
        case_ids=sorted(cases),
        trace_refs=_refs(cand.failures.get(metric, []), cases),
    )


def _hard_fail(
    metric: str,
    check: Check,
    misses: Sequence[Failure],
    message: str,
    cand: GateInput,
    base: GateInput,
    drop: Finding | None,
) -> Finding:
    """A miss of ground truth (planted exception, reviewer-confirmed expectation) in any
    agent-origin run is DO_NOT_SHIP, whatever the statistics say."""
    cases = {f.case_id for f in misses}
    paired = sorted(cand.case_means.get(metric, {}).keys() & base.case_means.get(metric, {}).keys())
    return Finding(
        check=check,
        verdict=Verdict.DO_NOT_SHIP,
        message=message + (f"; {drop.message}" if drop is not None else ""),
        metric=metric,
        baseline_value=mean_over(base.case_means[metric], paired) if paired else None,
        candidate_value=mean_over(cand.case_means[metric], paired) if paired else None,
        mean_delta=drop.mean_delta if drop is not None else None,
        ci=drop.ci if drop is not None else None,
        case_ids=sorted(cases),
        trace_refs=_refs(misses, cases),
    )


def evaluate_gate(candidate: GateInput, baseline: GateInput, config: GateConfig) -> GateResult:
    findings: list[Finding] = []
    dropped = {
        k: {c for c, d in paired_deltas(k, candidate, baseline).items() if d < -EPS}
        for k in config.critical
    }
    for metric, margin in config.critical.items():
        drop = _drop(metric, margin, candidate, baseline, config, Check.CRITICAL_DROP)
        if metric == PLANTED and candidate.planted_misses:
            k = len(candidate.planted_misses)
            message = f"planted critical exception missed in {k} agent-origin runs"
            drop = _hard_fail(
                metric,
                Check.PLANTED_MISS,
                candidate.planted_misses,
                message,
                candidate,
                baseline,
                drop,
            )
        elif metric == FEEDBACK and candidate.feedback_misses:
            k, n = len(candidate.feedback_misses), candidate.feedback_runs
            message = f"reviewer-confirmed expectation missed in {k} of {n} runs"
            drop = _hard_fail(
                metric,
                Check.FEEDBACK_MISS,
                candidate.feedback_misses,
                message,
                candidate,
                baseline,
                drop,
            )
        elif (
            drop is None
            and metric in baseline.case_means
            and not paired_deltas(metric, candidate, baseline)
        ):
            drop = Finding(
                check=Check.CRITICAL_DROP,
                verdict=Verdict.NEEDS_REVIEW,
                metric=metric,
                message="no paired case was scored for the candidate",
            )
        if drop is not None:
            root = (
                _root(metric, set(drop.case_ids), dropped, candidate, baseline, config)
                if drop.case_ids
                else None
            )
            findings.append(drop.model_copy(update={"root": root}))
    others = sorted(
        (candidate.case_means.keys() & baseline.case_means.keys()) - config.critical.keys()
    )
    for metric in others:
        drop = _drop(
            metric, config.default_margin, candidate, baseline, config, Check.NONCRITICAL_DROP
        )
        if drop is not None:
            findings.append(drop)
    findings.extend(thresholds(candidate, config))
    verdict = max((f.verdict for f in findings), key=SEVERITY.__getitem__, default=Verdict.SHIP)
    return GateResult(
        candidate_config_id=candidate.config_id,
        baseline_config_id=baseline.config_id,
        baseline_source=baseline.source_run_id,
        simulated=candidate.simulated or baseline.simulated,
        verdict=verdict,
        findings=findings,
    )
