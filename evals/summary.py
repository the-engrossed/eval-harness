"""Per-config aggregation and the saturation check (SPEC.md §10, §10.1). Pure, no I/O.

Accuracy uses only scores that count toward accuracy (SPEC.md §9.8): infra, data and
skipped scores never lower it. Runs with an infra or data failure are left out of
pass^N, outcome agreement and the per-workpaper latency and cost numbers. A judge failure
counts in the infra rate only; the run's agent scores stay in.

TODO(later task): when LangSmith or Langfuse is on, attach every StepScore to its trace
(LangSmith feedback / Langfuse score, keyed by metric name), per SPEC.md §7.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from auditlab.domain.scores import StepScore
from auditlab.models.config import ModelConfig
from evals.results import ConfigSummary, MetricAccuracy, RunResult, mean_micros
from evals.scorers import LOWER_IS_BETTER

SATURATION = 0.98
PASS_POW_N_METRICS = frozenset({"judgment_allowed"})
NO_OUTPUT = ("<no flag_exceptions output>",)


def metric_key(s: StepScore) -> str:
    return f"{s.node.value}.{s.metric}"


def run_passes(scores: Sequence[StepScore]) -> bool:
    """Critical metrics plus judgment_allowed. precision, false_positive_rate and judge_*
    are reported but do not fail a run for pass^N."""
    return all(s.passed for s in scores if s.critical or s.metric in PASS_POW_N_METRICS)


def percentile(values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def _accuracy(results: Sequence[RunResult]) -> tuple[dict[str, MetricAccuracy], list[str]]:
    values: dict[str, list[float]] = defaultdict(list)
    seen: list[str] = []
    for r in results:
        for s in r.scores:
            key = metric_key(s)
            if key not in seen:
                seen.append(key)
            if s.counts_toward_accuracy:
                values[key].append(s.value)
    accuracy = {
        k: MetricAccuracy(
            value=sum(values[k]) / len(values[k]),
            n=len(values[k]),
            lower_is_better=k.split(".", 1)[1] in LOWER_IS_BETTER,
        )
        for k in seen
        if values[k]
    }
    return accuracy, [k for k in seen if not values[k]]


def _by_case(results: Sequence[RunResult]) -> dict[str, list[RunResult]]:
    cases: dict[str, list[RunResult]] = {}
    for r in results:
        cases.setdefault(r.case_id, []).append(r)
    return cases


def _agreement(cases: dict[str, list[RunResult]]) -> float | None:
    matched = counted = 0
    for runs in cases.values():
        clean = [r for r in runs if not (r.infra_failed or r.data_failed)]
        if not clean:
            continue
        sets = Counter(
            tuple(r.exception_set) if r.exception_set is not None else NO_OUTPUT for r in clean
        )
        matched += max(sets.values())
        counted += len(clean)
    return matched / counted if counted else None


def summarize_config(model: ModelConfig, results: Sequence[RunResult]) -> ConfigSummary:
    if not results:
        raise ValueError(f"{model.config_id}: no run results to summarize")
    accuracy, unscored = _accuracy(results)
    cases = _by_case(results)
    pass_pow_n: dict[str, bool] = {}
    excluded: list[str] = []
    for case_id, runs in cases.items():
        if any(r.infra_failed or r.data_failed for r in runs):
            excluded.append(case_id)
        else:
            pass_pow_n[case_id] = all(run_passes(r.scores) for r in runs)
    clean = [r for r in results if not (r.infra_failed or r.data_failed)]
    papers = [r for r in clean if r.has_workpaper]
    latencies = [r.latency_ms for r in papers]
    infra = sum(r.infra_failed or r.judge_infra_failed for r in results)
    data = sum(r.data_failed for r in results)
    return ConfigSummary(
        config_id=model.config_id,
        model_id=model.model_id,
        provider=model.provider,
        prompt_version=model.prompt_version,
        simulated=model.simulated,
        runs=len(results),
        accuracy=accuracy,
        unscored_metrics=unscored,
        pass_pow_n=pass_pow_n,
        pass_pow_n_excluded=excluded,
        pass_pow_n_share=sum(pass_pow_n.values()) / len(pass_pow_n) if pass_pow_n else None,
        outcome_agreement=_agreement(cases),
        latency_p50_ms=percentile(latencies, 0.50) if latencies else None,
        latency_p95_ms=percentile(latencies, 0.95) if latencies else None,
        mean_cost_micros_per_workpaper=mean_micros([r.cost_micros for r in papers]),
        judge_mean_cost_micros_per_workpaper=mean_micros([r.judge_cost_micros for r in papers]),
        missed_by_final_only=sum(r.missed_by_final_only for r in results),
        infra_failed_runs=infra,
        data_failed_runs=data,
        infra_failure_rate=infra / len(results),
        data_failure_rate=data / len(results),
    )


def saturated_metrics(configs: Sequence[ConfigSummary]) -> list[str]:
    """Metrics every config scored at >= 0.98 (<= 0.02 when lower is better)."""
    if not configs:
        return []
    common = [k for k in configs[0].accuracy if all(k in c.accuracy for c in configs)]

    def high(a: MetricAccuracy) -> bool:
        return (1.0 - a.value if a.lower_is_better else a.value) >= SATURATION

    return [k for k in common if all(high(c.accuracy[k]) for c in configs)]
