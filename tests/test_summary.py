"""Pure runner aggregation: pass^N, agreement, percentiles, saturation (SPEC.md §10, §10.1)."""

import pytest

from auditlab.domain.enums import FailureOrigin, MetricKind, Node
from auditlab.domain.scores import StepScore
from auditlab.models.config import ModelConfig
from evals.results import ConfigSummary, MetricAccuracy, RunResult
from evals.summary import percentile, run_passes, saturated_metrics, summarize_config

MODEL = ModelConfig(
    config_id="prod", provider="stub", model_id="stub-prod", prompt_version="v1", simulated=True
)


def score(
    metric: str, value: float, *, critical: bool, node: Node = Node.SELECT_SAMPLES, run: int = 0
) -> StepScore:
    passed = value == 1.0
    return StepScore(
        case_id="c1",
        run_index=run,
        model_config_id="prod",
        node=node,
        metric=metric,
        kind=MetricKind.PROCESS,
        value=value,
        passed=passed,
        critical=critical,
        failure_origin=None if passed else FailureOrigin.AGENT,
        detail="d",
        trace_ref="local:t/1",
    )


def result(
    case_id: str,
    run_index: int,
    scores: list[StepScore],
    *,
    exceptions: list[str] | None = None,
    latency: int = 100,
    infra: bool = False,
) -> RunResult:
    return RunResult(
        model_config_id="prod",
        case_id=case_id,
        run_index=run_index,
        scores=scores,
        shadow_pass=True,
        missed_by_final_only=False,
        exception_set=exceptions if exceptions is not None else ["INV-1:CUTOFF_ERROR"],
        has_workpaper=True,
        latency_ms=latency,
        cost_micros=10,
        judge_cost_micros=0,
        infra_failed=infra,
        data_failed=False,
        model_calls=4,
        cache_hits=0,
    )


def test_precision_below_one_does_not_fail_a_run_for_pass_pow_n() -> None:
    scores = [
        score("required_recall", 1.0, critical=True),
        score("precision", 0.9, critical=False),
        score("judge_samples_identified", 0.0, critical=False, node=Node.DRAFT_WORKPAPER),
    ]
    assert run_passes(scores)


def test_critical_or_judgment_allowed_failure_fails_a_run() -> None:
    assert not run_passes([score("required_recall", 0.9, critical=True)])
    judgment = score("judgment_allowed", 0.5, critical=False, node=Node.FLAG_EXCEPTIONS)
    assert not run_passes([score("required_recall", 1.0, critical=True), judgment])


def test_percentile_is_nearest_rank() -> None:
    assert percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert percentile([5, 1, 3, 2, 4], 0.95) == 5
    assert percentile([7], 0.95) == 7
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_summary_pass_pow_n_agreement_and_infra_exclusion() -> None:
    ok = [score("required_recall", 1.0, critical=True)]
    bad = [score("required_recall", 0.5, critical=True)]
    results = [
        result("c1", 0, ok, latency=100),
        result("c1", 1, ok, latency=300),
        result("c2", 0, ok, exceptions=["INV-2:CUTOFF_ERROR"]),
        result("c2", 1, bad, exceptions=[]),
        result("c3", 0, ok),
        result("c3", 1, [], infra=True, exceptions=[]),
    ]
    s = summarize_config(MODEL, results)
    assert s.pass_pow_n == {"c1": True, "c2": False}
    assert s.pass_pow_n_excluded == ["c3"]
    assert s.pass_pow_n_share == 0.5
    assert s.outcome_agreement == pytest.approx(4 / 5)
    assert s.infra_failure_rate == pytest.approx(1 / 6)
    recall = s.accuracy["select_samples.required_recall"]
    assert (recall.value, recall.n) == (pytest.approx(0.9), 5)
    assert (s.latency_p50_ms, s.latency_p95_ms) == (100, 300)
    assert s.mean_cost_micros_per_workpaper == 10


def _config(config_id: str, accuracy: dict[str, float]) -> ConfigSummary:
    base = summarize_config(
        MODEL, [result("c1", 0, [score("required_recall", 1.0, critical=True)])]
    )
    return base.model_copy(
        update={
            "config_id": config_id,
            "accuracy": {m: MetricAccuracy(value=v, n=4) for m, v in accuracy.items()},
        }
    )


def test_saturated_when_every_config_scores_at_least_098() -> None:
    a = _config("a", {"x.m1": 1.0, "x.m2": 0.99, "x.m3": 1.0})
    b = _config("b", {"x.m1": 0.98, "x.m2": 0.97})
    assert saturated_metrics([a, b]) == ["x.m1"]
