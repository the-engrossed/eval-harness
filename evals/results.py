"""Runner record types (SPEC.md §10): one RunResult per agent run, summaries, manifest.

Pure: building a RunResult from a finished state and its grade does no I/O.
"""

from __future__ import annotations

import datetime as dt
from fractions import Fraction

from pydantic import Field

from auditlab.agent.state import AgentState
from auditlab.domain.base import DomainModel, Id, Text
from auditlab.domain.enums import FailureOrigin, Node
from auditlab.domain.scores import StepScore
from auditlab.models.config import ModelConfig, Price
from auditlab.money import cost_micros
from evals.scorers import RunGrade


class RunResult(DomainModel):
    """One line of results.jsonl. latency_ms sums model-reported call latencies only."""

    model_config_id: Id
    case_id: Id
    run_index: int = Field(ge=0)
    scores: list[StepScore]
    shadow_pass: bool
    missed_by_final_only: bool
    exception_set: list[Text] | None
    has_workpaper: bool
    latency_ms: int = Field(ge=0)
    cost_micros: int = Field(ge=0)
    judge_cost_micros: int = Field(ge=0)
    infra_failed: bool
    data_failed: bool
    # The judge failed (infra). Counted in the infra failure rate, but the run's agent
    # scores, pass^N and planted-miss checks still use it: a judge outage never hides one.
    judge_infra_failed: bool = False
    model_calls: int = Field(ge=0)
    cache_hits: int = Field(ge=0)


class MetricAccuracy(DomainModel):
    value: float = Field(ge=0.0, le=1.0)
    n: int = Field(ge=1)
    lower_is_better: bool = False


class ConfigSummary(DomainModel):
    """Per model config. A None value means no run was eligible for that number."""

    config_id: Id
    model_id: Id
    provider: Id
    prompt_version: Id
    simulated: bool
    runs: int = Field(ge=1)
    accuracy: dict[Id, MetricAccuracy]
    unscored_metrics: list[Id]
    pass_pow_n: dict[Id, bool]
    pass_pow_n_excluded: list[Id]
    pass_pow_n_share: float | None
    outcome_agreement: float | None
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    mean_cost_micros_per_workpaper: int | None
    judge_mean_cost_micros_per_workpaper: int | None
    missed_by_final_only: int = Field(ge=0)
    infra_failed_runs: int = Field(ge=0)
    data_failed_runs: int = Field(ge=0)
    infra_failure_rate: float = Field(ge=0.0, le=1.0)
    data_failure_rate: float = Field(ge=0.0, le=1.0)


class Summary(DomainModel):
    """summary.json. Holds no run id, timestamp or trace ref, so equal runs give equal bytes."""

    suite: Id
    suite_hash: Id
    n: int = Field(ge=2)
    case_ids: list[Id]
    judge_id: Id
    judge_simulated: bool
    configs: list[ConfigSummary]
    saturation_assessed: bool
    saturated_metrics: list[Id]


class SuiteRef(DomainModel):
    name: Id
    hash: Id
    case_ids: list[Id]


class JudgeRef(DomainModel):
    config_id: Id
    judge_id: Id
    prompt_version: Id
    simulated: bool


class CacheStats(DomainModel):
    enabled: bool
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)


class Manifest(DomainModel):
    run_id: Id
    created_at: dt.datetime
    git_sha: Id
    git_dirty: bool
    seed: int
    n: int = Field(ge=2)
    workers: int = Field(ge=1)
    suite: SuiteRef
    production_config_id: Id
    models: list[ModelConfig]
    prompt_versions: dict[Id, Id]
    judge: JudgeRef
    cache: CacheStats


def mean_micros(values: list[int]) -> int | None:
    return round(Fraction(sum(values), len(values))) if values else None


def price_micros(price: Price | None, simulated: bool, tokens_in: int, tokens_out: int) -> int:
    if price is not None:
        return cost_micros(
            tokens_in,
            tokens_out,
            input_micros_per_million=price.input_micros_per_mtok,
            output_micros_per_million=price.output_micros_per_mtok,
        )
    if simulated:
        return 0
    raise ValueError("a real judge needs a [[price]] entry in config/models.toml")


def exception_signature(state: AgentState) -> list[str] | None:
    if state.exceptions is None:
        return None
    flagged = [f"{e.item_id}:{e.type.value}" for e in state.exceptions.exceptions]
    judged = [f"{j.item_id}:{j.type.value}:{j.outcome.value}" for j in state.exceptions.judgments]
    return sorted(flagged + judged)


def run_result(state: AgentState, grade: RunGrade, judge_price: Price | None) -> RunResult:
    origins = {r.error.origin for r in state.step_records if r.error is not None}
    model_records = [r for r in state.step_records if r.node is not Node.TIE_OUT]
    judge_cost = 0
    if grade.judge is not None:
        j = grade.judge
        judge_cost = price_micros(judge_price, j.simulated, j.tokens_in, j.tokens_out)
    return RunResult(
        model_config_id=state.model_config_id,
        case_id=state.case_id,
        run_index=state.run_index,
        scores=grade.scores,
        shadow_pass=grade.shadow_pass,
        missed_by_final_only=grade.missed_by_final_only,
        exception_set=exception_signature(state),
        has_workpaper=state.workpaper is not None,
        latency_ms=sum(r.latency_ms for r in model_records if r.error is None),
        cost_micros=sum(r.cost_micros for r in state.step_records),
        judge_cost_micros=judge_cost,
        infra_failed=FailureOrigin.INFRA in origins,
        data_failed=FailureOrigin.DATA in origins,
        judge_infra_failed=grade.judge_error is not None,
        model_calls=len(model_records),
        cache_hits=sum(r.cached for r in model_records),
    )
