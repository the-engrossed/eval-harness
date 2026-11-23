"""Gate types (SPEC.md §11): config/gate.toml, the per-config gate input, findings, result.

Types and parsing only; the decision logic is in evals/gate.py.
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from auditlab.domain.base import DomainModel, Id, Text
from auditlab.domain.enums import Node, Verdict


class ConsistencyConfig(DomainModel):
    min_pass_pow_n: float = Field(ge=0.0, le=1.0)
    min_outcome_agreement: float = Field(default=0.9, ge=0.0, le=1.0)


class StatsConfig(DomainModel):
    bootstrap_resamples: int = Field(ge=1, strict=True)
    bootstrap_seed: int = Field(strict=True)
    confidence: float = Field(gt=0.0, lt=1.0)
    max_infra_failure_rate: float = Field(ge=0.0, le=1.0)


class JudgeGateConfig(DomainModel):
    """Read by the judge escalation (SPEC.md §9.7), not by the gate."""

    escalate_below: float = Field(ge=0.0, le=1.0)
    strong_config: str = ""


class GateConfig(DomainModel):
    default_margin: float = Field(ge=0.0, le=1.0)
    critical: dict[Id, float]
    consistency: ConsistencyConfig
    stats: StatsConfig
    judge: JudgeGateConfig

    @model_validator(mode="after")
    def _critical_keys_are_node_metrics(self) -> Self:
        nodes = {n.value for n in Node}
        for key, margin in self.critical.items():
            node, _, metric = key.partition(".")
            if node not in nodes or not metric:
                raise ValueError(f"[critical] {key!r} is not <node>.<metric>")
            if not 0.0 <= margin <= 1.0:
                raise ValueError(f"[critical] {key} margin {margin} is outside [0, 1]")
        return self


def load_gate_config(path: Path) -> GateConfig:
    with path.open("rb") as handle:
        return GateConfig.model_validate(tomllib.load(handle))


class Failure(DomainModel):
    """One counted failing score (or planted miss) of the config this input describes."""

    case_id: Id
    run_index: int = Field(ge=0)
    trace_ref: Id
    detail: Text


class GateInput(DomainModel):
    """One config of one run, as the gate sees it. Also the evals/baselines/*.json format."""

    source_run_id: Id
    config_id: Id
    simulated: bool
    suite_hash: Id
    case_means: dict[Id, dict[Id, float]]  # metric key -> case_id -> mean over counted runs
    failures: dict[Id, list[Failure]]  # metric key -> counted failing scores
    planted_misses: list[Failure]  # planted_recall not passed in a run with no infra/data fault
    feedback_misses: list[Failure] = []  # feedback_expectation_met not passed, same rule
    feedback_runs: int = Field(default=0, ge=0)  # runs with no infra/data fault scoring it
    pass_pow_n_share: float | None
    outcome_agreement: float | None
    infra_failure_rate: float = Field(ge=0.0, le=1.0)


class Check(StrEnum):
    PLANTED_MISS = "planted_miss"
    FEEDBACK_MISS = "feedback_miss"
    CRITICAL_DROP = "critical_drop"
    NONCRITICAL_DROP = "noncritical_drop"
    CONSISTENCY = "consistency"
    INFRA = "infra"


class RootStep(DomainModel):
    metric: Id
    baseline_value: float
    candidate_value: float
    detail: Text


class Finding(DomainModel):
    """verdict SHIP means a note: listed, but it does not move the verdict."""

    check: Check
    verdict: Verdict
    message: Text
    metric: Id | None = None
    measure: Id | None = None  # consistency/infra: pass_pow_n_share, outcome_agreement, ...
    baseline_value: float | None = None
    candidate_value: float | None = None
    mean_delta: float | None = None
    ci: tuple[float, float] | None = None
    case_ids: list[Id] = []
    trace_refs: list[Id] = []
    root: RootStep | None = None


class GateResult(DomainModel):
    candidate_config_id: Id
    baseline_config_id: Id
    baseline_source: Id
    simulated: bool
    verdict: Verdict
    findings: list[Finding]
