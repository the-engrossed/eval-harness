"""The run spec and the pure builders for summary.json, manifest.json and history.jsonl.

No I/O here: evals/runner.py runs the agent and writes what these functions build.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from auditlab.domain.suite import Suite
from auditlab.models.config import ModelsConfig, Price
from evals.feedback_cases import FeedbackCase
from evals.results import (
    CacheStats,
    ConfigSummary,
    JudgeRef,
    Manifest,
    RunResult,
    SuiteRef,
    Summary,
)
from evals.scorers.judge import JUDGE_PROMPT_VERSION, STUB_JUDGE_ID
from evals.scorers.stub_judge import JUDGE_ID as STUB_JUDGE_RESULT_ID
from evals.summary import saturated_metrics, summarize_config


@dataclass(frozen=True)
class EvalSpec:
    run_id: str
    suite: Suite
    suite_hash: str
    n: int
    seed: int
    workers: int
    runs_dir: Path
    engagements_dir: Path
    cache_dir: Path | None
    models: ModelsConfig
    config_ids: list[str]
    judge_config_id: str
    judge_price: Price | None
    feedback: list[FeedbackCase] = field(default_factory=list)

    @property
    def case_ids(self) -> list[str]:
        """Suite cases, then feedback cases (SPEC.md §12)."""
        return [c.case_id for c in self.suite.cases] + [f.case_id for f in self.feedback]


def price_for(models: ModelsConfig, config_id: str) -> Price | None:
    model_id = models.model(config_id).model_id
    return next((p for p in models.prices if p.model_id == model_id), None)


def judge_ref(spec: EvalSpec) -> JudgeRef:
    if spec.judge_config_id == STUB_JUDGE_ID:
        return JudgeRef(
            config_id=STUB_JUDGE_ID,
            judge_id=STUB_JUDGE_RESULT_ID,
            prompt_version=JUDGE_PROMPT_VERSION,
            simulated=True,
        )
    model = spec.models.model(spec.judge_config_id)
    return JudgeRef(
        config_id=model.config_id,
        judge_id=f"{model.model_id}@judge-{JUDGE_PROMPT_VERSION}",
        prompt_version=JUDGE_PROMPT_VERSION,
        simulated=model.simulated,
    )


def build_summary(spec: EvalSpec, results: list[RunResult]) -> Summary:
    configs = [
        summarize_config(spec.models.model(c), [r for r in results if r.model_config_id == c])
        for c in spec.config_ids
    ]
    judge = judge_ref(spec)
    assessed = len(configs) >= 2  # saturation only means something across configs
    return Summary(
        suite=spec.suite.name,
        suite_hash=spec.suite_hash,
        n=spec.n,
        case_ids=spec.case_ids,
        judge_id=judge.judge_id,
        judge_simulated=judge.simulated,
        configs=configs,
        saturation_assessed=assessed,
        saturated_metrics=saturated_metrics(configs) if assessed else [],
    )


def build_manifest(
    spec: EvalSpec,
    results: list[RunResult],
    *,
    created_at: dt.datetime,
    git_sha: str,
    git_dirty: bool,
) -> Manifest:
    hits = sum(r.cache_hits for r in results)
    return Manifest(
        run_id=spec.run_id,
        created_at=created_at,
        git_sha=git_sha,
        git_dirty=git_dirty,
        seed=spec.seed,
        n=spec.n,
        workers=spec.workers,
        suite=SuiteRef(
            name=spec.suite.name,
            hash=spec.suite_hash,
            case_ids=spec.case_ids,
        ),
        production_config_id=spec.models.production_config_id,
        models=[spec.models.model(c) for c in spec.config_ids],
        prompt_versions={c: spec.models.model(c).prompt_version for c in spec.config_ids},
        judge=judge_ref(spec),
        cache=CacheStats(
            enabled=spec.cache_dir is not None,
            hits=hits,
            misses=sum(r.model_calls for r in results) - hits,
        ),
    )


def _trend(c: ConfigSummary) -> dict[str, Any]:
    return {
        "simulated": c.simulated,
        "accuracy": {k: a.value for k, a in c.accuracy.items()},
        "pass_pow_n_share": c.pass_pow_n_share,
        "mean_cost_micros_per_workpaper": c.mean_cost_micros_per_workpaper,
    }


def history_line(manifest: Manifest, summary: Summary) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "created_at": manifest.created_at.isoformat(),
        "git_sha": manifest.git_sha,
        "suite": summary.suite,
        "suite_hash": summary.suite_hash,
        "n": summary.n,
        "configs": {c.config_id: _trend(c) for c in summary.configs},
    }
