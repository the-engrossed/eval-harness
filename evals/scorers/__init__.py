"""Deterministic step scorers plus the workpaper judge (SPEC.md §9)."""

from __future__ import annotations

from pydantic import Field

from auditlab.agent.state import AgentState
from auditlab.domain.answer_key import AnswerKey
from auditlab.domain.base import DomainModel
from auditlab.domain.scores import StepScore
from evals.scorers.common import Pages
from evals.scorers.exceptions import FALSE_POSITIVE_RATE, score_exceptions
from evals.scorers.extract import score_extract
from evals.scorers.judge import Judge, JudgeResult
from evals.scorers.select import score_select
from evals.scorers.tieout import score_tieout
from evals.scorers.workpaper import score_workpaper

LOWER_IS_BETTER = frozenset({FALSE_POSITIVE_RATE.name})


class RunGrade(DomainModel):
    """Every §9 score for one run, plus the §9.9 shadow grade."""

    scores: list[StepScore] = Field(min_length=1)
    shadow_pass: bool
    missed_by_final_only: bool
    judge: JudgeResult | None
    judge_error: str | None = None  # the judge failed (infra); judge_* scores say so


def score_run(state: AgentState, key: AnswerKey, pages: Pages, judge: Judge) -> RunGrade:
    scores = [
        *score_select(state, key),
        *score_extract(state, key, pages),
        *score_tieout(state, key),
        *score_exceptions(state, key),
    ]
    workpaper_scores, judge_result, judge_error = score_workpaper(state, judge)
    scores = [*scores, *workpaper_scores]
    shadow_scores = [
        s for s in workpaper_scores if s.metric == "guard_pass" or s.metric.startswith("judge_")
    ]
    shadow_pass = bool(shadow_scores) and all(s.passed and not s.skipped for s in shadow_scores)
    critical_failed = any(s.critical and not s.passed and not s.skipped for s in scores)
    return RunGrade(
        scores=scores,
        shadow_pass=shadow_pass,
        missed_by_final_only=shadow_pass and critical_failed,
        judge=judge_result,
        judge_error=judge_error,
    )
