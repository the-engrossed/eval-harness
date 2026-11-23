"""A judge HTTP or output error scores that run's judge_* as infra; the run continues."""

import pytest
from scoring_fixtures import KEY, PAGES, by_metric, oracle, with_workpaper

from auditlab.domain.enums import FailureOrigin, Node
from auditlab.models.config import ModelConfig
from auditlab.models.errors import (
    ModelOutputError,
    PromptTooLargeError,
    ProviderError,
    RecordedFailure,
)
from evals.results import run_result
from evals.scorers import score_run
from evals.scorers.judge import CRITERIA, JudgeInput, JudgeResult
from evals.summary import summarize_config

MODEL = ModelConfig(
    config_id="prod", provider="stub", model_id="stub-prod", prompt_version="v1", simulated=True
)


class FailingJudge:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def grade(self, inp: JudgeInput) -> JudgeResult:
        raise self.error


ERRORS = [
    ProviderError("openai_compat", "judge-model", Node.DRAFT_WORKPAPER, "HTTP 429: rate limit"),
    ModelOutputError("judge-model", Node.DRAFT_WORKPAPER, "{not json", "invalid JSON"),
    PromptTooLargeError("judge-model", Node.DRAFT_WORKPAPER, 2_000_000, 1_050_000),
    RecordedFailure(FailureOrigin.INFRA, "openai_compat judge-model at draft_workpaper: timed out"),
]


@pytest.mark.parametrize("error", ERRORS, ids=["http", "output", "too-large", "recorded"])
def test_judge_error_marks_only_judge_scores_infra(error: Exception) -> None:
    state = with_workpaper(oracle())
    grade = score_run(state, KEY, PAGES, FailingJudge(error))
    scores = by_metric(grade.scores)
    for c in CRITERIA:
        s = scores[f"judge_{c.value}"]
        assert not s.passed and s.failure_origin is FailureOrigin.INFRA
        assert not s.counts_toward_accuracy
        assert s.detail.startswith("judge failed (infra): ")
    assert scores["guard_pass"].passed
    assert scores["planted_recall"].passed and scores["planted_recall"].counts_toward_accuracy
    assert grade.judge is None and grade.judge_error is not None
    assert not grade.shadow_pass and not grade.missed_by_final_only

    result = run_result(state, grade, None)
    assert result.judge_infra_failed and not result.infra_failed
    assert result.judge_cost_micros == 0


def test_judge_infra_counts_in_the_infra_rate_but_keeps_the_run_in_pass_pow_n() -> None:
    state = with_workpaper(oracle())
    ok = run_result(state, score_run(state, KEY, PAGES, _stub()), None)
    failed = run_result(state, score_run(state, KEY, PAGES, FailingJudge(ERRORS[0])), None)
    failed = failed.model_copy(update={"run_index": 1})
    s = summarize_config(MODEL, [ok, failed])
    assert s.infra_failed_runs == 1
    assert s.infra_failure_rate == pytest.approx(0.5)
    assert s.pass_pow_n == {state.case_id: True}
    assert s.pass_pow_n_excluded == []


def _stub():  # type: ignore[no-untyped-def]
    from evals.scorers.stub_judge import StubJudge

    return StubJudge()
