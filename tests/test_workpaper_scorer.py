from pathlib import Path

import pytest
from cases import CASE_D
from scoring_fixtures import KEY, PAGES, by_metric, oracle, record, with_workpaper

from auditlab.domain.enums import FailureOrigin, Node
from auditlab.domain.models import (
    GuardFinding,
    StepError,
    Workpaper,
)
from auditlab.guard import GuardCode, validate_prose
from auditlab.models.base import ModelRequest, ModelResponse
from auditlab.models.config import ModelConfig, ModelsConfig
from data.sampling import cutoff_window
from evals.scorers import score_run
from evals.scorers.judge import (
    CRITERIA,
    CriterionVerdict,
    JudgeCriterion,
    JudgeInput,
    JudgeVerdict,
    ModelJudge,
    build_judge,
)
from evals.scorers.stub_judge import StubJudge

REPO = Path(__file__).resolve().parent.parent
JUDGE_MODULES = [
    REPO / "evals/scorers/judge.py",
    REPO / "evals/scorers/stub_judge.py",
    REPO / "evals/scorers/workpaper.py",
]


def grade(state=None):  # type: ignore[no-untyped-def]
    return score_run(with_workpaper(state or oracle()), KEY, PAGES, StubJudge())


def test_stub_prod_workpaper_passes_guard_and_all_criteria() -> None:
    result = grade()
    scores = by_metric(result.scores)
    assert scores["guard_pass"].value == 1.0 and scores["guard_pass"].critical
    for c in CRITERIA:
        assert scores[f"judge_{c.value}"].passed
        assert scores[f"judge_{c.value}"].kind.value == "outcome"
        assert not scores[f"judge_{c.value}"].critical
    assert result.shadow_pass and not result.missed_by_final_only
    assert result.judge is not None and result.judge.simulated


def test_a_workpaper_with_a_bare_amount_fails_guard_pass() -> None:
    state = with_workpaper(oracle())
    assert state.workpaper is not None and state.evidence is not None and state.samples is not None
    prose = state.workpaper.prose + " Amount due: $12,400."
    violations = validate_prose(
        prose,
        frozenset(f.evidence_id for f in state.evidence.fields),
        item_ids=frozenset(state.samples.item_ids),
    )
    assert any(v.code is GuardCode.UNBOUND_NUMERAL for v in violations)
    state.workpaper = Workpaper(
        prose=prose,
        guard_violations=[GuardFinding(code=v.code, message=v.message) for v in violations],
    )
    result = score_run(state, KEY, PAGES, StubJudge())
    score = by_metric(result.scores)["guard_pass"]
    assert score.value == 0.0 and not score.passed
    assert score.failure_origin is FailureOrigin.AGENT
    assert GuardCode.UNBOUND_NUMERAL.value in score.detail
    assert not result.shadow_pass


def test_a_cutoff_dropped_run_passes_the_shadow_grade_but_is_missed_by_final_only() -> None:
    window = set(cutoff_window(CASE_D.population))
    keep = set(KEY.required_sample_ids) - window
    result = grade(oracle(keep=keep))
    scores = by_metric(result.scores)
    assert scores["required_recall"].value < 1.0 and scores["planted_recall"].value < 1.0
    assert scores["guard_pass"].passed
    assert all(scores[f"judge_{c.value}"].passed for c in CRITERIA)
    assert result.shadow_pass and result.missed_by_final_only


def test_each_criterion_can_fail_on_its_own() -> None:
    state = with_workpaper(oracle())
    judge = StubJudge()
    base = JudgeInput.from_state(state)
    assert state.exceptions is not None and state.exceptions.exceptions
    [first_exc] = state.exceptions.exceptions[:1]

    no_procedures = base.model_copy(
        update={"prose": "Samples tested: [[item:INV-1]]. Conclusion: ok."}
    )
    assert not _verdict(judge, no_procedures, JudgeCriterion.PROCEDURES_DESCRIBED).passed

    stripped = base.prose
    for iid in base.item_ids:
        stripped = stripped.replace(f"[[item:{iid}]]", "an item")
    assert not _verdict(
        judge, base.model_copy(update={"prose": stripped}), JudgeCriterion.SAMPLES_IDENTIFIED
    ).passed

    silent = base.model_copy(
        update={
            "prose": base.prose.replace(first_exc.type.value.replace("_", " ").lower(), "matter")
        }
    )
    assert not _verdict(judge, silent, JudgeCriterion.EXCEPTIONS_EXPLAINED).passed

    no_conclusion = base.model_copy(update={"prose": base.prose.replace("Conclusion:", "Summary:")})
    assert not _verdict(judge, no_conclusion, JudgeCriterion.CONCLUSION_SUPPORTED).passed

    invented = base.model_copy(
        update={
            "prose": base.prose + " We also noted amount mismatch on [[item:INV-9999]].",
            "item_ids": [*base.item_ids, "INV-9999"],
        }
    )
    assert not _verdict(judge, invented, JudgeCriterion.NO_UNSUPPORTED_CLAIMS).passed


def _verdict(judge: StubJudge, inp: JudgeInput, criterion: JudgeCriterion) -> CriterionVerdict:
    result = judge.grade(inp)
    return next(v for v in result.verdict.verdicts if v.criterion is criterion)


def test_the_judge_returns_confidence_in_zero_to_one() -> None:
    result = StubJudge().grade(JudgeInput.from_state(with_workpaper(oracle())))
    for v in result.verdict.verdicts:
        assert 0.0 <= v.confidence <= 1.0
    with pytest.raises(ValueError):
        CriterionVerdict(
            criterion=JudgeCriterion.PROCEDURES_DESCRIBED,
            passed=True,
            confidence=1.5,
            reason="x",
        )
    dup = CriterionVerdict(
        criterion=JudgeCriterion.PROCEDURES_DESCRIBED,
        passed=True,
        confidence=1.0,
        reason="ok",
    )
    with pytest.raises(ValueError, match="each criterion"):
        JudgeVerdict(verdicts=[dup] * len(CRITERIA))


def test_the_stub_judge_never_reads_the_answer_key() -> None:
    offenders = [
        str(p.relative_to(REPO))
        for p in JUDGE_MODULES
        if "answer_key" in p.read_text(encoding="utf-8")
        or "AnswerKey" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []
    assert "answer_key" not in JudgeInput.model_fields
    # grade takes only JudgeInput — call signature is the proof
    StubJudge().grade(JudgeInput.from_state(with_workpaper(oracle())))


def test_a_failed_workpaper_node_skips_the_judge() -> None:
    state = oracle()
    state.workpaper = None
    error = StepError(origin=FailureOrigin.INFRA, message="timed out")
    state.step_records = [
        r if r.node is not Node.DRAFT_WORKPAPER else record(Node.DRAFT_WORKPAPER, error)
        for r in state.step_records
    ]
    called: list[int] = []

    class Spy(StubJudge):
        def grade(self, inp: JudgeInput):  # type: ignore[no-untyped-def]
            called.append(1)
            return super().grade(inp)

    result = score_run(state, KEY, PAGES, Spy())
    assert called == []
    work = [s for s in result.scores if s.node is Node.DRAFT_WORKPAPER]
    assert work and all(s.failure_origin is FailureOrigin.INFRA for s in work)
    assert not result.shadow_pass and result.judge is None


class FixedClient:
    provider = "stub"
    model_id = "fixed"
    simulated = True

    def __init__(self, verdict: JudgeVerdict) -> None:
        self.verdict = verdict
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            output=self.verdict,
            tokens_in=10,
            tokens_out=5,
            latency_ms=1,
            provider_model_id="fixed",
            cached=False,
            simulated=True,
        )


def _all_pass() -> JudgeVerdict:
    return JudgeVerdict(
        verdicts=[
            CriterionVerdict(criterion=c, passed=True, confidence=0.9, reason="ok")
            for c in CRITERIA
        ]
    )


def test_model_judge_sends_the_v1_prompt_and_no_case_id() -> None:
    client = FixedClient(_all_pass())
    judge = ModelJudge(client, prompt_version="v1")
    inp = JudgeInput.from_state(with_workpaper(oracle()))
    result = judge.grade(inp)
    assert result.verdict == _all_pass()
    assert result.judge_id == "fixed@judge-v1"
    [req] = client.requests
    assert "procedures_described" in req.system
    assert CASE_D.case_id not in req.system and CASE_D.case_id not in req.payload_json()
    assert req.output_schema is JudgeVerdict


def _config() -> ModelsConfig:
    return ModelsConfig(
        production_config_id="prod",
        models=[
            ModelConfig(
                config_id="prod",
                provider="stub",
                model_id="stub-prod",
                prompt_version="v1",
                simulated=True,
            ),
            ModelConfig(
                config_id="cand",
                provider="stub",
                model_id="stub-prod",
                prompt_version="v1",
                simulated=True,
            ),
        ],
        prices=[],
    )


def test_build_judge_defaults_to_stub_and_rejects_the_model_under_test() -> None:
    config = _config()
    stub = build_judge({}, config, model_config_ids=["prod", "cand"])
    assert isinstance(stub, StubJudge)
    named = build_judge({"AUDITLAB_JUDGE_CONFIG": "stub-judge"}, config, model_config_ids=["prod"])
    assert isinstance(named, StubJudge)
    with pytest.raises(ValueError, match="unknown"):
        build_judge({"AUDITLAB_JUDGE_CONFIG": "nope"}, config, model_config_ids=["prod"])
    with pytest.raises(ValueError, match="model under test"):
        build_judge(
            {"AUDITLAB_JUDGE_CONFIG": "cand"},
            config,
            model_config_ids=["prod", "cand"],
        )
    # fails before any model is built — even with only the colliding id in the list
    with pytest.raises(ValueError, match="cand"):
        build_judge({"AUDITLAB_JUDGE_CONFIG": "cand"}, config, model_config_ids=["cand"])


def test_a_real_judge_is_cached_and_a_replay_judge_is_not_wrapped(tmp_path: Path) -> None:
    from auditlab.models.cache import CachedClient
    from auditlab.models.replay import ReplayModel

    real = ModelConfig(
        config_id="judge-real",
        provider="openai_compat",
        model_id="judge-model",
        prompt_version="v1",
        simulated=False,
        base_url="http://localhost:1/v1",
    )
    replay = ModelConfig(
        config_id="judge-replay",
        provider="replay",
        replay_provider="openai_compat",
        model_id="judge-model",
        prompt_version="v1",
        simulated=False,
    )
    base = _config()
    config = base.model_copy(update={"models": [*base.models, real, replay]})
    env = {"AUDITLAB_JUDGE_CONFIG": "judge-real"}
    judge = build_judge(env, config, model_config_ids=["prod"], cache_dir=tmp_path)
    assert isinstance(judge, ModelJudge)
    assert isinstance(judge._client, CachedClient) and judge._client.cache.root == tmp_path
    uncached = build_judge(env, config, model_config_ids=["prod"], cache_dir=None)
    assert isinstance(uncached, ModelJudge) and not isinstance(uncached._client, CachedClient)
    env = {"AUDITLAB_JUDGE_CONFIG": "judge-replay"}
    replayed = build_judge(env, config, model_config_ids=["prod"], cache_dir=tmp_path)
    assert isinstance(replayed, ModelJudge) and isinstance(replayed._client, ReplayModel)
