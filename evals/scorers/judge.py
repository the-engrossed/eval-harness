"""Workpaper judge interface (SPEC.md §9.5, §9.7). Never receives the answer key."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Self

from pydantic import Field, model_validator

from auditlab.agent.state import AgentState
from auditlab.domain.base import DomainModel, Id, Text
from auditlab.domain.enums import Node
from auditlab.domain.models import EvidenceSet, ExceptionSet, TieOutSet
from auditlab.models.base import ModelClient, ModelRequest
from auditlab.models.cache import CachedClient, ResponseCache
from auditlab.models.config import ModelsConfig
from auditlab.models.registry import build_client

PROMPTS_DIR = Path(__file__).resolve().parent / "judge_prompts"
STUB_JUDGE_ID = "stub-judge"
JUDGE_PROMPT_VERSION = "v1"
JUDGE_ENV = "AUDITLAB_JUDGE_CONFIG"


class JudgeCriterion(StrEnum):
    PROCEDURES_DESCRIBED = "procedures_described"
    SAMPLES_IDENTIFIED = "samples_identified"
    EXCEPTIONS_EXPLAINED = "exceptions_explained"
    CONCLUSION_SUPPORTED = "conclusion_supported"
    NO_UNSUPPORTED_CLAIMS = "no_unsupported_claims"


CRITERIA: tuple[JudgeCriterion, ...] = tuple(JudgeCriterion)


class CriterionVerdict(DomainModel):
    criterion: JudgeCriterion
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: Text


class JudgeVerdict(DomainModel):
    verdicts: list[CriterionVerdict] = Field(min_length=len(CRITERIA), max_length=len(CRITERIA))

    @model_validator(mode="after")
    def _each_criterion_once(self) -> Self:
        got = [v.criterion for v in self.verdicts]
        if set(got) != set(CRITERIA) or len(got) != len(CRITERIA):
            raise ValueError(
                f"JudgeVerdict must cover each criterion once; got {[c.value for c in got]}"
            )
        return self


class JudgeInput(DomainModel):
    """What a judge sees: the deliverable and the agent's own outputs. No answer key."""

    prose: Text
    item_ids: list[Id]
    evidence: EvidenceSet
    tieouts: TieOutSet
    exceptions: ExceptionSet

    @classmethod
    def from_state(cls, state: AgentState) -> JudgeInput:
        if (
            state.workpaper is None
            or state.samples is None
            or state.evidence is None
            or state.tieouts is None
            or state.exceptions is None
        ):
            raise ValueError(
                "JudgeInput needs workpaper, samples, evidence, tieouts and exceptions"
            )
        return cls(
            prose=state.workpaper.prose,
            item_ids=list(state.samples.item_ids),
            evidence=state.evidence,
            tieouts=state.tieouts,
            exceptions=state.exceptions,
        )


class JudgeResult(DomainModel):
    verdict: JudgeVerdict
    judge_id: Id
    simulated: bool
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class Judge(Protocol):
    def grade(self, inp: JudgeInput) -> JudgeResult: ...


def load_judge_prompt(version: str = JUDGE_PROMPT_VERSION) -> str:
    path = PROMPTS_DIR / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"judge prompt {path} not found")
    return path.read_text(encoding="utf-8")


class ModelJudge:
    """Wraps any ModelClient; system prompt is judge_prompts/<version>.md."""

    def __init__(self, client: ModelClient, *, prompt_version: str = JUDGE_PROMPT_VERSION) -> None:
        self._client = client
        self._prompt_version = prompt_version
        self._system = load_judge_prompt(prompt_version)

    def grade(self, inp: JudgeInput) -> JudgeResult:
        request = ModelRequest(
            case_id="jg-run",
            node=Node.DRAFT_WORKPAPER,
            prompt_version=f"judge-{self._prompt_version}",
            system=self._system,
            payload=inp.model_dump(mode="json"),
            output_schema=JudgeVerdict,
            run_index=0,
            seed=0,
        )
        response = self._client.complete(request)
        verdict = response.output
        if not isinstance(verdict, JudgeVerdict):
            raise TypeError(f"judge returned {type(verdict).__name__}, expected JudgeVerdict")
        return JudgeResult(
            verdict=verdict,
            judge_id=f"{self._client.model_id}@judge-{self._prompt_version}",
            simulated=response.simulated,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            latency_ms=response.latency_ms,
        )


def judge_config_id(env: Mapping[str, str]) -> str:
    """The judge's config_id from AUDITLAB_JUDGE_CONFIG; stub-judge when unset."""
    raw = env.get(JUDGE_ENV, os.environ.get(JUDGE_ENV, "")).strip()
    return raw or STUB_JUDGE_ID


def build_judge(
    env: Mapping[str, str],
    models: ModelsConfig,
    *,
    model_config_ids: Sequence[str],
    cache_dir: Path | None = None,
) -> Judge:
    """Build the judge for a run. Checks every model under test up front (for T12).

    A real judge is wrapped in the §6 response cache at `cache_dir` (the runner's), so a
    recording stores judge responses next to the model's and a replay judge can serve them.
    """
    from evals.scorers.stub_judge import StubJudge

    config_id = judge_config_id(env)
    if config_id == STUB_JUDGE_ID:
        return StubJudge()
    overlap = [m for m in model_config_ids if m == config_id]
    if overlap:
        raise ValueError(
            f"judge config {config_id!r} is also a model under test "
            f"({', '.join(overlap)}). The judge must be configured separately "
            f"(set {JUDGE_ENV} to stub-judge or another config_id)."
        )
    try:
        models.model(config_id)
    except ValueError as exc:
        raise ValueError(
            f"unknown judge config {config_id!r} in {JUDGE_ENV}; "
            f"configured: {[m.config_id for m in models.models]}"
        ) from exc
    client = build_client(config_id, models)
    if cache_dir is not None and models.model(config_id).provider != "replay":
        client = CachedClient(client, ResponseCache(cache_dir))
    return ModelJudge(client)
