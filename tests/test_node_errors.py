"""A provider or output failure at one node is a failed step, not a crashed run (§9.8)."""

from pathlib import Path

import pytest
from cases import CASE_D
from scoring_fixtures import PAGES

from auditlab.agent.graph import run_case
from auditlab.domain.enums import FailureOrigin, Node
from auditlab.models.base import ModelClient, ModelRequest, ModelResponse
from auditlab.models.errors import (
    ModelOutputError,
    PromptTooLargeError,
    ProviderError,
    RecordedFailure,
)
from auditlab.models.stub import StubModel
from evals.scorers import score_run
from evals.scorers.stub_judge import StubJudge


class FailAt:
    """Passes calls through, except it raises at one node."""

    def __init__(self, inner: ModelClient, node: Node, error: str) -> None:
        self.inner, self.node, self.error = inner, node, error
        self.provider, self.model_id, self.simulated = inner.provider, inner.model_id, True
        self.calls: list[Node] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request.node)
        if request.node is self.node:
            if self.error == "provider":
                raise ProviderError(self.provider, self.model_id, request.node, "HTTP 503")
            if self.error == "recorded":
                raise RecordedFailure(FailureOrigin.INFRA, "openai_compat m at x: timed out")
            if self.error == "too-large":
                raise PromptTooLargeError(self.model_id, request.node, 40_000, 32_768)
            raise ModelOutputError(self.model_id, request.node, "{not json", "invalid JSON")
        return self.inner.complete(request)


@pytest.mark.parametrize(
    ("error", "origin"),
    [
        ("provider", FailureOrigin.INFRA),
        ("output", FailureOrigin.AGENT),
        ("too-large", FailureOrigin.AGENT),
        ("recorded", FailureOrigin.INFRA),
    ],
)
def test_failed_node_records_error_and_later_nodes_return_immediately(
    engagements: Path, tmp_path: Path, error: str, origin: FailureOrigin
) -> None:
    client = FailAt(StubModel("stub-prod", engagements), Node.EXTRACT_EVIDENCE, error)
    state = run_case(
        CASE_D.case_id,
        client=client,
        model_config_id="prod",
        engagements_dir=engagements,
        runs_dir=tmp_path,
        run_id=f"err-{error}",
    )
    assert [r.node for r in state.step_records] == [Node.SELECT_SAMPLES, Node.EXTRACT_EVIDENCE]
    failed = state.step_records[1]
    assert failed.error is not None and failed.error.origin is origin
    if error == "too-large":
        assert "prompt too large: 40000 tokens" in failed.error.message
    if error == "recorded":
        assert failed.error.message == "openai_compat m at x: timed out"
    assert (failed.tokens_in, failed.tokens_out, failed.cost_micros) == (0, 0, 0)
    assert client.calls == [Node.SELECT_SAMPLES, Node.EXTRACT_EVIDENCE]
    assert state.samples is not None
    assert (state.evidence, state.tieouts, state.exceptions, state.workpaper) == (
        None,
        None,
        None,
        None,
    )

    grade = score_run(state, CASE_D.answer_key, PAGES, StubJudge())
    extract = [s for s in grade.scores if s.node is Node.EXTRACT_EVIDENCE]
    assert extract and all(not s.passed and s.failure_origin is origin for s in extract)
    assert all(s.trace_ref == failed.trace_ref for s in extract)
    later = [s for s in grade.scores if list(Node).index(s.node) > 1]
    assert later and all(s.skipped for s in later)
    assert all(s.passed for s in grade.scores if s.node is Node.SELECT_SAMPLES)
    assert grade.judge is None
