import json
from pathlib import Path

from cases import CASE_D

from auditlab.agent.graph import GRAPH, run_case
from auditlab.agent.nodes.extract_evidence import extract_payload
from auditlab.docs.store import DocStore
from auditlab.domain.enums import Node
from auditlab.domain.models import SampleSelection
from auditlab.models.base import ModelClient, ModelRequest, ModelResponse
from auditlab.models.stub import StubModel

KEY = CASE_D.answer_key


class Recorder:
    """Passes calls through and keeps every request; optionally rewrites select_samples."""

    def __init__(self, inner: ModelClient, extra_ids: list[str] | None = None) -> None:
        self.inner = inner
        self.provider, self.model_id, self.simulated = inner.provider, inner.model_id, True
        self.extra_ids = extra_ids or []
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = self.inner.complete(request)
        if request.node is Node.SELECT_SAMPLES and self.extra_ids:
            sel = response.output
            assert isinstance(sel, SampleSelection)
            out = SampleSelection(
                item_ids=[*sel.item_ids, *self.extra_ids], rationale=sel.rationale
            )
            response = response.model_copy(update={"output": out})
        return response


def run(engagements: Path, tmp_path: Path, client: ModelClient) -> object:
    return run_case(
        CASE_D.case_id,
        client=client,
        model_config_id="prod",
        engagements_dir=engagements,
        runs_dir=tmp_path,
        run_id="test-run",
    )


def test_graph_is_the_five_node_chain() -> None:
    edges = {(e.source, e.target) for e in GRAPH.get_graph().edges}
    chain = ["__start__", *[n.value for n in Node], "__end__"]
    assert edges == set(zip(chain, chain[1:], strict=False))


def test_end_to_end_on_one_case_with_stub_prod(engagements: Path, tmp_path: Path) -> None:
    recorder = Recorder(StubModel("stub-prod", engagements))
    state = run_case(
        CASE_D.case_id,
        client=recorder,
        model_config_id="prod",
        engagements_dir=engagements,
        runs_dir=tmp_path,
        run_id="test-run",
    )
    assert state.samples is not None and state.samples.item_ids == KEY.required_sample_ids
    assert state.evidence is not None
    assert {f.item_id for f in state.evidence.fields} == set(KEY.required_sample_ids)
    assert state.tieouts is not None
    assert state.tieouts.tieouts == KEY.expected_tieouts
    assert state.exceptions is not None
    assert {(e.item_id, e.type) for e in state.exceptions.exceptions} == {
        (e.item_id, e.type) for e in KEY.expected_exceptions
    }
    assert state.workpaper is not None and state.workpaper.guard_violations == []
    assert "[[item:" in state.workpaper.prose

    assert [r.node for r in state.step_records] == list(Node)
    spans = [json.loads(line) for line in (tmp_path / "test-run" / "spans.jsonl").open()]
    span_refs = {f"local:test-run/{s['span_id']}" for s in spans}
    assert all(r.trace_ref in span_refs for r in state.step_records)
    tie = next(r for r in state.step_records if r.node is Node.TIE_OUT)
    assert (tie.tokens_in, tie.tokens_out, tie.cost_micros) == (0, 0, 0)

    [select] = [r for r in recorder.requests if r.node is Node.SELECT_SAMPLES]
    assert set(select.payload) == {"population_summary", "sampling_rule"}
    assert set(select.payload["population_summary"]) == {"invoices", "credit_memo_invoice_ids"}
    assert Node.TIE_OUT not in {r.node for r in recorder.requests}


def test_an_invented_id_is_dropped_as_unknown_item_not_a_crash(
    engagements: Path, tmp_path: Path
) -> None:
    client = Recorder(StubModel("stub-prod", engagements), extra_ids=["INV-9999", "../population"])
    state = run_case(
        CASE_D.case_id,
        client=client,
        model_config_id="prod",
        engagements_dir=engagements,
        runs_dir=tmp_path,
        run_id="invented",
    )
    assert state.evidence is not None
    unknown = {d.item_id for d in state.evidence.dropped if d.reason == "unknown_item"}
    assert unknown == {"INV-9999", "../population"}
    assert state.workpaper is not None


def test_bank_pages_sent_do_not_depend_on_whether_a_deposit_exists(engagements: Path) -> None:
    docs = DocStore(engagements / CASE_D.case_id)
    items = KEY.required_sample_ids
    pop = CASE_D.population
    without = pop.model_copy(update={"deposits": []})
    with_payload, _ = extract_payload(pop, items, docs)
    without_payload, _ = extract_payload(without, items, docs)
    assert with_payload == without_payload
    assert with_payload["documents"]["bank-statement"]
