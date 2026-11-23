from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from auditlab.domain.answer_key import AnswerKey
from auditlab.domain.enums import ExceptionType, Node
from auditlab.domain.models import EvidenceSet, ExceptionSet, SampleSelection, Workpaper
from auditlab.guard import validate_prose
from auditlab.models.base import ModelRequest
from auditlab.models.stub import StubModel
from data.generate import build_case, write_case

CASE = build_case(1, "D")


@pytest.fixture(scope="module")
def stub(tmp_path_factory: pytest.TempPathFactory) -> StubModel:
    root = tmp_path_factory.mktemp("engagements")
    write_case(CASE, root)
    return StubModel("stub-prod", root)


def ask(
    stub: StubModel, node: Node, schema: type[BaseModel], payload: dict[str, Any], run: int = 0
) -> Any:
    req = ModelRequest(
        case_id=CASE.case_id,
        node=node,
        prompt_version="v1",
        system=f"You are the {node.value} step of a revenue test.",
        payload=payload,
        output_schema=schema,
        run_index=run,
        seed=7,
    )
    assert CASE.case_id not in req.system
    assert CASE.case_id not in req.payload_json()
    response = stub.complete(req)
    assert response.simulated is True
    assert isinstance(response.output, schema)
    assert schema.model_validate_json(response.output.model_dump_json()) == response.output
    return response.output


def key() -> AnswerKey:
    return CASE.answer_key


def run_chain(stub: StubModel, item_ids: list[str]) -> tuple[EvidenceSet, ExceptionSet, Workpaper]:
    evidence = ask(stub, Node.EXTRACT_EVIDENCE, EvidenceSet, {"item_ids": item_ids})
    exceptions = ask(
        stub, Node.FLAG_EXCEPTIONS, ExceptionSet, {"evidence": evidence.model_dump(mode="json")}
    )
    workpaper = ask(
        stub,
        Node.DRAFT_WORKPAPER,
        Workpaper,
        {
            "evidence": evidence.model_dump(mode="json"),
            "exceptions": exceptions.model_dump(mode="json"),
        },
    )
    return evidence, exceptions, workpaper


def test_select_samples_returns_the_required_sample(stub: StubModel) -> None:
    out = ask(stub, Node.SELECT_SAMPLES, SampleSelection, {"sampling_rule": "rule text"})
    assert out.item_ids == key().required_sample_ids
    assert len(out.rationale) == 3


def test_full_chain_on_the_required_sample_finds_every_planted_item(stub: StubModel) -> None:
    evidence, exceptions, workpaper = run_chain(stub, key().required_sample_ids)
    assert {f.item_id for f in evidence.fields} == set(key().required_sample_ids)
    assert {(e.item_id, e.type) for e in exceptions.exceptions} == {
        (e.item_id, e.type) for e in key().expected_exceptions
    }
    assert {j.item_id for j in exceptions.judgments} == {j.item_id for j in key().judgment_items}
    ids = frozenset(f.evidence_id for f in evidence.fields)
    items = frozenset(f.item_id for f in evidence.fields)
    assert validate_prose(workpaper.prose, ids, item_ids=items) == []
    assert all(f"[[item:{i}]]" in workpaper.prose for i in items)
    assert workpaper.guard_violations == []


def test_downstream_nodes_answer_only_for_the_ids_they_were_given(stub: StubModel) -> None:
    cutoffs = [e for e in key().expected_exceptions if e.type is ExceptionType.CUTOFF_ERROR]
    [cutoff] = [e.item_id for e in cutoffs]
    subset = [i for i in key().required_sample_ids if i != cutoff]
    evidence, exceptions, workpaper = run_chain(stub, subset)
    assert cutoff not in {f.item_id for f in evidence.fields}
    assert cutoff not in {e.item_id for e in exceptions.exceptions}
    assert ExceptionType.CUTOFF_ERROR not in {e.type for e in exceptions.exceptions}
    assert f"ev-{cutoff}-" not in workpaper.prose
    assert f"[[item:{cutoff}]]" not in workpaper.prose


def test_an_id_with_no_answer_key_evidence_is_dropped_not_invented(stub: StubModel) -> None:
    extra = next(
        i.invoice_id
        for i in CASE.population.invoices
        if i.invoice_id not in key().required_sample_ids
    )
    evidence = ask(stub, Node.EXTRACT_EVIDENCE, EvidenceSet, {"item_ids": [extra]})
    assert evidence.fields == []
    assert [d.item_id for d in evidence.dropped] == [extra]


def test_same_run_index_gives_the_same_output(stub: StubModel) -> None:
    a = run_chain(stub, key().required_sample_ids)
    b = run_chain(stub, key().required_sample_ids)
    assert a == b


def test_stub_refuses_tie_out(stub: StubModel) -> None:
    req = ModelRequest(
        case_id=CASE.case_id,
        node=Node.TIE_OUT,
        prompt_version="v1",
        system="tie out",
        payload={},
        output_schema=SampleSelection,
        run_index=0,
        seed=7,
    )
    with pytest.raises(ValueError, match="tie_out"):
        stub.complete(req)


def test_request_refuses_case_id_in_prompt_text() -> None:
    base: dict[str, Any] = {
        "case_id": "s1-mixD",
        "node": Node.SELECT_SAMPLES,
        "prompt_version": "v1",
        "output_schema": SampleSelection,
        "run_index": 0,
        "seed": 7,
    }
    with pytest.raises(ValueError, match="case_id"):
        ModelRequest(**base, system="Case s1-mixD.", payload={})
    with pytest.raises(ValueError, match="case_id"):
        ModelRequest(**base, system="ok", payload={"note": ["see s1-mixD"]})


def test_unknown_profile_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stub-cand-unknown"):
        StubModel("stub-cand-unknown", tmp_path)
