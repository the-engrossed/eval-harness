import json
import threading
from pathlib import Path

import pytest
from cases import CASE_D

from auditlab.agent.graph import run_case
from auditlab.domain.enums import Node
from auditlab.models.stub import StubModel
from auditlab.telemetry import open_run, pick_trace_ref


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_spans_land_in_the_given_jsonl_file(tmp_path: Path) -> None:
    spans_path = tmp_path / "run-x" / "spans.jsonl"
    with (
        open_run("run-x", spans_path) as run,
        run.span("agent_run"),
        run.tracer.start_as_current_span("tie_out") as span,
    ):
        span.set_attribute("audit.node", "tie_out")

    records = _lines(spans_path)
    assert [r["name"] for r in records] == ["tie_out", "agent_run"]
    child = records[0]
    assert child["attributes"] == {"audit.run_id": "run-x", "audit.node": "tie_out"}
    assert child["service_name"] == "auditlab"
    assert len(str(child["span_id"])) == 16
    assert len(str(child["trace_id"])) == 32


def test_a_run_id_cannot_be_opened_twice_at_once(tmp_path: Path) -> None:
    with (
        open_run("dup", tmp_path / "a.jsonl"),
        pytest.raises(ValueError, match="dup"),
        open_run("dup", tmp_path / "b.jsonl"),
    ):
        pass


def test_many_spans_from_two_threads_stay_in_their_own_files(tmp_path: Path) -> None:
    def emit(run_id: str) -> None:
        with open_run(run_id, tmp_path / run_id / "spans.jsonl") as run, run.span("agent_run"):
            for i in range(200):
                with run.tracer.start_as_current_span(f"s{i}"):
                    pass

    threads = [threading.Thread(target=emit, args=(r,)) for r in ("ta", "tb")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for run_id in ("ta", "tb"):
        records = _lines(tmp_path / run_id / "spans.jsonl")
        assert len(records) == 201
        assert {r["attributes"]["audit.run_id"] for r in records} == {run_id}  # type: ignore[index]
        assert len({r["trace_id"] for r in records}) == 1


def test_two_agent_runs_on_threads_write_correct_separate_span_files(
    engagements: Path, tmp_path: Path
) -> None:
    states = {}

    def go(run_id: str) -> None:
        states[run_id] = run_case(
            CASE_D.case_id,
            client=StubModel("stub-prod", engagements),
            model_config_id="prod",
            engagements_dir=engagements,
            runs_dir=tmp_path,
            run_id=run_id,
        )

    threads = [threading.Thread(target=go, args=(r,)) for r in ("par-a", "par-b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for run_id in ("par-a", "par-b"):
        records = _lines(tmp_path / run_id / "spans.jsonl")
        assert sorted(str(r["name"]) for r in records) == sorted(
            ["agent_run", *[n.value for n in Node]]
        )
        assert {r["attributes"]["audit.run_id"] for r in records} == {run_id}  # type: ignore[index]
        refs = {f"local:{run_id}/{r['span_id']}" for r in records}
        assert all(s.trace_ref in refs for s in states[run_id].step_records)


def test_trace_ref_is_the_first_remote_ref_else_local() -> None:
    local = "local:r/0000000000000001"
    assert pick_trace_ref({"local": local}) == local
    assert pick_trace_ref({"local": local, "otlp": "otlp:x"}) == "otlp:x"
    assert pick_trace_ref({"local": local, "otlp": "o", "langfuse": "lf"}) == "lf"
    assert (
        pick_trace_ref({"local": local, "otlp": "o", "langfuse": "lf", "langsmith": "ls"}) == "ls"
    )
