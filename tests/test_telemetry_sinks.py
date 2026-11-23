import logging
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import requests
from cases import CASE_D
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from auditlab.agent.graph import run_case
from auditlab.agent.state import AgentState
from auditlab.domain.enums import Node
from auditlab.models.stub import StubModel
from auditlab.telemetry import reset_telemetry, sinks
from auditlab.telemetry.sinks import SinkConfigError, enabled_sinks

REPO = Path(__file__).resolve().parent.parent
SINK_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)
LANGFUSE_ENV = {
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "LANGFUSE_HOST": "http://langfuse.test",
}
NODE_NAMES = {n.value for n in Node}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in SINK_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_telemetry()
    yield
    reset_telemetry()


def _run(engagements: Path, runs: Path, run_id: str = "sink-run") -> AgentState:
    return run_case(
        CASE_D.case_id,
        client=StubModel("stub-prod", engagements),
        model_config_id="prod",
        engagements_dir=engagements,
        runs_dir=runs,
        run_id=run_id,
    )


class RaisingExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        raise ConnectionError("collector unreachable")


class FailingExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.FAILURE


class OkSession(requests.Session):
    def request(self, method: str | bytes, url: str | bytes, *args: Any, **kwargs: Any) -> Any:
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.url = str(url)
        return response


class DownSession(requests.Session):
    def request(self, method: str | bytes, url: str | bytes, *args: Any, **kwargs: Any) -> Any:
        raise requests.ConnectionError("network down")


def _fake_langsmith(monkeypatch: pytest.MonkeyPatch, session: requests.Session) -> Any:
    import langchain_core.tracers.langchain as lc_tracer
    from langsmith import Client

    client = Client(api_key="lsv2-test", api_url="http://langsmith.test", session=session)
    monkeypatch.setattr(lc_tracer, "get_client", lambda: client)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://langsmith.test")
    return client


def _fake_langfuse(monkeypatch: pytest.MonkeyPatch, exporter: SpanExporter, key: str) -> None:
    monkeypatch.setattr(sinks, "_langfuse_exporter", lambda host, pk, sk: exporter)
    for var, value in {**LANGFUSE_ENV, "LANGFUSE_PUBLIC_KEY": key}.items():
        monkeypatch.setenv(var, value)


def _fake_otlp(monkeypatch: pytest.MonkeyPatch, exporter: SpanExporter) -> None:
    monkeypatch.setattr(sinks, "_otlp_exporter", lambda: exporter)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otlp.test:4318")


def test_with_no_sink_env_no_sdk_is_imported(engagements: Path, tmp_path: Path) -> None:
    code = f"""
import sys
from pathlib import Path

ROOTS = {{"langsmith": "langsmith", "langfuse": "langfuse", "otlp": "opentelemetry.exporter.otlp"}}

def loaded(*names):
    roots = [ROOTS[n] for n in names]
    return sorted(m for m in sys.modules if any(m == r or m.startswith(r + ".") for r in roots))

from auditlab.telemetry import telemetry
telemetry()
assert loaded("langsmith", "langfuse", "otlp") == [], loaded("langsmith", "langfuse", "otlp")

from auditlab.agent.graph import run_case
from auditlab.models.stub import StubModel
eng = Path({str(engagements)!r})
run_case({CASE_D.case_id!r}, client=StubModel("stub-prod", eng), model_config_id="prod",
         engagements_dir=eng, runs_dir=Path({str(tmp_path)!r}), run_id="lazy")
assert loaded("langfuse", "otlp") == [], loaded("langfuse", "otlp")
"""
    env = {k: v for k, v in os.environ.items() if k not in SINK_VARS}
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, ()),
        ({"LANGSMITH_TRACING": "false", "LANGSMITH_API_KEY": "k"}, ()),
        ({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "k"}, ("langsmith",)),
        ({**LANGFUSE_ENV, "LANGFUSE_PUBLIC_KEY": "pk"}, ("langfuse",)),
        ({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x"}, ("otlp",)),
        (
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://x",
                **LANGFUSE_ENV,
                "LANGFUSE_PUBLIC_KEY": "pk",
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_API_KEY": "k",
            },
            ("langsmith", "langfuse", "otlp"),
        ),
    ],
)
def test_each_sink_is_selected_only_by_its_env_vars(
    env: dict[str, str], expected: tuple[str, ...]
) -> None:
    assert enabled_sinks(env) == expected


@pytest.mark.parametrize(
    ("env", "missing"),
    [
        ({"LANGSMITH_TRACING": "true"}, "LANGSMITH_API_KEY"),
        ({"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_HOST": "h"}, "LANGFUSE_SECRET_KEY"),
        ({"LANGFUSE_SECRET_KEY": "sk"}, "LANGFUSE_PUBLIC_KEY"),
    ],
)
def test_half_set_config_raises_naming_what_is_missing(env: dict[str, str], missing: str) -> None:
    with pytest.raises(SinkConfigError, match=missing):
        enabled_sinks(env)


def test_an_enabled_sink_without_its_sdk_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch, engagements: Path, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otlp.test:4318")
    with pytest.raises(SinkConfigError, match="--extra otlp"):
        _run(engagements, tmp_path)


def test_otlp_ref_lands_on_every_step_record(
    monkeypatch: pytest.MonkeyPatch, engagements: Path, tmp_path: Path
) -> None:
    exporter = InMemorySpanExporter()
    _fake_otlp(monkeypatch, exporter)
    state = _run(engagements, tmp_path)
    for record in state.step_records:
        assert re.fullmatch(r"otlp:[0-9a-f]{32}/[0-9a-f]{16}", record.trace_refs["otlp"])
        assert record.trace_ref == record.trace_refs["otlp"]
        assert record.trace_refs["local"].startswith("local:sink-run/")
    assert {s.name for s in exporter.get_finished_spans()} >= NODE_NAMES


def test_langfuse_ref_lands_on_every_step_record_across_two_runs(
    monkeypatch: pytest.MonkeyPatch, engagements: Path, tmp_path: Path
) -> None:
    exporter = InMemorySpanExporter()
    _fake_langfuse(monkeypatch, exporter, "pk-lf-two-runs")
    first = _run(engagements, tmp_path, "lf-1")
    second = _run(engagements, tmp_path, "lf-2")
    trace_ids = set()
    for state in (first, second):
        refs = {r.trace_refs["langfuse"] for r in state.step_records}
        assert len(refs) == 1
        [ref] = refs
        assert re.fullmatch(r"langfuse:[0-9a-f]{32}", ref)
        assert all(r.trace_ref == ref for r in state.step_records)
        trace_ids.add(ref.removeprefix("langfuse:"))
    exported = {f"{s.context.trace_id:032x}" for s in exporter.get_finished_spans() if s.context}
    assert trace_ids <= exported and len(trace_ids) == 2


def test_langsmith_ref_lands_and_wins_over_the_other_sinks(
    monkeypatch: pytest.MonkeyPatch, engagements: Path, tmp_path: Path
) -> None:
    _fake_langsmith(monkeypatch, OkSession())
    _fake_langfuse(monkeypatch, InMemorySpanExporter(), "pk-lf-with-langsmith")
    _fake_otlp(monkeypatch, InMemorySpanExporter())
    state = _run(engagements, tmp_path)
    ids = set()
    for record in state.step_records:
        ref = record.trace_refs["langsmith"]
        assert re.fullmatch(r"langsmith:[0-9a-f-]{36}", ref)
        assert record.trace_ref == ref
        assert {"local", "langsmith", "langfuse", "otlp"} == set(record.trace_refs)
        ids.add(ref)
    assert len(ids) == len(state.step_records)


@pytest.mark.parametrize("exporter", [RaisingExporter(), FailingExporter()])
def test_a_failing_otlp_export_never_fails_the_run(
    monkeypatch: pytest.MonkeyPatch,
    engagements: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    exporter: SpanExporter,
) -> None:
    _fake_otlp(monkeypatch, exporter)
    with caplog.at_level(logging.WARNING, logger="auditlab.telemetry"):
        state = _run(engagements, tmp_path)
    assert state.workpaper is not None
    for record in state.step_records:
        assert set(record.trace_refs) == {"local"}
        assert record.trace_ref == record.trace_refs["local"]
    warnings = [r for r in caplog.records if r.name == "auditlab.telemetry"]
    assert len(warnings) == 1 and "otlp" in warnings[0].getMessage()


def test_a_failing_langfuse_export_drops_only_the_langfuse_ref(
    monkeypatch: pytest.MonkeyPatch, engagements: Path, tmp_path: Path
) -> None:
    _fake_langfuse(monkeypatch, RaisingExporter(), "pk-lf-failing")
    _fake_otlp(monkeypatch, InMemorySpanExporter())
    state = _run(engagements, tmp_path)
    for record in state.step_records:
        assert set(record.trace_refs) == {"local", "otlp"}
        assert record.trace_ref == record.trace_refs["otlp"]


def test_langsmith_down_never_fails_the_run(
    monkeypatch: pytest.MonkeyPatch, engagements: Path, tmp_path: Path
) -> None:
    client = _fake_langsmith(monkeypatch, DownSession())
    state = _run(engagements, tmp_path)
    client.flush()
    assert state.workpaper is not None
    assert [r.node for r in state.step_records] == list(Node)


def test_a_failed_langsmith_id_lookup_leaves_its_ref_off(
    monkeypatch: pytest.MonkeyPatch,
    engagements: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import langsmith.run_helpers

    _fake_langsmith(monkeypatch, OkSession())
    monkeypatch.setattr(langsmith.run_helpers, "get_current_run_tree", lambda: None)
    with caplog.at_level(logging.WARNING, logger="auditlab.telemetry"):
        state = _run(engagements, tmp_path)
    for record in state.step_records:
        assert set(record.trace_refs) == {"local"}
    warnings = [r for r in caplog.records if r.name == "auditlab.telemetry"]
    assert len(warnings) == 1 and "langsmith" in warnings[0].getMessage()
