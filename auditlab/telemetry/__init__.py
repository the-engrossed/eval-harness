"""OpenTelemetry for agent runs: one process-wide provider, local JSONL plus optional sinks.

The provider is built on first use from the environment and shared by every run, so
concurrent runs and SDKs that bind to one provider per process (Langfuse) both work.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Collection, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Span

from auditlab.domain.models import StepRecord
from auditlab.telemetry.ids import SeededIdGenerator
from auditlab.telemetry.ids import seeded_ids as seeded_ids
from auditlab.telemetry.jsonl import RUN_ID_ATTR, SERVICE_SCOPE, JsonlRouter, RunIdPropagator
from auditlab.telemetry.sinks import REMOTE_SINKS, Sink, build_sinks, enabled_sinks

log = logging.getLogger("auditlab.telemetry")


@dataclass(frozen=True)
class Telemetry:
    provider: TracerProvider
    tracer: trace.Tracer
    router: JsonlRouter
    sinks: tuple[Sink, ...]


_LOCK = threading.Lock()
_current: Telemetry | None = None


def telemetry() -> Telemetry:
    global _current
    with _LOCK:
        if _current is None:
            names = enabled_sinks(os.environ)
            provider = TracerProvider(
                resource=Resource.create({"service.name": SERVICE_SCOPE}),
                id_generator=SeededIdGenerator(),
            )
            provider.add_span_processor(RunIdPropagator())
            router = JsonlRouter()
            provider.add_span_processor(SimpleSpanProcessor(router))
            try:
                sinks = build_sinks(names, provider, os.environ)
            except Exception:
                provider.shutdown()
                raise
            _current = Telemetry(provider, provider.get_tracer(SERVICE_SCOPE), router, sinks)
        return _current


def reset_telemetry() -> None:
    """Flush and forget the provider; the next run re-reads the environment."""
    global _current
    with _LOCK:
        if _current is None:
            return
        _current.provider.shutdown()
        for sink in _current.sinks:
            try:
                sink.shutdown()
            except Exception as exc:
                log.warning("%s sink failed to shut down: %s", sink.name, exc)
        _current = None


def pick_trace_ref(refs: Mapping[str, str]) -> str:
    for name in REMOTE_SINKS:
        if name in refs:
            return refs[name]
    if "local" not in refs:
        raise KeyError("trace_refs has no local ref")
    return refs["local"]


def without_sinks(records: list[StepRecord], failed: Collection[str]) -> list[StepRecord]:
    out = []
    for record in records:
        refs = {k: v for k, v in record.trace_refs.items() if k not in failed}
        out.append(
            record.model_copy(update={"trace_refs": refs, "trace_ref": pick_trace_ref(refs)})
        )
    return out


class RunTelemetry:
    def __init__(self, run_id: str, tel: Telemetry) -> None:
        self.run_id = run_id
        self.tracer = tel.tracer
        self._sinks = tel.sinks
        self._lock = threading.Lock()
        self._failed: set[str] = set()

    @contextmanager
    def span(self, name: str) -> Iterator[Span]:
        """A root span for this run, inside every sink's per-run context."""
        with ExitStack() as stack:
            for sink in self._sinks:
                stack.enter_context(sink.context())
            attributes = {RUN_ID_ATTR: self.run_id}
            yield stack.enter_context(
                self.tracer.start_as_current_span(name, attributes=attributes)
            )

    def refs(self, span: Span) -> dict[str, str]:
        out = {}
        for sink in self._sinks:
            if self._is_failed(sink.name):
                continue
            try:
                out[sink.name] = sink.ref(span)
            except Exception as exc:
                self._fail(sink.name, f"id lookup failed: {exc}")
        return out

    def finish(self, trace_id: int) -> frozenset[str]:
        """Flush every sink; return the sinks whose refs must be dropped for this run."""
        for sink in self._sinks:
            try:
                ok = sink.finish(trace_id)
            except Exception as exc:
                self._fail(sink.name, f"flush failed: {exc}")
                continue
            if not ok:
                self._fail(sink.name, "export failed")
        with self._lock:
            return frozenset(self._failed)

    def _is_failed(self, name: str) -> bool:
        with self._lock:
            return name in self._failed

    def _fail(self, name: str, reason: str) -> None:
        with self._lock:
            if name in self._failed:
                return
            self._failed.add(name)
        log.warning(
            "%s sink failed for run %s (%s); dropping its refs and keeping the local ref",
            name,
            self.run_id,
            reason,
        )


@contextmanager
def open_run(run_id: str, spans_path: Path) -> Iterator[RunTelemetry]:
    tel = telemetry()
    tel.router.register(run_id, spans_path)
    try:
        yield RunTelemetry(run_id, tel)
    finally:
        tel.router.unregister(run_id)
