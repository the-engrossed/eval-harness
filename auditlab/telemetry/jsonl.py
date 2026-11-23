"""Local JSONL sink: each span is appended to its run's runs/<run_id>/spans.jsonl.

One exporter serves the whole process. Spans are routed by their `audit.run_id`
attribute, and each file has its own lock so concurrent runs never interleave lines.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

RUN_ID_ATTR = "audit.run_id"
SERVICE_SCOPE = "auditlab"


class RunIdPropagator(SpanProcessor):
    """Copies `audit.run_id` from the parent span so every child span can be routed."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        if span.attributes and RUN_ID_ATTR in span.attributes:
            return
        parent = trace.get_current_span(parent_context)
        attrs = parent.attributes if isinstance(parent, ReadableSpan) else None
        if attrs and RUN_ID_ATTR in attrs:
            span.set_attribute(RUN_ID_ATTR, attrs[RUN_ID_ATTR])


def span_record(span: ReadableSpan) -> dict[str, object]:
    ctx = span.get_span_context()
    if ctx is None:
        raise ValueError(f"span {span.name!r} has no context")
    return {
        "name": span.name,
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
        "service_name": span.resource.attributes.get("service.name"),
        "attributes": dict(span.attributes or {}),
    }


class JsonlRouter(SpanExporter):
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._paths: dict[str, Path] = {}
        self._locks: dict[Path, threading.Lock] = {}

    def register(self, run_id: str, path: Path) -> None:
        with self._guard:
            if run_id in self._paths:
                raise ValueError(f"run {run_id!r} is already open; run ids must be unique")
            self._paths[run_id] = path
            self._locks.setdefault(path, threading.Lock())

    def unregister(self, run_id: str) -> None:
        with self._guard:
            self._paths.pop(run_id, None)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            run_id = (span.attributes or {}).get(RUN_ID_ATTR)
            with self._guard:
                path = self._paths.get(str(run_id))
                lock = self._locks.get(path) if path is not None else None
            if path is None or lock is None:
                raise KeyError(f"span {span.name!r} has no open run ({RUN_ID_ATTR}={run_id!r})")
            line = json.dumps(span_record(span)) + "\n"
            with lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True
