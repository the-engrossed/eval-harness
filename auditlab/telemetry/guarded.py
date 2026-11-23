"""Span exporter wrapper for remote sinks: never raises, remembers failed traces."""

from __future__ import annotations

import threading
from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class GuardedExporter(SpanExporter):
    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._failed: set[int] = set()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            result = self._inner.export(spans)
        except Exception:
            result = SpanExportResult.FAILURE
        if result is not SpanExportResult.SUCCESS:
            with self._lock:
                self._failed.update(s.context.trace_id for s in spans if s.context)
        return result

    def failed(self, trace_id: int) -> bool:
        with self._lock:
            return trace_id in self._failed

    def forget(self, trace_id: int) -> None:
        with self._lock:
            self._failed.discard(trace_id)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)
