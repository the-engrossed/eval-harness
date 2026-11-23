"""Optional remote sinks (SPEC.md §7): LangSmith, Langfuse, any OTLP backend.

Each is off unless its env vars are set, and its SDK is imported only when it is on.
Half-set config raises at startup. At runtime a sink can only fail softly: a failed
export or id lookup is recorded so the run drops that sink's refs and carries on.
Refs are ids, not URLs; URLs are built at report time, off the run's hot path.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Span

from auditlab.telemetry.guarded import GuardedExporter
from auditlab.telemetry.jsonl import SERVICE_SCOPE

REMOTE_SINKS = ("langsmith", "langfuse", "otlp")
LANGFUSE_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")


class SinkConfigError(RuntimeError):
    pass


def enabled_sinks(env: Mapping[str, str]) -> tuple[str, ...]:
    names: list[str] = []
    tracing = env.get("LANGSMITH_TRACING", "").strip().lower()
    if tracing == "true":
        if not env.get("LANGSMITH_API_KEY"):
            raise SinkConfigError(
                "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty. "
                "Set the key in .env, or set LANGSMITH_TRACING=false."
            )
        names.append("langsmith")
    elif tracing not in ("", "false"):
        raise SinkConfigError(f"LANGSMITH_TRACING must be true or false, got {tracing!r}")
    present = [v for v in LANGFUSE_VARS if env.get(v)]
    if present:
        missing = [v for v in LANGFUSE_VARS if not env.get(v)]
        if missing:
            raise SinkConfigError(
                f"Langfuse is half-configured: {', '.join(missing)} not set. "
                f"Set all of {', '.join(LANGFUSE_VARS)} in .env, or none."
            )
        names.append("langfuse")
    if env.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        names.append("otlp")
    return tuple(names)


class Sink(Protocol):
    name: str

    def context(self) -> AbstractContextManager[None]: ...

    def ref(self, span: Span) -> str: ...

    def finish(self, trace_id: int) -> bool: ...

    def shutdown(self) -> None: ...


def _otlp_exporter() -> SpanExporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter()


def _langfuse_exporter(host: str, public_key: str, secret_key: str) -> SpanExporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return OTLPSpanExporter(
        endpoint=f"{host.rstrip('/')}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {token}"},
    )


def _is_ours(span: ReadableSpan) -> bool:
    scope = span.instrumentation_scope
    return scope is not None and scope.name == SERVICE_SCOPE


class LangSmithSink:
    """LangGraph's native tracing does the export; we only read the node's run id."""

    name = "langsmith"

    def __init__(self, project: str | None) -> None:
        self._project = project

    def context(self) -> AbstractContextManager[None]:
        from langsmith.run_helpers import tracing_context

        return tracing_context(enabled=True, project_name=self._project)

    def ref(self, span: Span) -> str:
        from langsmith.run_helpers import get_current_run_tree

        run_tree = get_current_run_tree()
        if run_tree is None:
            raise LookupError("no active LangSmith run for this node")
        return f"langsmith:{run_tree.id}"

    def finish(self, trace_id: int) -> bool:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
        return True

    def shutdown(self) -> None:
        return None


class OtelSink:
    def __init__(self, name: str, exporter: GuardedExporter, flush: Any, ref_fmt: str) -> None:
        self.name = name
        self._exporter = exporter
        self._flush = flush
        self._ref_fmt = ref_fmt

    def context(self) -> AbstractContextManager[None]:
        return nullcontext()

    def ref(self, span: Span) -> str:
        ctx = span.get_span_context()
        return self._ref_fmt.format(trace=f"{ctx.trace_id:032x}", span=f"{ctx.span_id:016x}")

    def finish(self, trace_id: int) -> bool:
        flushed = self._flush()
        ok = flushed is not False and not self._exporter.failed(trace_id)
        self._exporter.forget(trace_id)
        return ok

    def shutdown(self) -> None:
        self._exporter.shutdown()


def _build(name: str, provider: TracerProvider, env: Mapping[str, str]) -> Sink:
    if name == "langsmith":
        import langsmith  # noqa: F401

        return LangSmithSink(env.get("LANGSMITH_PROJECT") or None)
    if name == "langfuse":
        from langfuse import Langfuse

        public_key, secret_key, host = (env[v] for v in LANGFUSE_VARS)
        exporter = GuardedExporter(_langfuse_exporter(host, public_key, secret_key))
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            tracer_provider=provider,
            span_exporter=exporter,
            should_export_span=_is_ours,
        )
        return OtelSink(name, exporter, client.flush, "langfuse:{trace}")
    if name == "otlp":
        exporter = GuardedExporter(_otlp_exporter())
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        return OtelSink(name, exporter, processor.force_flush, "otlp:{trace}/{span}")
    raise SinkConfigError(f"unknown sink {name!r}")


def build_sinks(
    names: Sequence[str], provider: TracerProvider, env: Mapping[str, str]
) -> tuple[Sink, ...]:
    built = []
    for name in names:
        try:
            built.append(_build(name, provider, env))
        except ImportError as exc:
            raise SinkConfigError(
                f"the {name} sink is switched on but its SDK is not installed. "
                f"Run `uv sync --extra {name}` or unset its env vars."
            ) from exc
    return tuple(built)
