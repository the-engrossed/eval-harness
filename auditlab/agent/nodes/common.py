"""Shared node plumbing: one span and one StepRecord per node run.

A ProviderError (infra), ModelOutputError or PromptTooLargeError (agent) at a model call
becomes a StepRecord with `error` set and no output; every later node then returns without
running, so one bad call scores as a failed step instead of crashing the run (SPEC.md §9.8).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel

from auditlab.agent.deps import RunDeps
from auditlab.agent.prompts import load_prompt
from auditlab.agent.state import AgentState
from auditlab.domain.enums import Node
from auditlab.domain.models import StepError, StepRecord
from auditlab.models.base import ModelRequest
from auditlab.models.errors import (
    ModelOutputError,
    PromptTooLargeError,
    ProviderError,
    RecordedFailure,
)
from auditlab.money import cost_micros
from auditlab.telemetry import pick_trace_ref


@dataclass
class StepMeter:
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int | None = None
    cost_micros: int = 0
    cached: bool = False
    error: StepError | None = None


@contextmanager
def step(state: AgentState, deps: RunDeps, node: Node) -> Iterator[tuple[Span, StepMeter]]:
    with deps.telemetry.tracer.start_as_current_span(node.value) as span:
        span.set_attribute("audit.case_id", state.case_id)
        span.set_attribute("audit.run_index", state.run_index)
        span.set_attribute("audit.node", node.value)
        span.set_attribute("audit.model_config_id", state.model_config_id)
        span.set_attribute("audit.prompt_version", deps.prompt_version)
        yield span, StepMeter()


def record(span: Span, meter: StepMeter, node: Node, deps: RunDeps, started_ns: int) -> StepRecord:
    latency = meter.latency_ms
    if latency is None:
        latency = (time.perf_counter_ns() - started_ns) // 1_000_000
    span.set_attribute("audit.latency_ms", latency)
    span.set_attribute("audit.cost_micros", meter.cost_micros)
    local = f"local:{deps.run_id}/{span.get_span_context().span_id:016x}"
    refs = {"local": local, **deps.telemetry.refs(span)}
    return StepRecord(
        node=node,
        latency_ms=latency,
        tokens_in=meter.tokens_in,
        tokens_out=meter.tokens_out,
        cost_micros=meter.cost_micros,
        trace_ref=pick_trace_ref(refs),
        trace_refs=refs,
        cached=meter.cached,
        error=meter.error,
    )


def upstream_failed(state: AgentState) -> bool:
    """True once any earlier node recorded an error; later nodes then do nothing."""
    return any(r.error is not None for r in state.step_records)


def _cost(deps: RunDeps, tokens_in: int, tokens_out: int) -> int:
    if deps.price is not None:
        return cost_micros(
            tokens_in,
            tokens_out,
            input_micros_per_million=deps.price.input_micros_per_mtok,
            output_micros_per_million=deps.price.output_micros_per_mtok,
        )
    if deps.client.simulated:
        return 0
    raise RuntimeError(f"no price for real model {deps.client.model_id}; add a [[price]] entry")


def call_model[T: BaseModel](
    state: AgentState,
    deps: RunDeps,
    node: Node,
    payload: dict[str, Any],
    schema: type[T],
) -> tuple[T | None, StepRecord]:
    started = time.perf_counter_ns()
    with step(state, deps, node) as (span, meter):
        request = ModelRequest(
            case_id=state.case_id,
            node=node,
            prompt_version=deps.prompt_version,
            system=load_prompt(deps.prompt_version, node.value),
            payload=payload,
            output_schema=schema,
            run_index=state.run_index,
            seed=deps.seed,
        )
        try:
            response = deps.client.complete(request)
        except (ProviderError, ModelOutputError, PromptTooLargeError, RecordedFailure) as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.set_attribute("audit.failure_origin", exc.failure_origin.value)
            meter.error = StepError(origin=exc.failure_origin, message=str(exc))
            return None, record(span, meter, node, deps, started)
        output = response.output
        if not isinstance(output, schema):
            raise TypeError(
                f"{node.value}: expected {schema.__name__}, got {type(output).__name__}"
            )
        meter.tokens_in, meter.tokens_out = response.tokens_in, response.tokens_out
        meter.latency_ms = response.latency_ms
        meter.cached = response.cached
        meter.cost_micros = _cost(deps, response.tokens_in, response.tokens_out)
        span.set_attribute("gen_ai.system", deps.client.provider)
        span.set_attribute("gen_ai.request.model", deps.client.model_id)
        span.set_attribute("gen_ai.response.model", response.provider_model_id)
        span.set_attribute("gen_ai.usage.input_tokens", response.tokens_in)
        span.set_attribute("gen_ai.usage.output_tokens", response.tokens_out)
        span.set_attribute("audit.simulated", response.simulated)
        span.set_attribute("audit.cached", response.cached)
        return output, record(span, meter, node, deps, started)


def run_code[T: BaseModel](
    state: AgentState, deps: RunDeps, node: Node, fn: Callable[[], T]
) -> tuple[T, StepRecord]:
    started = time.perf_counter_ns()
    with step(state, deps, node) as (span, meter):
        output = fn()
        return output, record(span, meter, node, deps, started)
