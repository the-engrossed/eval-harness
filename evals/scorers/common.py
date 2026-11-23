"""Shared scorer plumbing: node status from step records, and StepScore construction.

A node whose record carries an error scores every metric at its worst value with that
error's origin. A node after a failed node is skipped (`upstream_failed`) and points at
the failed node's trace, so the root step stays the only failure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from auditlab.agent.state import AgentState
from auditlab.domain.enums import FailureOrigin, MetricKind, Node
from auditlab.domain.models import StepError
from auditlab.domain.scores import UPSTREAM_FAILED, StepScore

Pages = Mapping[str, Mapping[int, str]]
MAX_IDS_IN_DETAIL = 10


@dataclass(frozen=True)
class Metric:
    name: str
    kind: MetricKind
    critical: bool
    lower_is_better: bool = False

    @property
    def best(self) -> float:
        return 0.0 if self.lower_is_better else 1.0

    @property
    def worst(self) -> float:
        return 1.0 if self.lower_is_better else 0.0


@dataclass(frozen=True)
class NodeStatus:
    node: Node
    trace_ref: str
    error: StepError | None
    upstream_failed: bool


def node_status(state: AgentState, node: Node) -> NodeStatus:
    records = {r.node: r for r in state.step_records}
    order = list(Node)
    for earlier in order[: order.index(node)]:
        record = records.get(earlier)
        if record is not None and record.error is not None:
            return NodeStatus(node, record.trace_ref, None, upstream_failed=True)
    record = records.get(node)
    if record is None:
        raise ValueError(
            f"run {state.case_id}/{state.run_index} has no step record for {node.value} "
            "and no upstream failure; the run is incomplete"
        )
    return NodeStatus(node, record.trace_ref, record.error, upstream_failed=False)


def output_of[T](value: T | None, status: NodeStatus) -> T:
    if value is None:
        raise ValueError(f"{status.node.value} has a successful step record but no output")
    return value


def make_score(
    state: AgentState, status: NodeStatus, metric: Metric, value: float, detail: str
) -> StepScore:
    passed = value == metric.best
    return StepScore(
        case_id=state.case_id,
        run_index=state.run_index,
        model_config_id=state.model_config_id,
        node=status.node,
        metric=metric.name,
        kind=metric.kind,
        value=value,
        passed=passed,
        critical=metric.critical,
        failure_origin=None if passed else FailureOrigin.AGENT,
        detail=detail,
        trace_ref=status.trace_ref,
    )


def unscored(
    state: AgentState, status: NodeStatus, metrics: list[Metric]
) -> list[StepScore] | None:
    """Scores for a failed or skipped node; None when the node ran and should be scored."""
    if status.upstream_failed:
        return [_unscored(state, status, m, None, UPSTREAM_FAILED, skipped=True) for m in metrics]
    if status.error is not None:
        detail = f"node failed ({status.error.origin.value}): {status.error.message}"
        return [_unscored(state, status, m, status.error.origin, detail) for m in metrics]
    return None


def _unscored(
    state: AgentState,
    status: NodeStatus,
    metric: Metric,
    origin: FailureOrigin | None,
    detail: str,
    *,
    skipped: bool = False,
) -> StepScore:
    return StepScore(
        case_id=state.case_id,
        run_index=state.run_index,
        model_config_id=state.model_config_id,
        node=status.node,
        metric=metric.name,
        kind=metric.kind,
        value=metric.worst,
        passed=False,
        critical=metric.critical,
        failure_origin=origin,
        detail=detail,
        trace_ref=status.trace_ref,
        skipped=skipped,
    )


def share(hits: int, total: int) -> float:
    if total <= 0:
        raise ValueError("share of an empty set is undefined; omit the metric instead")
    return hits / total


def ids_text(ids: Iterable[str]) -> str:
    items = sorted(ids)
    shown = ", ".join(items[:MAX_IDS_IN_DETAIL])
    more = len(items) - MAX_IDS_IN_DETAIL
    return f"{shown} (+{more} more)" if more > 0 else shown
