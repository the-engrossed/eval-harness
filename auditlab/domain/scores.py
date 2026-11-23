"""StepScore (SPEC.md §9): one metric for one node of one run."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from auditlab.domain.base import DomainModel, Id, Text
from auditlab.domain.enums import FailureOrigin, MetricKind, Node

UPSTREAM_FAILED = "upstream_failed"


class StepScore(DomainModel):
    """A failed score names its origin; a skipped score (an earlier node failed) has none.

    Only `agent` failures and passes count toward accuracy (SPEC.md §9.8); skipped,
    `infra` and `data` scores are reported separately.
    """

    case_id: Id
    run_index: int = Field(ge=0)
    model_config_id: Id
    node: Node
    metric: Id
    kind: MetricKind
    value: float = Field(ge=0.0, le=1.0)
    passed: bool
    critical: bool
    failure_origin: FailureOrigin | None
    detail: Text
    trace_ref: Id
    skipped: bool = False

    @model_validator(mode="after")
    def _status_is_consistent(self) -> Self:
        if self.skipped:
            if self.passed or self.failure_origin is not None or self.detail != UPSTREAM_FAILED:
                raise ValueError(
                    f"{self.metric}: a skipped score is not passed, has no failure_origin, "
                    f"and its detail is {UPSTREAM_FAILED!r}"
                )
        elif self.passed and self.failure_origin is not None:
            raise ValueError(f"{self.metric}: a passed score has no failure_origin")
        elif not self.passed and self.failure_origin is None:
            raise ValueError(f"{self.metric}: a failed score needs a failure_origin")
        return self

    @property
    def counts_toward_accuracy(self) -> bool:
        return not self.skipped and (self.passed or self.failure_origin is FailureOrigin.AGENT)
