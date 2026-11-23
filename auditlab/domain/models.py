"""Node outputs and step records: the types of the AgentState fields (SPEC.md §5.2)."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from auditlab.domain.base import Amount, DomainModel, Id, Text, require_unique
from auditlab.domain.enums import (
    JUDGMENT_TYPES,
    ExceptionType,
    FailureOrigin,
    JudgmentOutcome,
    Node,
)
from auditlab.guard import GuardCode
from auditlab.money import Cents, Micros


class SampleSelection(DomainModel):
    item_ids: list[Id] = Field(min_length=1)
    rationale: list[Text] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        require_unique("item_id", self.item_ids)
        return self


class CitedValue(DomainModel):
    item_id: Id
    field: Id
    value_cents: Amount | None = None
    value_text: Text | None = None
    doc_id: Id
    page_no: int = Field(ge=1)
    quote: Text

    @model_validator(mode="after")
    def _exactly_one_value(self) -> Self:
        if (self.value_cents is None) == (self.value_text is None):
            raise ValueError("set exactly one of value_cents or value_text")
        return self


class EvidenceField(CitedValue):
    evidence_id: Id


class DroppedField(DomainModel):
    item_id: Id
    field: Id
    reason: Text


class EvidenceSet(DomainModel):
    fields: list[EvidenceField]
    dropped: list[DroppedField]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        require_unique("evidence_id", [f.evidence_id for f in self.fields])
        return self


class TieOut(DomainModel):
    """Deltas are (other amount - invoice amount); None when the other amount is missing.

    `matched` covers amounts only (invoice vs GL vs deposit). `evidence_complete` records
    whether the bill of lading is present and never affects `matched`.
    """

    item_id: Id
    invoice_cents: Amount
    gl_cents: Amount | None
    deposit_cents: Amount | None
    gl_delta_cents: Cents | None
    deposit_delta_cents: Cents | None
    matched: bool
    evidence_complete: bool

    @model_validator(mode="after")
    def _arithmetic_agrees(self) -> Self:
        gl_delta = None if self.gl_cents is None else self.gl_cents - self.invoice_cents
        dep_delta = None if self.deposit_cents is None else self.deposit_cents - self.invoice_cents
        matched = gl_delta == 0 and dep_delta == 0
        if (self.gl_delta_cents, self.deposit_delta_cents, self.matched) != (
            gl_delta,
            dep_delta,
            matched,
        ):
            raise ValueError(
                f"tie-out for {self.item_id} disagrees with its amounts: expected "
                f"gl_delta={gl_delta}, deposit_delta={dep_delta}, matched={matched}"
            )
        return self


class TieOutSet(DomainModel):
    tieouts: list[TieOut]


class FlaggedException(DomainModel):
    item_id: Id
    type: ExceptionType
    rationale: Text

    @model_validator(mode="after")
    def _not_judgment(self) -> Self:
        if self.type in JUDGMENT_TYPES:
            raise ValueError(f"{self.type} is a judgment type; record it as a JudgmentDecision")
        return self


class JudgmentDecision(DomainModel):
    item_id: Id
    type: ExceptionType
    outcome: JudgmentOutcome
    rationale: Text

    @model_validator(mode="after")
    def _is_judgment(self) -> Self:
        if self.type not in JUDGMENT_TYPES:
            raise ValueError(f"{self.type} is not a judgment type")
        return self


class ExceptionSet(DomainModel):
    exceptions: list[FlaggedException]
    judgments: list[JudgmentDecision]


class GuardFinding(DomainModel):
    code: GuardCode
    message: Text


class Workpaper(DomainModel):
    prose: Text
    guard_violations: list[GuardFinding]


class StepError(DomainModel):
    """Why a node raised (ProviderError -> infra, ModelOutputError -> agent)."""

    origin: FailureOrigin
    message: Text


class StepRecord(DomainModel):
    node: Node
    latency_ms: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_micros: Annotated[Micros, Field(ge=0)]
    trace_ref: Id
    trace_refs: dict[Id, Id] = Field(min_length=1)
    cached: bool = False
    error: StepError | None = None

    @model_validator(mode="after")
    def _trace_ref_is_a_sink_ref(self) -> Self:
        if self.trace_ref not in self.trace_refs.values():
            raise ValueError(f"trace_ref {self.trace_ref} is not among trace_refs")
        return self
