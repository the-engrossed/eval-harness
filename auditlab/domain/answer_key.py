"""Answer key (SPEC.md §4.5). Only the stub model, evals/ and feedback/ may import this."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from auditlab.domain.base import DomainModel, Id, Text, require_unique
from auditlab.domain.enums import CRITICAL_TYPES, JUDGMENT_TYPES, ExceptionType, JudgmentOutcome
from auditlab.domain.models import CitedValue, TieOut


class ExpectedEvidence(CitedValue):
    pass


ExpectedTieOut = TieOut


class ExpectedException(DomainModel):
    item_id: Id
    type: ExceptionType
    critical: bool

    @model_validator(mode="after")
    def _critical_matches_type(self) -> Self:
        if self.critical != (self.type in CRITICAL_TYPES):
            raise ValueError(f"critical={self.critical} is wrong for {self.type}")
        return self


class JudgmentItem(DomainModel):
    item_id: Id
    type: ExceptionType
    allowed_outcomes: list[JudgmentOutcome] = Field(min_length=1)
    rubric: list[Text] = Field(min_length=1)

    @model_validator(mode="after")
    def _is_judgment(self) -> Self:
        if self.type not in JUDGMENT_TYPES:
            raise ValueError(f"{self.type} is not a judgment type")
        return self


class AnswerKey(DomainModel):
    required_sample_ids: list[Id] = Field(min_length=1)
    evidence: list[ExpectedEvidence]
    expected_tieouts: list[ExpectedTieOut]
    expected_exceptions: list[ExpectedException]
    judgment_items: list[JudgmentItem]

    @model_validator(mode="after")
    def _items_inside_required_sample(self) -> Self:
        required = require_unique("required_sample_id", self.required_sample_ids)
        items = [
            *(e.item_id for e in self.evidence),
            *(t.item_id for t in self.expected_tieouts),
            *(x.item_id for x in self.expected_exceptions),
            *(j.item_id for j in self.judgment_items),
        ]
        for item_id in items:
            if item_id not in required:
                raise ValueError(f"not in required_sample_ids: {item_id}")
        return self
