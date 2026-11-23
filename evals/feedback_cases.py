"""Feedback-derived eval cases (SPEC.md §12): a reviewer's corrected expectation on one item.

A feedback case reruns its source engagement and is scored only on
`flag_exceptions.feedback_expectation_met`. Types and the loader only; no I/O beyond reading.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from auditlab.domain.base import DomainModel, Id, Text, require_unique
from auditlab.domain.enums import JUDGMENT_TYPES, ExceptionType, JudgmentOutcome, Node

FEEDBACK_TAG = "from_feedback"
FEEDBACK_PREFIX = "fb-"


class Expectation(DomainModel):
    """The corrected outcome: the item must be flagged with this type (and, for a judgment
    type, decided with this outcome)."""

    item_id: Id
    type: ExceptionType
    outcome: JudgmentOutcome | None = None

    @model_validator(mode="after")
    def _outcome_matches_type(self) -> Self:
        judgment = self.type in JUDGMENT_TYPES
        if judgment and self.outcome is None:
            raise ValueError(f"{self.type.value} is a judgment type: outcome is required")
        if not judgment and self.outcome is not None:
            raise ValueError(f"{self.type.value} is not a judgment type: omit outcome")
        return self


class Provenance(DomainModel):
    trace_ref: Id
    reviewer: Text
    reason: Text
    created_at: dt.datetime


class FeedbackCase(DomainModel):
    case_id: Id = Field(pattern=r"^fb-[0-9a-f]{10}$")
    source_case_id: Id
    source_run_id: Id
    source_config_id: Id
    source_run_index: int = Field(ge=0)
    node: Literal[Node.FLAG_EXCEPTIONS]
    item_id: Id
    expected: Expectation
    provenance: Provenance
    critical: Literal[True] = True
    tags: list[Text] = [FEEDBACK_TAG]

    @model_validator(mode="after")
    def _item_is_the_expected_item(self) -> Self:
        if self.item_id != self.expected.item_id:
            raise ValueError("item_id must equal expected.item_id")
        return self


def load_feedback_cases(directory: Path) -> list[FeedbackCase]:
    if not directory.is_dir():
        return []
    cases = [
        FeedbackCase.model_validate_json(p.read_text("utf-8"))
        for p in sorted(directory.glob("*.json"))
    ]
    require_unique("feedback case_id", [c.case_id for c in cases])
    return cases
