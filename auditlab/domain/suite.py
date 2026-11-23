"""Eval suite files (SPEC.md §4.6): `evals/suites/<name>.json`."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from auditlab.domain.base import DomainModel, Id, Text, require_unique


class SuiteCase(DomainModel):
    case_id: Id
    seed: int = Field(ge=1)
    mix: Text
    tags: list[Text]


class Suite(DomainModel):
    name: Id
    cases: list[SuiteCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        require_unique("case_id", [c.case_id for c in self.cases])
        return self
