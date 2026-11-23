from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from auditlab.money import Cents

Id = Annotated[str, Field(min_length=1)]
Text = Annotated[str, Field(min_length=1)]
Amount = Annotated[Cents, Field(ge=0)]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def require_unique(label: str, values: list[str]) -> set[str]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label}: {value}")
        seen.add(value)
    return seen
