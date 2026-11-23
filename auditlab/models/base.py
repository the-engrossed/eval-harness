"""The model-swap interface (SPEC.md §6.1). Nodes talk to a ModelClient, never a provider SDK."""

from __future__ import annotations

import json
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from auditlab.domain.enums import Node


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ModelRequest(BaseModel):
    """One call to a model for one node.

    `case_id` is metadata only: it routes the stub to its answer key and is part of the
    cache key, but it must never appear in `system` or `payload`, the text a model sees.
    A case id such as s1-mixD names the mix, and the mix names the planted exception
    types. Construction fails if the case id appears in either.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    node: Node
    prompt_version: str = Field(min_length=1)
    system: str
    payload: dict[str, JsonValue]
    output_schema: type[BaseModel]
    run_index: int = Field(ge=0)
    seed: int

    @model_validator(mode="after")
    def _case_id_stays_out_of_prompt_text(self) -> Self:
        if self.case_id in self.system or self.case_id in self.payload_json():
            raise ValueError(
                f"case_id {self.case_id!r} appears in system or payload; "
                "it is metadata and must not be sent to a model"
            )
        return self

    def payload_json(self) -> str:
        return canonical_json(self.payload)

    def canonical_input(self) -> str:
        return canonical_json(
            {
                "case_id": self.case_id,
                "system": self.system,
                "payload": self.payload,
                "output_schema": self.output_schema.__name__,
            }
        )


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    output: BaseModel
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    provider_model_id: str = Field(min_length=1)
    cached: bool
    simulated: bool


class ModelClient(Protocol):
    provider: str
    model_id: str
    simulated: bool

    def complete(self, request: ModelRequest) -> ModelResponse: ...
