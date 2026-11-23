"""AgentState (SPEC.md §5.2). Each node writes only its own key plus one StepRecord."""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, ConfigDict

from auditlab.domain.models import (
    EvidenceSet,
    ExceptionSet,
    SampleSelection,
    StepRecord,
    TieOutSet,
    Workpaper,
)
from auditlab.domain.population import Population


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    run_index: int
    model_config_id: str
    population: Population
    sampling_rule: str
    samples: SampleSelection | None = None
    evidence: EvidenceSet | None = None
    tieouts: TieOutSet | None = None
    exceptions: ExceptionSet | None = None
    workpaper: Workpaper | None = None
    step_records: Annotated[list[StepRecord], operator.add] = []
