"""flag_exceptions (model): typed exceptions and judgment decisions from evidence and tie-outs."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from auditlab.agent.deps import deps_from
from auditlab.agent.nodes.common import call_model, upstream_failed
from auditlab.agent.state import AgentState
from auditlab.domain.enums import Node
from auditlab.domain.models import ExceptionSet


def flag_exceptions(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    if upstream_failed(state):
        return {}
    if state.evidence is None or state.tieouts is None:
        raise RuntimeError("flag_exceptions ran before extract_evidence and tie_out")
    payload = {
        "evidence": state.evidence.model_dump(mode="json"),
        "tieouts": state.tieouts.model_dump(mode="json"),
    }
    output, step = call_model(state, deps_from(config), Node.FLAG_EXCEPTIONS, payload, ExceptionSet)
    if output is None:
        return {"step_records": [step]}
    return {"exceptions": output, "step_records": [step]}
