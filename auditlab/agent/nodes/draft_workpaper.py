"""draft_workpaper (model): prose with [[ev:id]] and [[item:id]] markers; the guard runs after."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from auditlab.agent.deps import deps_from
from auditlab.agent.nodes.common import call_model, upstream_failed
from auditlab.agent.state import AgentState
from auditlab.domain.enums import Node
from auditlab.domain.models import GuardFinding, Workpaper
from auditlab.guard import validate_prose


def draft_workpaper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    if upstream_failed(state):
        return {}
    if state.samples is None or state.evidence is None or state.exceptions is None:
        raise RuntimeError("draft_workpaper ran before the earlier nodes")
    if state.tieouts is None:
        raise RuntimeError("draft_workpaper ran before tie_out")
    payload = {
        "item_ids": list(state.samples.item_ids),
        "evidence": state.evidence.model_dump(mode="json"),
        "tieouts": state.tieouts.model_dump(mode="json"),
        "exceptions": state.exceptions.model_dump(mode="json"),
    }
    draft, step = call_model(state, deps_from(config), Node.DRAFT_WORKPAPER, payload, Workpaper)
    if draft is None:
        return {"step_records": [step]}
    violations = validate_prose(
        draft.prose,
        frozenset(f.evidence_id for f in state.evidence.fields),
        item_ids=frozenset(state.samples.item_ids),
    )
    workpaper = Workpaper(
        prose=draft.prose,
        guard_violations=[GuardFinding(code=v.code, message=v.message) for v in violations],
    )
    return {"workpaper": workpaper, "step_records": [step]}
