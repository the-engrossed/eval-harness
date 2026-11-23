"""select_samples (model): sees only the population summary and the sampling rule text."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from auditlab.agent.deps import deps_from
from auditlab.agent.nodes.common import call_model, upstream_failed
from auditlab.agent.state import AgentState
from auditlab.domain.enums import Node
from auditlab.domain.models import SampleSelection
from auditlab.domain.population import Population


def population_summary(pop: Population) -> dict[str, Any]:
    return {
        "invoices": [
            {
                "invoice_id": i.invoice_id,
                "customer_id": i.customer_id,
                "invoice_date": i.invoice_date.isoformat(),
                "amount_cents": i.amount_cents,
            }
            for i in pop.invoices
        ],
        "credit_memo_invoice_ids": sorted({c.invoice_id for c in pop.credit_memos}),
    }


def select_samples(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    if upstream_failed(state):
        return {}
    payload = {
        "population_summary": population_summary(state.population),
        "sampling_rule": state.sampling_rule,
    }
    output, step = call_model(
        state, deps_from(config), Node.SELECT_SAMPLES, payload, SampleSelection
    )
    if output is None:
        return {"step_records": [step]}
    return {"samples": output, "step_records": [step]}
