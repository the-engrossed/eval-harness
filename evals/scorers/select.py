"""select_samples: required_recall (critical), precision.

A miss names the sampling rules (SPEC.md §4.3) its missing ids fall under, from
data/sampling.py, the same code that builds the answer key's required ids.
"""

from __future__ import annotations

from auditlab.agent.state import AgentState
from auditlab.domain.answer_key import AnswerKey
from auditlab.domain.enums import MetricKind, Node
from auditlab.domain.population import Population
from auditlab.domain.scores import StepScore
from data.sampling import credit_memo_items, cutoff_window, key_items
from evals.scorers.common import (
    Metric,
    ids_text,
    make_score,
    node_status,
    output_of,
    share,
    unscored,
)

REQUIRED_RECALL = Metric("required_recall", MetricKind.PROCESS, critical=True)
PRECISION = Metric("precision", MetricKind.PROCESS, critical=False)
RULES = (
    ("key_item", key_items),
    ("cutoff_window", cutoff_window),
    ("credit_memo", credit_memo_items),
)


def missing_rules(missing: set[str], pop: Population) -> list[str]:
    """Names of the sampling rules that select at least one missing id, in rule order."""
    return [name for name, rule in RULES if missing & rule(pop)]


def score_select(state: AgentState, key: AnswerKey) -> list[StepScore]:
    status = node_status(state, Node.SELECT_SAMPLES)
    if (early := unscored(state, status, [REQUIRED_RECALL, PRECISION])) is not None:
        return early
    selected = set(output_of(state.samples, status).item_ids)
    required = set(key.required_sample_ids)
    missing, extra = required - selected, selected - required
    hit = len(required & selected)
    return [
        make_score(
            state,
            status,
            REQUIRED_RECALL,
            share(hit, len(required)),
            f"missing required ids (rules: {', '.join(missing_rules(missing, state.population))}): "
            f"{ids_text(missing)}"
            if missing
            else "all required ids selected",
        ),
        make_score(
            state,
            status,
            PRECISION,
            share(hit, len(selected)),
            f"selected but not required: {ids_text(extra)}" if extra else "no extra ids selected",
        ),
    ]
