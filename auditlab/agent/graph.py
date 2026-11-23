"""The five-node linear graph (SPEC.md §5.1). Compiled once; invoked per run."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from auditlab.agent.deps import RunDeps
from auditlab.agent.nodes.draft_workpaper import draft_workpaper
from auditlab.agent.nodes.extract_evidence import extract_evidence
from auditlab.agent.nodes.flag_exceptions import flag_exceptions
from auditlab.agent.nodes.select_samples import select_samples
from auditlab.agent.nodes.tie_out import tie_out
from auditlab.agent.prompts import load_prompt
from auditlab.agent.state import AgentState
from auditlab.docs.store import DocStore
from auditlab.domain.enums import Node
from auditlab.domain.population import Population
from auditlab.models.base import ModelClient
from auditlab.models.config import Price
from auditlab.telemetry import open_run, seeded_ids, without_sinks


def _build() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node(Node.SELECT_SAMPLES.value, select_samples)
    graph.add_node(Node.EXTRACT_EVIDENCE.value, extract_evidence)
    graph.add_node(Node.TIE_OUT.value, tie_out)
    graph.add_node(Node.FLAG_EXCEPTIONS.value, flag_exceptions)
    graph.add_node(Node.DRAFT_WORKPAPER.value, draft_workpaper)
    order = list(Node)
    graph.add_edge(START, order[0].value)
    for upstream, downstream in zip(order, order[1:], strict=False):
        graph.add_edge(upstream.value, downstream.value)
    graph.add_edge(order[-1].value, END)
    return graph.compile()


GRAPH = _build()


def run_case(
    case_id: str,
    *,
    client: ModelClient,
    model_config_id: str,
    engagements_dir: Path,
    runs_dir: Path,
    run_id: str,
    run_index: int = 0,
    seed: int = 7,
    prompt_version: str = "v1",
    price: Price | None = None,
    telemetry_key: str | None = None,
    id_seed: str | None = None,
) -> AgentState:
    """Run the graph once. Spans go to runs_dir/run_id/spans.jsonl.

    `telemetry_key` registers this agent run with the span router under its own key, so
    concurrent runs can share one run_id; `id_seed` makes its span ids reproducible.
    """
    case_dir = engagements_dir / case_id
    population_path = case_dir / "population.json"
    if not population_path.is_file():
        raise FileNotFoundError(f"{population_path} not found. Run `make data` first.")
    population = Population.model_validate_json(population_path.read_text(encoding="utf-8"))
    initial = AgentState(
        case_id=case_id,
        run_index=run_index,
        model_config_id=model_config_id,
        population=population,
        sampling_rule=load_prompt(prompt_version, "sampling_rule"),
    )
    ids = seeded_ids(id_seed) if id_seed is not None else nullcontext()
    with ids, open_run(telemetry_key or run_id, runs_dir / run_id / "spans.jsonl") as run:
        deps = RunDeps(
            client=client,
            docs=DocStore(case_dir),
            price=price,
            telemetry=run,
            run_id=run_id,
            seed=seed,
            prompt_version=prompt_version,
        )
        with run.span("agent_run") as root:
            root.set_attribute("audit.case_id", case_id)
            root.set_attribute("audit.run_index", run_index)
            root.set_attribute("audit.model_config_id", model_config_id)
            result = GRAPH.invoke(initial, config={"configurable": {"deps": deps}})
        failed = run.finish(root.get_span_context().trace_id)
    state = AgentState.model_validate(result)
    if failed:
        state = state.model_copy(update={"step_records": without_sinks(state.step_records, failed)})
    return state
