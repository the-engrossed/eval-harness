"""Per-run dependencies, passed to nodes through LangGraph's per-call config."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig

from auditlab.docs.store import DocStore
from auditlab.models.base import ModelClient
from auditlab.models.config import Price
from auditlab.telemetry import RunTelemetry


@dataclass(frozen=True)
class RunDeps:
    client: ModelClient
    docs: DocStore
    price: Price | None
    telemetry: RunTelemetry
    run_id: str
    seed: int
    prompt_version: str


def deps_from(config: RunnableConfig) -> RunDeps:
    deps = config.get("configurable", {}).get("deps")
    if not isinstance(deps, RunDeps):
        raise RuntimeError("graph invoked without RunDeps in config['configurable']['deps']")
    return deps
