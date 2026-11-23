"""`python -m auditlab.cli <command>`."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from dotenv import load_dotenv

from auditlab.agent.graph import run_case
from auditlab.agent.state import AgentState
from auditlab.config import DEMO_CACHE_DIR, ENGAGEMENTS_DIR, MODELS_TOML, RUNS_DIR
from auditlab.models.config import load_models_config
from auditlab.models.registry import build_client
from auditlab.money import format_cents

_MARKER = re.compile(r"\[\[(ev|item):([A-Za-z0-9_.-]+)\]\]")


def resolve_markers(state: AgentState) -> str:
    if state.workpaper is None or state.evidence is None:
        raise RuntimeError("run finished without a workpaper")
    values = {
        f.evidence_id: format_cents(f.value_cents)
        if f.value_cents is not None
        else str(f.value_text)
        for f in state.evidence.fields
    }

    def repl(match: re.Match[str]) -> str:
        kind, ident = match.groups()
        return ident if kind == "item" else values.get(ident, match.group(0))

    return _MARKER.sub(repl, state.workpaper.prose)


def _run_case(args: argparse.Namespace) -> int:
    config = load_models_config(args.models_toml)
    model = config.model(args.model)
    client = build_client(
        args.model, config, cache_dir=DEMO_CACHE_DIR, engagements_dir=args.engagements_dir
    )
    price = next((p for p in config.prices if p.model_id == model.model_id), None)
    run_id = f"cli-{args.case_id}-{args.model}-r{args.run_index}"
    spans = args.runs_dir / run_id / "spans.jsonl"
    spans.unlink(missing_ok=True)
    state = run_case(
        args.case_id,
        client=client,
        model_config_id=args.model,
        engagements_dir=args.engagements_dir,
        runs_dir=args.runs_dir,
        run_id=run_id,
        run_index=args.run_index,
        seed=args.seed,
        prompt_version=model.prompt_version,
        price=price,
    )
    label = " | SIMULATED" if client.simulated else ""
    print(f"{args.case_id} | model {args.model} ({model.model_id}) | run {args.run_index}{label}")
    print()
    for line in step_lines(state):
        print(line)
    print()
    print(f"spans: {spans}")
    if state.workpaper is None:
        print("no workpaper: a node failed (see the steps above)")
        return 1
    print()
    print(resolve_markers(state))
    print()
    violations = state.workpaper.guard_violations
    print(f"guard violations: {len(violations)}")
    for v in violations:
        print(f"  {v}")
    return 0


def step_lines(state: AgentState) -> list[str]:
    """One line per node: latency, tokens, and the error if the node failed."""
    lines = []
    for r in state.step_records:
        status = "ok" if r.error is None else f"FAILED ({r.error.origin.value}): {r.error.message}"
        lines.append(
            f"  {r.node.value:<17} {r.latency_ms:>7} ms  in {r.tokens_in:>6}  out {r.tokens_out:>5}"
            f"  {status}"
        )
    if state.samples is not None:
        lines.append(f"  samples selected: {len(state.samples.item_ids)}")
    if state.evidence is not None:
        dropped = len(state.evidence.dropped)
        lines.append(
            f"  evidence fields: {len(state.evidence.fields)}, dropped (quote not found): {dropped}"
        )
    if state.exceptions is not None:
        ex = state.exceptions
        lines.append(f"  exceptions flagged: {len(ex.exceptions)}, judgments: {len(ex.judgments)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m auditlab.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-case", help="run the agent on one case and print the workpaper")
    run.add_argument("case_id")
    run.add_argument("--model", default="prod")
    run.add_argument("--run-index", type=int, default=0)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--engagements-dir", type=Path, default=ENGAGEMENTS_DIR)
    run.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    run.add_argument("--models-toml", type=Path, default=MODELS_TOML)
    args = parser.parse_args(argv)
    if args.command == "run-case":
        return _run_case(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
