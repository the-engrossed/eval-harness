"""Gate and baseline commands (SPEC.md §11). All the I/O around the pure evals/gate.py.

`python -m evals.gate_cli gate --run <run_id> [--baseline <file>]` gates every non-baseline
config of the run against the [production] config of the same run (or against a baseline
file) and exits 1 if any candidate is DO_NOT_SHIP. `--report-only` prints the same output
and always exits 0 (`make demo` uses it; `make gate` does not).
`python -m evals.gate_cli baseline --run <run_id>` writes evals/baselines/production.json
from that run's production config. Only this command writes a baseline; the gate reads it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from auditlab.config import CONFIG_DIR, REPO_ROOT, RUNS_DIR
from auditlab.domain.enums import Verdict
from evals.gate import evaluate_gate, gate_input
from evals.gate_models import Check, Finding, GateInput, GateResult, load_gate_config
from evals.gate_text import root_cause_lines, root_causes
from evals.results import Manifest, RunResult, Summary

GATE_TOML = CONFIG_DIR / "gate.toml"
BASELINE = REPO_ROOT / "evals" / "baselines" / "production.json"
SHOW_CASES = 5
SHOW_TRACES = 3


def load_run(run_dir: Path) -> tuple[Manifest, dict[str, GateInput]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run {run_dir} not found")
    manifest = Manifest.model_validate_json((run_dir / "manifest.json").read_text("utf-8"))
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text("utf-8"))
    with (run_dir / "results.jsonl").open(encoding="utf-8") as handle:
        results = [RunResult.model_validate_json(line) for line in handle if line.strip()]
    inputs = {
        c.config_id: gate_input(
            c,
            [r for r in results if r.model_config_id == c.config_id],
            manifest.run_id,
            summary.suite_hash,
        )
        for c in summary.configs
    }
    return manifest, inputs


def _cases(ids: list[str]) -> str:
    shown = ", ".join(ids[:SHOW_CASES])
    return shown + (f" ... (+{len(ids) - SHOW_CASES} more)" if len(ids) > SHOW_CASES else "")


def _finding(f: Finding) -> list[str]:
    if f.metric is None:
        label = "infra" if f.check is Check.INFRA else "consistency"
        return [f"  {label}: {f.message}"]
    note = "note: " if f.verdict is Verdict.SHIP else ""
    head = f"  {note}{f.metric}"
    if f.baseline_value is not None and f.candidate_value is not None:
        head += f"  {f.baseline_value:.2f} -> {f.candidate_value:.2f}"
    if f.case_ids:
        head += f"  cases: {_cases(f.case_ids)}"
    lines = [head, f"    {f.message}"]
    if f.root is not None:
        r = f.root
        lines.append(
            f"    root step: {r.metric}  {r.baseline_value:.2f} -> {r.candidate_value:.2f}"
            f"  ({r.detail})"
        )
    if f.trace_refs:
        more = " ..." if len(f.trace_refs) > SHOW_TRACES else ""
        lines.append(f"    traces: {' '.join(f.trace_refs[:SHOW_TRACES])}{more}")
    return lines


def format_result(result: GateResult) -> str:
    badge = "  [SIMULATED]" if result.simulated else ""
    lines = [
        f"{result.candidate_config_id}  {result.verdict.value}"
        f"  (vs {result.baseline_config_id} from {result.baseline_source}){badge}"
    ]
    lines.extend(f"  root cause: {r}" for r in root_cause_lines(result))
    order = list(root_causes(result))

    def group(f: Finding) -> int:
        return order.index(f.root.metric) if f.root is not None else len(order)

    for f in sorted(result.findings, key=group):  # stable: graph order within a group
        lines.extend(_finding(f))
    return "\n".join(lines)


def run_gate(run_dir: Path, gate_toml: Path, baseline_file: Path | None) -> list[GateResult]:
    config = load_gate_config(gate_toml)
    manifest, inputs = load_run(run_dir)
    if baseline_file is not None:
        baseline = GateInput.model_validate_json(baseline_file.read_text("utf-8"))
    else:
        prod = manifest.production_config_id
        if prod not in inputs:
            raise ValueError(
                f"production config {prod!r} is not in run {manifest.run_id}; "
                "add it to MODELS or pass --baseline"
            )
        baseline = inputs[prod]
    candidates = [
        g for c, g in inputs.items() if baseline_file is not None or c != baseline.config_id
    ]
    for cand in candidates:
        shared = {c for m in cand.case_means.values() for c in m}
        if not shared & {c for m in baseline.case_means.values() for c in m}:
            raise ValueError(f"{cand.config_id} and the baseline share no cases")
    return [evaluate_gate(c, baseline, config) for c in candidates]


def write_baseline(run_dir: Path, out: Path) -> Path:
    manifest, inputs = load_run(run_dir)
    prod = manifest.production_config_id
    if prod not in inputs:
        raise ValueError(f"production config {prod!r} is not in run {manifest.run_id}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(inputs[prod].model_dump_json(indent=2) + "\n", "utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.gate_cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--run", required=True)
    g.add_argument("--baseline", type=Path)
    g.add_argument("--gate-toml", type=Path, default=GATE_TOML)
    g.add_argument("--report-only", action="store_true", help="print verdicts, always exit 0")
    b = sub.add_parser("baseline")
    b.add_argument("--run", required=True)
    b.add_argument("--out", type=Path, default=BASELINE)
    for parser in (g, b):
        parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = p.parse_args(argv)
    run_dir = args.runs_dir / args.run
    if args.cmd == "baseline":
        print(f"baseline written: {write_baseline(run_dir, args.out)}")
        return 0
    results = run_gate(run_dir, args.gate_toml, args.baseline)
    print("\n\n".join(format_result(r) for r in results))
    if args.report_only:
        return 0
    return 1 if any(r.verdict is Verdict.DO_NOT_SHIP for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
