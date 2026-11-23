"""`python -m report.scorecard --run <run_id>` → out/scorecard.html (SPEC.md §13).

One static file: inline CSS and SVG, no JavaScript, no external links. Every number comes
from the run's artifacts (manifest, summary, results, the gate on them) and
runs/history.jsonl; the generated time is passed in so equal runs give equal bytes.
Trace refs render as text.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from auditlab.config import OUT_DIR, REPO_ROOT, RUNS_DIR
from evals.gate_cli import GATE_TOML, run_gate
from evals.gate_models import load_gate_config
from evals.results import Manifest, RunResult, Summary
from report.headline import headline
from report.labels import cause_lines
from report.rows import config_row, details, heatmap, trace_summary
from report.trend import HEIGHT, TREND_METRIC, WIDTH, trend_series

TEMPLATES = Path(__file__).parent / "templates"
FEEDBACK_DIR = REPO_ROOT / "evals" / "suites" / "from_feedback"
SIMULATED_BANNER = "All results below are from simulated models; no real model was called."
SATURATED_NOTE = "Saturated: every model scores ≥ 0.98, so this metric no longer separates models."


def build_context(
    run_dir: Path,
    history_path: Path,
    feedback_dir: Path,
    gate_toml: Path,
    generated_at: dt.datetime,
) -> dict[str, Any]:
    manifest = Manifest.model_validate_json((run_dir / "manifest.json").read_text("utf-8"))
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text("utf-8"))
    with (run_dir / "results.jsonl").open(encoding="utf-8") as handle:
        results = [RunResult.model_validate_json(line) for line in handle if line.strip()]
    prod = manifest.production_config_id
    configs = sorted(summary.configs, key=lambda c: c.config_id != prod)  # baseline first
    summary = summary.model_copy(update={"configs": configs})
    gate = run_gate(run_dir, gate_toml, None)
    simulated = {c.config_id: c.simulated for c in configs}
    all_simulated = all(simulated.values())
    verdicts = [
        {
            "config_id": g.candidate_config_id,
            "verdict": g.verdict.value,
            "root_causes": cause_lines(g),
            "simulated": simulated[g.candidate_config_id],
            "traces": trace_summary(g),
        }
        for g in gate
    ]
    critical = list(load_gate_config(gate_toml).critical)
    feedback = sorted(feedback_dir.glob("*.json")) if feedback_dir.is_dir() else []
    return {
        "headline": headline(gate),
        "baseline": prod,
        "verdicts": verdicts,
        "config_ids": [c.config_id for c in configs],
        "configs": [config_row(c, configs[0], summary.judge_simulated) for c in configs],
        "heatmap": heatmap(summary, results, critical),
        "details": details(gate),
        "simulated_banner": SIMULATED_BANNER if all_simulated else "",
        "show_badges": not all_simulated,
        "saturated_note": SATURATED_NOTE,
        "saturation_assessed": summary.saturation_assessed,
        "judge_simulated": summary.judge_simulated,
        "trend_metric": TREND_METRIC,
        "svg_w": WIDTH,
        "svg_h": HEIGHT,
        "trends": trend_series(
            history_path, summary.suite_hash, manifest.created_at, [c.config_id for c in configs]
        ),
        "feedback_cases": len(feedback),
        "run_id": manifest.run_id,
        "git_sha": manifest.git_sha[:12] + (" (dirty)" if manifest.git_dirty else ""),
        "suite": f"{summary.suite} ({len(summary.case_ids)} cases, N={summary.n})",
        "suite_hash": summary.suite_hash,
        "judge_id": summary.judge_id,
        "generated_at": generated_at.isoformat(timespec="seconds"),
    }


def render(context: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=True,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template("scorecard.html.j2").render(**context)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m report.scorecard")
    p.add_argument("--run", required=True)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--out", type=Path, default=OUT_DIR / "scorecard.html")
    p.add_argument("--gate-toml", type=Path, default=GATE_TOML)
    p.add_argument("--feedback-dir", type=Path, default=FEEDBACK_DIR)
    args = p.parse_args(argv)
    context = build_context(
        args.runs_dir / args.run,
        args.runs_dir / "history.jsonl",
        args.feedback_dir,
        args.gate_toml,
        dt.datetime.now(dt.UTC),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(context), "utf-8")
    print(f"scorecard: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
