"""Scorecard rows: heatmap cells, per-config numbers and gate details, formatted for reading.

Pure: everything comes from the summary, results and gate results passed in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from auditlab.domain.enums import Node, Verdict
from evals.gate_models import GateResult
from evals.results import ConfigSummary, RunResult, Summary
from evals.summary import metric_key
from report.labels import MEASURES, failure_phrase, metric_label

NODE_ORDER = {n.value: i for i, n in enumerate(Node)}
WARN_DROP = 0.05
MINUS = "−"


def usd(micros: int | None) -> str:
    """Three significant figures under $1 ($0.180, $0.0360), cents above."""
    if micros is None:
        return "—"
    value = micros / 1_000_000
    if value == 0:
        return "$0"
    return f"${value:,.2f}" if value >= 1 else f"${value:#.3g}"


def vs_base(micros: int | None, base: int | None) -> str:
    if micros is None or not base:
        return "—"
    pct = round((micros - base) * 100 / base)
    return "0%" if pct == 0 else f"+{pct}%" if pct > 0 else f"{MINUS}{-pct}%"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def heatmap(
    summary: Summary, results: Sequence[RunResult], critical: Sequence[str]
) -> list[dict[str, Any]]:
    kinds = {metric_key(s): s.kind.value for r in results for s in r.scores}
    base = summary.configs[0]
    rows = []
    for key in sorted(critical, key=lambda k: NODE_ORDER[k.split(".")[0]]):
        cells = []
        for c in summary.configs:
            acc, ref = c.accuracy.get(key), base.accuracy.get(key)
            if acc is None:
                cells.append({"text": "—", "cls": "na", "delta": ""})
                continue
            delta = acc.value - ref.value if ref is not None else 0.0
            if c is base:
                cls = "base"
            else:
                cls = "ok" if delta >= -1e-9 else "warn" if delta > -WARN_DROP else "bad"
            shown = "" if c is base else f"{delta:+.2f}".replace("-", MINUS)
            cells.append({"text": f"{acc.value:.2f}", "cls": cls, "delta": shown})
        if all(cell["cls"] == "na" for cell in cells):
            continue  # e.g. the feedback metric before any reviewer rejection exists
        rows.append(
            {
                "metric": key,
                "label": metric_label(key),
                "kind": kinds.get(key, "process"),
                "saturated": key in summary.saturated_metrics,
                "cells": cells,
            }
        )
    return [r for r in rows if r["kind"] == "process"] + [r for r in rows if r["kind"] == "outcome"]


def config_row(c: ConfigSummary, base: ConfigSummary, judge_simulated: bool) -> dict[str, Any]:
    cost = c.mean_cost_micros_per_workpaper
    judge = (
        "— (simulated judge)" if judge_simulated else usd(c.judge_mean_cost_micros_per_workpaper)
    )
    return {
        "config_id": c.config_id,
        "simulated": c.simulated,
        "missed": f"{c.missed_by_final_only} of {c.runs} runs",
        "cost": usd(cost),
        "vs_base": "—" if c is base else vs_base(cost, base.mean_cost_micros_per_workpaper),
        "judge_cost": judge,
        "p50": "—" if c.latency_p50_ms is None else f"{c.latency_p50_ms} ms",
        "p95": "—" if c.latency_p95_ms is None else f"{c.latency_p95_ms} ms",
        "pass_pow_n": _num(c.pass_pow_n_share),
        "agreement": _num(c.outcome_agreement),
        "infra_rate": f"{c.infra_failure_rate:.2f}",
        "infra_runs": c.infra_failed_runs,
        "data_rate": f"{c.data_failure_rate:.2f}",
        "data_runs": c.data_failed_runs,
    }


def details(gate: Sequence[GateResult]) -> list[dict[str, Any]]:
    """Every finding of every non-SHIP candidate, with the full case and trace lists."""
    out = []
    for g in gate:
        if g.verdict is Verdict.SHIP:
            continue
        items = []
        for f in g.findings:
            if f.metric is not None:
                title, code = metric_label(f.metric), f.metric
            else:
                title, code = MEASURES.get(f.measure or "", (f.check.value, ""))[0], f.measure or ""
            change = ""
            if f.baseline_value is not None and f.candidate_value is not None:
                change = f"{f.baseline_value:.2f} → {f.candidate_value:.2f}"
            root = None
            if f.root is not None:
                root = {
                    "label": failure_phrase(f.root.metric, f.root.detail),
                    "code": f.root.metric,
                    "change": f"{f.root.baseline_value:.2f} → {f.root.candidate_value:.2f}",
                    "detail": f.root.detail,
                }
            items.append(
                {
                    "title": title,
                    "code": code,
                    "note": f.verdict is Verdict.SHIP,
                    "change": change,
                    "message": f.message,
                    "cases": f.case_ids,
                    "traces": f.trace_refs,
                    "root": root,
                }
            )
        out.append({"config_id": g.candidate_config_id, "verdict": g.verdict.value, "items": items})
    return out


def trace_summary(g: GateResult) -> dict[str, Any]:
    refs = list(
        dict.fromkeys(t for f in g.findings if f.verdict is not Verdict.SHIP for t in f.trace_refs)
    )
    return {"first": refs[0] if refs else "", "more": max(0, len(refs) - 1)}
