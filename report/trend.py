"""Trend series for the scorecard: one headline metric per config across runs/history.jsonl.

Only history lines for the same suite hash, up to and including this run, in time order.
SVG coordinates are computed here so the template only draws them.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

TREND_METRIC = "flag_exceptions.planted_recall"
WIDTH, HEIGHT, PAD = 180, 44, 4


def _history(path: Path, suite_hash: str, until: dt.datetime) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    kept = [
        h
        for h in lines
        if h.get("suite_hash") == suite_hash and dt.datetime.fromisoformat(h["created_at"]) <= until
    ]
    return sorted(kept, key=lambda h: (h["created_at"], h["run_id"]))


def _points(values: Sequence[float]) -> str:
    step = (WIDTH - 2 * PAD) / (len(values) - 1)
    return " ".join(
        f"{PAD + i * step:.1f},{PAD + (1.0 - v) * (HEIGHT - 2 * PAD):.1f}"
        for i, v in enumerate(values)
    )


def trend_series(
    path: Path, suite_hash: str, until: dt.datetime, config_ids: Sequence[str]
) -> list[dict[str, Any]]:
    history = _history(path, suite_hash, until)
    out = []
    for config_id in config_ids:
        values = [
            float(h["configs"][config_id]["accuracy"][TREND_METRIC])
            for h in history
            if TREND_METRIC in h.get("configs", {}).get(config_id, {}).get("accuracy", {})
        ]
        out.append(
            {
                "config_id": config_id,
                "n": len(values),
                "points": _points(values) if len(values) >= 2 else "",
                "first": f"{values[0]:.2f}" if values else "—",
                "last": f"{values[-1]:.2f}" if values else "—",
            }
        )
    return out


__all__ = ["HEIGHT", "TREND_METRIC", "WIDTH", "trend_series"]
