"""Paired per-case deltas and the seeded bootstrap CI (SPEC.md §11.1). Pure."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from evals.gate_models import GateInput
from evals.scorers import LOWER_IS_BETTER


def bootstrap_ci(
    deltas: Sequence[float], resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    """Percentile bootstrap of the mean, resampling cases with replacement."""
    if not deltas:
        raise ValueError("bootstrap of no deltas")
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(resamples))
    alpha = (1.0 - confidence) / 2
    low = means[max(0, math.ceil(alpha * resamples) - 1)]
    high = means[min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1)]
    return low, high


def orient(metric: str, value: float) -> float:
    return 1.0 - value if metric.split(".", 1)[1] in LOWER_IS_BETTER else value


def paired_deltas(metric: str, cand: GateInput, base: GateInput) -> dict[str, float]:
    c, b = cand.case_means.get(metric, {}), base.case_means.get(metric, {})
    return {k: orient(metric, c[k]) - orient(metric, b[k]) for k in sorted(c.keys() & b.keys())}


def mean_over(means: dict[str, float], cases: Sequence[str]) -> float:
    return sum(means[c] for c in cases) / len(cases)
