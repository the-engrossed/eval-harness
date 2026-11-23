"""Build the gate's view of one config (SPEC.md §11) from its results.jsonl rows. Pure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from evals.gate_models import Failure, GateInput
from evals.results import ConfigSummary, RunResult
from evals.summary import metric_key

PLANTED = "flag_exceptions.planted_recall"
FEEDBACK = "flag_exceptions.feedback_expectation_met"


def gate_input(
    summary: ConfigSummary, results: Sequence[RunResult], run_id: str, suite_hash: str
) -> GateInput:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    failures: dict[str, list[Failure]] = defaultdict(list)
    planted: list[Failure] = []
    feedback: list[Failure] = []
    feedback_runs = 0
    for r in results:
        clean = not (r.infra_failed or r.data_failed)
        for s in r.scores:
            key = metric_key(s)
            if s.counts_toward_accuracy:
                values[key][r.case_id].append(s.value)
                if not s.passed:
                    failures[key].append(
                        Failure(
                            case_id=r.case_id,
                            run_index=r.run_index,
                            trace_ref=s.trace_ref,
                            detail=s.detail,
                        )
                    )
            if key in (PLANTED, FEEDBACK) and clean:
                feedback_runs += key == FEEDBACK
                if not s.passed:
                    miss = Failure(
                        case_id=r.case_id,
                        run_index=r.run_index,
                        trace_ref=s.trace_ref,
                        detail=s.detail,
                    )
                    (planted if key == PLANTED else feedback).append(miss)
    return GateInput(
        source_run_id=run_id,
        config_id=summary.config_id,
        simulated=summary.simulated,
        suite_hash=suite_hash,
        case_means={
            k: {c: sum(v) / len(v) for c, v in cases.items()} for k, cases in values.items()
        },
        failures=dict(failures),
        planted_misses=planted,
        feedback_misses=feedback,
        feedback_runs=feedback_runs,
        pass_pow_n_share=summary.pass_pow_n_share,
        outcome_agreement=summary.outcome_agreement,
        infra_failure_rate=summary.infra_failure_rate,
    )
