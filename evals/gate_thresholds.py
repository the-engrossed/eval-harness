"""Absolute gate thresholds (SPEC.md §11): consistency and infra. Pure."""

from __future__ import annotations

from auditlab.domain.enums import Verdict
from evals.gate_models import Check, Finding, GateConfig, GateInput


def thresholds(cand: GateInput, config: GateConfig) -> list[Finding]:
    out: list[Finding] = []
    checks = [
        (
            "pass_pow_n_share",
            "pass^N share",
            cand.pass_pow_n_share,
            "min_pass_pow_n",
            config.consistency.min_pass_pow_n,
        ),
        (
            "outcome_agreement",
            "outcome agreement",
            cand.outcome_agreement,
            "min_outcome_agreement",
            config.consistency.min_outcome_agreement,
        ),
    ]
    for measure, label, value, name, minimum in checks:
        if value is None:
            out.append(
                Finding(
                    check=Check.CONSISTENCY,
                    verdict=Verdict.NEEDS_REVIEW,
                    message=f"{label}: no eligible case (all had infra or data faults)",
                    measure=measure,
                )
            )
        elif value < minimum:
            out.append(
                Finding(
                    check=Check.CONSISTENCY,
                    verdict=Verdict.NEEDS_REVIEW,
                    message=f"{label} {value:.2f} below {name} {minimum:.2f}",
                    measure=measure,
                    candidate_value=value,
                )
            )
    limit = config.stats.max_infra_failure_rate
    if cand.infra_failure_rate > limit:
        out.append(
            Finding(
                check=Check.INFRA,
                verdict=Verdict.NEEDS_REVIEW,
                message=f"infra failure rate {cand.infra_failure_rate:.2f} above "
                f"max_infra_failure_rate {limit:.2f}",
                measure="infra_failure_rate",
                candidate_value=cand.infra_failure_rate,
            )
        )
    return out
