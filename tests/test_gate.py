"""The release gate (SPEC.md §11, §11.1): verdicts, paired bootstrap, root step, CLI."""

import json
from pathlib import Path

import pytest
from matrix_run import CASE_IDS, RUN_ID

from auditlab.config import CONFIG_DIR
from auditlab.domain.enums import Verdict
from evals.gate import bootstrap_ci, evaluate_gate
from evals.gate_cli import format_result, main, run_gate
from evals.gate_models import Check, Finding, GateInput, GateResult, RootStep, load_gate_config

CONFIG = load_gate_config(CONFIG_DIR / "gate.toml")


@pytest.fixture(scope="module")
def run_dir(matrix_run: Path) -> Path:
    return matrix_run


@pytest.fixture(scope="module")
def results(run_dir: Path) -> dict[str, GateResult]:
    return {r.candidate_config_id: r for r in run_gate(run_dir, CONFIG_DIR / "gate.toml", None)}


def test_prod_is_the_baseline_and_not_a_candidate(results: dict[str, GateResult]) -> None:
    assert set(results) == {"stub-cand-good", "stub-cand-cutoff", "stub-cand-noisy"}
    assert {r.baseline_config_id for r in results.values()} == {"prod"}


def test_cand_good_ships_and_costs_less(run_dir: Path, results: dict[str, GateResult]) -> None:
    good = results["stub-cand-good"]
    assert good.verdict is Verdict.SHIP
    assert all(f.verdict is Verdict.SHIP for f in good.findings)
    configs = {
        c["config_id"]: c for c in json.loads((run_dir / "summary.json").read_text())["configs"]
    }
    good_cost = configs["stub-cand-good"]["mean_cost_micros_per_workpaper"]
    assert 0 < good_cost < configs["prod"]["mean_cost_micros_per_workpaper"]


def test_cand_cutoff_is_do_not_ship_rooted_at_select_samples(
    results: dict[str, GateResult],
) -> None:
    cutoff = results["stub-cand-cutoff"]
    assert cutoff.verdict is Verdict.DO_NOT_SHIP
    [planted] = [f for f in cutoff.findings if f.check is Check.PLANTED_MISS]
    assert planted.verdict is Verdict.DO_NOT_SHIP
    assert "s1-mixA" in planted.case_ids and planted.trace_refs
    assert all(t.startswith(f"local:{RUN_ID}/") for t in planted.trace_refs)
    assert planted.root is not None
    assert planted.root.metric == "select_samples.required_recall"
    assert planted.root.candidate_value < planted.root.baseline_value
    assert planted.root.detail.startswith("missing required ids (rules: cutoff_window): ")
    for f in cutoff.findings:
        if f.check is Check.CRITICAL_DROP:
            assert f.verdict is Verdict.DO_NOT_SHIP
            assert f.root is not None and f.root.metric == "select_samples.required_recall"


def test_cand_noisy_needs_review_for_outcome_agreement_only(results: dict[str, GateResult]) -> None:
    noisy = results["stub-cand-noisy"]
    assert noisy.verdict is Verdict.NEEDS_REVIEW
    moving = [f for f in noisy.findings if f.verdict is not Verdict.SHIP]
    assert [f.check for f in moving] == [Check.CONSISTENCY]
    assert moving[0].message.startswith("outcome agreement")


def test_cli_prints_the_section_11_format_and_exits_1(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["gate", "--run", RUN_ID, "--runs-dir", str(run_dir.parent)])
    out = capsys.readouterr().out
    assert code == 1
    assert "stub-cand-cutoff  DO_NOT_SHIP" in out
    cutoff = out.split("stub-cand-cutoff  DO_NOT_SHIP")[1].splitlines()
    assert cutoff[1] == "  root cause: select_samples.required_recall (explains 4 critical drops)"
    noisy = out.split("stub-cand-noisy  NEEDS_REVIEW")[1].splitlines()
    assert noisy[1] == (
        "  root cause: inconsistent outcomes across runs "
        "(outcome agreement 0.73, no critical metric dropped)"
    )
    good = out.split("stub-cand-good  SHIP")[1].split("\n\n")[0]
    assert "root cause" not in good
    assert "stub-cand-good  SHIP" in out
    assert "stub-cand-noisy  NEEDS_REVIEW" in out
    assert "flag_exceptions.planted_recall  1.00 -> " in out
    assert "root step: select_samples.required_recall  1.00 -> " in out
    assert f"traces: local:{RUN_ID}/" in out


def test_report_only_prints_the_same_output_and_exits_0(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["gate", "--run", RUN_ID, "--runs-dir", str(run_dir.parent)]
    assert main(args) == 1
    gated = capsys.readouterr().out
    assert main([*args, "--report-only"]) == 0
    assert capsys.readouterr().out == gated
    assert "DO_NOT_SHIP" in gated


def test_baseline_file_round_trip_gives_the_same_verdicts(
    run_dir: Path, results: dict[str, GateResult], tmp_path: Path
) -> None:
    out = tmp_path / "baselines" / "production.json"
    assert (
        main(["baseline", "--run", RUN_ID, "--runs-dir", str(run_dir.parent), "--out", str(out)])
        == 0
    )
    before = out.read_bytes()
    stored = GateInput.model_validate_json(before)
    assert stored.config_id == "prod" and stored.source_run_id == RUN_ID
    assert set(stored.case_means["select_samples.required_recall"]) == set(CASE_IDS)
    again = {r.candidate_config_id: r for r in run_gate(run_dir, CONFIG_DIR / "gate.toml", out)}
    assert again["prod"].verdict is Verdict.SHIP
    for config_id, result in results.items():
        assert again[config_id].verdict is result.verdict, config_id
    assert out.read_bytes() == before  # the gate never edits a baseline


# --- pure tests on hand-built inputs -------------------------------------------------------


def gate_in(config_id: str, means: dict[str, dict[str, float]], **kw: object) -> GateInput:
    fields: dict[str, object] = {
        "source_run_id": "r",
        "config_id": config_id,
        "simulated": True,
        "suite_hash": "h",
        "case_means": means,
        "failures": {},
        "planted_misses": [],
        "pass_pow_n_share": 1.0,
        "outcome_agreement": 1.0,
        "infra_failure_rate": 0.0,
    }
    return GateInput.model_validate(fields | kw)


def all_ones(cases: list[str]) -> dict[str, dict[str, float]]:
    return {k: {c: 1.0 for c in cases} for k in CONFIG.critical}


THREE = ["c1", "c2", "c3"]


def test_small_drop_on_three_cases_is_not_confirmed() -> None:
    cand_means = all_ones(THREE)
    cand_means["extract_evidence.field_exact"] = {"c1": 0.9, "c2": 1.0, "c3": 1.0}
    result = evaluate_gate(gate_in("cand", cand_means), gate_in("base", all_ones(THREE)), CONFIG)
    assert result.verdict is Verdict.NEEDS_REVIEW
    [f] = result.findings
    assert f.metric == "extract_evidence.field_exact" and f.case_ids == ["c1"]
    assert f.message.startswith("drop not confirmed at this suite size: 3 cases")
    assert f.ci is not None and f.ci[1] >= 0


def test_same_seed_gives_an_identical_ci_and_another_seed_differs() -> None:
    deltas = [-0.3, -0.1, 0.0, 0.05, -0.2, -0.15, 0.1, -0.05]
    a = bootstrap_ci(deltas, 2000, 7, 0.95)
    assert bootstrap_ci(deltas, 2000, 7, 0.95) == a
    assert bootstrap_ci(deltas, 2000, 8, 0.95) != a
    assert a[0] < sum(deltas) / len(deltas) < a[1]


def test_zero_deltas_give_a_zero_ci_and_ship() -> None:
    assert bootstrap_ci([0.0] * 5, 2000, 7, 0.95) == (0.0, 0.0)
    result = evaluate_gate(
        gate_in("cand", all_ones(THREE)), gate_in("base", all_ones(THREE)), CONFIG
    )
    assert result.verdict is Verdict.SHIP and result.findings == []


def test_infra_heavy_candidate_needs_review() -> None:
    cand = gate_in("cand", all_ones(THREE), infra_failure_rate=0.2)
    result = evaluate_gate(cand, gate_in("base", all_ones(THREE)), CONFIG)
    assert result.verdict is Verdict.NEEDS_REVIEW
    [f] = result.findings
    assert f.check is Check.INFRA and "0.20" in f.message


def test_unconfirmed_noncritical_drop_is_a_note_and_confirmed_one_needs_review() -> None:
    base_means = all_ones(THREE) | {"select_samples.precision": {c: 1.0 for c in THREE}}
    wobble = all_ones(THREE) | {"select_samples.precision": {"c1": 0.9, "c2": 1.0, "c3": 1.0}}
    note = evaluate_gate(gate_in("cand", wobble), gate_in("base", base_means), CONFIG)
    assert note.verdict is Verdict.SHIP
    [f] = note.findings
    assert f.check is Check.NONCRITICAL_DROP and f.verdict is Verdict.SHIP
    drop = all_ones(THREE) | {"select_samples.precision": {c: 0.8 for c in THREE}}
    review = evaluate_gate(gate_in("cand", drop), gate_in("base", base_means), CONFIG)
    assert review.verdict is Verdict.NEEDS_REVIEW


def test_lower_is_better_metric_rising_is_a_drop() -> None:
    fpr = "flag_exceptions.false_positive_rate"
    base_means = all_ones(THREE) | {fpr: {c: 0.0 for c in THREE}}
    worse = all_ones(THREE) | {fpr: {c: 0.3 for c in THREE}}
    result = evaluate_gate(gate_in("cand", worse), gate_in("base", base_means), CONFIG)
    [f] = result.findings
    assert f.metric == fpr and f.mean_delta == pytest.approx(-0.3)
    assert result.verdict is Verdict.NEEDS_REVIEW


def test_planted_miss_is_do_not_ship_even_without_a_mean_drop() -> None:
    miss = {"case_id": "c2", "run_index": 3, "trace_ref": "local:r/abc", "detail": "missed"}
    cand = gate_in("cand", all_ones(THREE), planted_misses=[miss])
    result = evaluate_gate(cand, gate_in("base", all_ones(THREE)), CONFIG)
    assert result.verdict is Verdict.DO_NOT_SHIP
    [f] = result.findings
    assert (
        f.check is Check.PLANTED_MISS and f.case_ids == ["c2"] and f.trace_refs == ["local:r/abc"]
    )


def test_consistency_below_thresholds_needs_review() -> None:
    cand = gate_in("cand", all_ones(THREE), pass_pow_n_share=0.5, outcome_agreement=None)
    result = evaluate_gate(cand, gate_in("base", all_ones(THREE)), CONFIG)
    assert result.verdict is Verdict.NEEDS_REVIEW
    assert [f.check for f in result.findings] == [Check.CONSISTENCY, Check.CONSISTENCY]


def test_gate_toml_parses_with_the_outcome_agreement_threshold() -> None:
    assert CONFIG.consistency.min_outcome_agreement == 0.9
    assert CONFIG.critical["extract_evidence.field_exact"] == 0.02
    assert CONFIG.stats.bootstrap_seed == 7


def test_root_cause_lines_group_findings_by_root_step() -> None:
    def drop(metric: str, root: str | None) -> Finding:
        r = (
            None
            if root is None
            else RootStep(metric=root, baseline_value=1.0, candidate_value=0.5, detail="d")
        )
        return Finding(
            check=Check.CRITICAL_DROP,
            verdict=Verdict.DO_NOT_SHIP,
            message="m",
            metric=metric,
            root=r,
        )

    findings = [
        drop("extract_evidence.field_exact", "extract_evidence.field_exact"),
        drop("tie_out.tieout_accuracy", "select_samples.required_recall"),
        Finding(check=Check.INFRA, verdict=Verdict.NEEDS_REVIEW, message="infra high"),
        drop("draft_workpaper.guard_pass", "extract_evidence.field_exact"),
    ]
    result = GateResult(
        candidate_config_id="cand",
        baseline_config_id="base",
        baseline_source="r",
        simulated=True,
        verdict=Verdict.DO_NOT_SHIP,
        findings=findings,
    )
    lines = format_result(result).splitlines()
    assert lines[1:3] == [
        "  root cause: extract_evidence.field_exact (explains 2 critical drops)",
        "  root cause: select_samples.required_recall (explains 1 critical drop)",
    ]
    heads = [line.split()[0] for line in lines[3:] if not line.startswith("    ")]
    assert heads == [
        "extract_evidence.field_exact",
        "draft_workpaper.guard_pass",
        "tie_out.tieout_accuracy",
        "infra:",
    ]


def test_unrooted_verdict_names_each_consistency_and_infra_reason() -> None:
    cand = gate_in(
        "cand",
        all_ones(THREE),
        pass_pow_n_share=0.5,
        outcome_agreement=None,
        infra_failure_rate=0.2,
    )
    lines = format_result(
        evaluate_gate(cand, gate_in("base", all_ones(THREE)), CONFIG)
    ).splitlines()
    assert lines[1:4] == [
        "  root cause: cases that fail in some runs "
        "(pass^N share 0.50, no critical metric dropped)",
        "  root cause: inconsistent outcomes across runs "
        "(outcome agreement no eligible case, no critical metric dropped)",
        "  root cause: infra failures (infra failure rate 0.20, no critical metric dropped)",
    ]
