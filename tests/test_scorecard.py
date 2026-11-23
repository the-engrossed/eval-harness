"""The static scorecard (SPEC.md §13): sections, verdicts from the gate, no network refs."""

import datetime as dt
import json
import re
import shutil
from pathlib import Path

import pytest
from matrix_run import CONFIGS

from auditlab.config import CONFIG_DIR
from auditlab.domain.enums import Verdict
from evals.gate_cli import run_gate
from evals.gate_models import Check, Finding, GateResult, RootStep
from report.headline import headline
from report.rows import usd, vs_base
from report.scorecard import build_context, main, render

GATE_TOML = CONFIG_DIR / "gate.toml"
AT = dt.datetime(2026, 11, 18, 12, 0, tzinfo=dt.UTC)
SECTIONS = [
    "headline",
    "verdicts",
    "heatmap",
    "final-only",
    "cost",
    "consistency",
    "reliability",
    "trend",
    "feedback",
    "footer",
]


def html_for(run_dir: Path, tmp: Path, at: dt.datetime = AT) -> str:
    history = run_dir.parent / "history.jsonl"
    return render(build_context(run_dir, history, tmp / "no-feedback", GATE_TOML, at))


@pytest.fixture(scope="module")
def html(matrix_run: Path, tmp_path_factory: pytest.TempPathFactory) -> str:
    return html_for(matrix_run, tmp_path_factory.mktemp("sc"))


def test_every_section_renders(html: str) -> None:
    for section in SECTIONS:
        assert f'id="{section}"' in html, section
    for config_id in CONFIGS:
        assert config_id in html
    assert "flag_exceptions.planted_recall" in html and "select_samples.required_recall" in html
    assert "not enough runs yet (1)" in html  # one history line: trend skipped
    assert "<strong>0</strong> cases from reviewer rejections" in html


def test_every_verdict_matches_evaluate_gate(matrix_run: Path, html: str) -> None:
    gate = run_gate(matrix_run, GATE_TOML, None)
    shown = dict(re.findall(r'data-verdict="([^"]+)">([A-Z_]+)<', html))
    assert shown == {g.candidate_config_id: g.verdict.value for g in gate}
    assert shown["stub-cand-cutoff"] == "DO_NOT_SHIP"
    assert "Sample selection left out cutoff-window invoices (explains 4 critical drops)" in html


def test_headline_is_built_from_the_gate(matrix_run: Path, html: str) -> None:
    sentence = headline(run_gate(matrix_run, GATE_TOML, None))
    assert sentence.startswith("1 of 3 candidates is safe to ship.")
    assert "stub-cand-cutoff is blocked: sample selection left out cutoff-window invoices." in (
        sentence
    )
    assert "stub-cand-noisy needs review: outcomes change between runs." in sentence
    assert f'id="headline">{sentence}</p>' in html


def test_headline_has_no_per_config_text() -> None:
    root = RootStep(
        metric="tie_out.tieout_accuracy", baseline_value=1, candidate_value=0.5, detail="d"
    )
    drop = Finding(check=Check.CRITICAL_DROP, verdict=Verdict.DO_NOT_SHIP, message="m", root=root)
    infra = Finding(
        check=Check.INFRA, verdict=Verdict.NEEDS_REVIEW, message="m", measure="infra_failure_rate"
    )

    def result(config_id: str, verdict: Verdict, findings: list[Finding]) -> GateResult:
        return GateResult(
            candidate_config_id=config_id,
            baseline_config_id="base",
            baseline_source="r",
            simulated=False,
            verdict=verdict,
            findings=findings,
        )

    sentence = headline(
        [result("x-new", Verdict.DO_NOT_SHIP, [drop]), result("y", Verdict.NEEDS_REVIEW, [infra])]
    )
    assert sentence == (
        "0 of 2 candidates are safe to ship. x-new is blocked: tie-out results were wrong. "
        "y needs review: too many runs hit infra failures."
    )


def test_no_url_and_no_script_without_a_sink(html: str) -> None:
    assert "http" not in html
    assert "built-in method" not in html and " object at 0x" not in html  # no dict-method leaks
    assert "<script" not in html and "src=" not in html and "<link" not in html


def test_all_simulated_run_has_one_banner_and_no_badges(html: str) -> None:
    assert html.count('id="simulated-banner"') == 1
    assert "no real model was called" in html
    assert 'class="badge SIM"' not in html


def test_mixed_run_keeps_per_config_badges(matrix_run: Path, tmp_path: Path) -> None:
    mixed = tmp_path / "runs" / matrix_run.name
    shutil.copytree(matrix_run, mixed)
    summary = json.loads((mixed / "summary.json").read_text())
    summary["configs"][1]["simulated"] = False  # pretend stub-cand-good is a real model
    (mixed / "summary.json").write_text(json.dumps(summary))
    html = html_for(mixed, tmp_path)
    assert 'id="simulated-banner"' not in html
    assert '<td class="mono">prod <span class="badge SIM">SIMULATED</span>' in html
    assert '<td class="mono">stub-cand-good <span class="badge SIM">' not in html


def test_metric_rows_show_readable_label_over_code_name(html: str) -> None:
    for label, code in [
        ("Required samples selected", "select_samples.required_recall"),
        ("Evidence quotes valid", "extract_evidence.citation_validity"),
        ("Evidence fields correct", "extract_evidence.field_exact"),
        ("Amounts tie out", "tie_out.tieout_accuracy"),
        ("Planted exceptions found", "flag_exceptions.planted_recall"),
        ("Workpaper has no unsupported numbers", "draft_workpaper.guard_pass"),
    ]:
        assert re.search(
            rf"<td>{label}(<span class=\"tag\">saturated\*</span>)?"
            rf'<span class="code">{code}</span></td>',
            html,
        ), label


def test_saturated_rows_get_a_tag_and_one_footnote(html: str) -> None:
    assert '<span class="tag">saturated*</span>' in html
    assert html.count("Saturated: every model scores ≥ 0.98, so this metric no longer") == 1


def test_verdict_table_shows_one_trace_and_links_to_details(html: str) -> None:
    verdicts = html.split('id="verdicts"')[1].split("</section>")[0]
    assert verdicts.count("local:") == 1
    assert re.search(r'<a href="#details-stub-cand-cutoff">\+\d+ more</a>', verdicts)
    assert 'id="details-stub-cand-cutoff"' in html
    details = html.split('id="details"')[1].split("</section>")[0]
    assert details.count("local:") > 5


def test_cost_reads_to_three_figures_with_change_vs_prod(html: str) -> None:
    assert usd(180_292) == "$0.180" and usd(36_049) == "$0.0360"
    assert usd(1_234_567) == "$1.23" and usd(0) == "$0" and usd(None) == "—"
    assert vs_base(36_049, 180_292) == "\u221280%" and vs_base(180_346, 180_292) == "0%"
    assert "$0.000000" not in html
    assert "— (simulated judge)" in html
    assert "\u221280%" in html


def test_missed_by_final_only_reads_as_a_fraction(html: str) -> None:
    section = html.split('id="final-only"')[1].split("</section>")[0]
    assert re.search(r"\d+ of 15 runs", section)
    assert ">Runs<" not in section


def test_same_run_gives_identical_bytes_apart_from_generated_time(
    matrix_run: Path, tmp_path: Path
) -> None:
    a, b = html_for(matrix_run, tmp_path), html_for(matrix_run, tmp_path)
    assert a == b
    later = html_for(matrix_run, tmp_path, AT + dt.timedelta(hours=1))
    assert later != a
    assert later.replace("2026-11-18T13:00:00+00:00", "2026-11-18T12:00:00+00:00") == a


def test_trend_draws_a_line_with_two_or_more_history_points(
    matrix_run: Path, tmp_path: Path
) -> None:
    [line] = (matrix_run.parent / "history.jsonl").read_text().splitlines()
    earlier = json.loads(line) | {"run_id": "older", "created_at": "2026-01-01T00:00:00+00:00"}
    other_suite = json.loads(line) | {"run_id": "other", "suite_hash": "different"}
    history = tmp_path / "history.jsonl"
    history.write_text("\n".join([json.dumps(earlier), line, json.dumps(other_suite)]) + "\n")
    html = render(build_context(matrix_run, history, tmp_path / "fb", GATE_TOML, AT))
    assert html.count("<polyline") == len(CONFIGS)
    assert "(2 runs)" in html


def test_feedback_cases_are_counted(matrix_run: Path, tmp_path: Path) -> None:
    feedback = tmp_path / "from_feedback"
    feedback.mkdir()
    for i in range(2):
        (feedback / f"fb{i}.json").write_text("{}")
    history = matrix_run.parent / "history.jsonl"
    html = render(build_context(matrix_run, history, feedback, GATE_TOML, AT))
    assert "<strong>2</strong> cases from reviewer rejections" in html


def test_cli_writes_the_file(matrix_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "out" / "scorecard.html"
    args = ["--run", matrix_run.name, "--runs-dir", str(matrix_run.parent), "--out", str(out)]
    assert main([*args, "--feedback-dir", str(tmp_path / "none")]) == 0
    assert out.is_file() and f"scorecard: {out}" in capsys.readouterr().out
