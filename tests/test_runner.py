"""The eval runner end to end on stub-prod, in temp dirs, no network (SPEC.md §10)."""

import json
from pathlib import Path

import pytest
from cases import CASES

from auditlab.domain.enums import FailureOrigin, Node
from auditlab.models.base import ModelClient, ModelRequest, ModelResponse
from auditlab.models.config import ModelConfig, ModelsConfig
from auditlab.models.errors import ModelOutputError, ProviderError
from auditlab.models.stub import StubModel
from data.generate import write_case
from evals.runner import EvalSpec, main, run_eval
from evals.scorers.stub_judge import StubJudge

TWO = [c for c in CASES if c.case_id in ("s1-mixA", "s1-mixD")]
CRITICAL = [
    "select_samples.required_recall",
    "extract_evidence.citation_validity",
    "extract_evidence.field_exact",
    "tie_out.tieout_accuracy",
    "flag_exceptions.planted_recall",
    "draft_workpaper.guard_pass",
]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("world")
    for case in TWO:
        write_case(case, root / "engagements")
    suites = root / "suites"
    suites.mkdir()
    cases = [{"case_id": c.case_id, "seed": 1, "mix": c.case_id[-1], "tags": ["t"]} for c in TWO]
    (suites / "two.json").write_text(json.dumps({"name": "two", "cases": cases}))
    (root / "models.toml").write_text(
        '[production]\nconfig_id = "prod"\n\n'
        '[[model]]\nconfig_id = "prod"\nprovider = "stub"\nmodel_id = "stub-prod"\n'
        'prompt_version = "v1"\nsimulated = true\n'
    )
    return root


def stub_config(*ids: str) -> ModelsConfig:
    models = [
        ModelConfig(
            config_id=i, provider="stub", model_id="stub-prod", prompt_version="v1", simulated=True
        )
        for i in ids
    ]
    return ModelsConfig(production_config_id=ids[0], models=models, prices=[])


class Counting:
    def __init__(
        self, inner: ModelClient, fail: tuple[Node, str, int, type[Exception]] | None = None
    ) -> None:
        self.inner, self.fail = inner, fail
        self.provider, self.model_id, self.simulated = inner.provider, inner.model_id, True
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.fail is not None:
            node, case_id, run_index, kind = self.fail
            if (request.node, request.case_id, request.run_index) == (node, case_id, run_index):
                if kind is ProviderError:
                    raise ProviderError(self.provider, self.model_id, node, "HTTP 503")
                raise ModelOutputError(self.model_id, node, "{", "invalid JSON")
        return self.inner.complete(request)


def spec(world: Path, tmp: Path, run_id: str, *ids: str, workers: int = 4, cache: bool = True):
    from auditlab.domain.suite import Suite

    suite_path = world / "suites" / "two.json"
    return EvalSpec(
        run_id=run_id,
        suite=Suite.model_validate_json(suite_path.read_text()),
        suite_hash="h",
        n=2,
        seed=7,
        workers=workers,
        runs_dir=tmp / "runs",
        engagements_dir=world / "engagements",
        cache_dir=tmp / "cache" if cache else None,
        models=stub_config(*(ids or ("prod",))),
        config_ids=list(ids or ("prod",)),
        judge_config_id="stub-judge",
        judge_price=None,
    )


def clients(world: Path, *ids: str, fail: dict[str, tuple] | None = None) -> dict[str, Counting]:
    fail = fail or {}
    return {
        i: Counting(StubModel("stub-prod", world / "engagements"), fail.get(i))
        for i in (ids or ("prod",))
    }


def read(run_dir: Path) -> tuple[dict, list[dict], dict]:
    summary = json.loads((run_dir / "summary.json").read_text())
    results = [json.loads(line) for line in (run_dir / "results.jsonl").open()]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return summary, results, manifest


def test_two_cases_n2_writes_every_artifact_and_criticals_are_one(world: Path, tmp_path: Path):
    run_dir = run_eval(spec(world, tmp_path, "r1"), clients(world), StubJudge())
    for name in ("manifest.json", "results.jsonl", "summary.json", "spans.jsonl"):
        assert (run_dir / name).is_file(), name
    summary, results, manifest = read(run_dir)
    assert len(results) == 4
    assert [(r["case_id"], r["run_index"]) for r in results] == [
        ("s1-mixA", 0),
        ("s1-mixA", 1),
        ("s1-mixD", 0),
        ("s1-mixD", 1),
    ]
    [prod] = summary["configs"]
    for metric in CRITICAL:
        assert prod["accuracy"][metric]["value"] == 1.0, metric
    assert prod["simulated"] is True
    assert prod["pass_pow_n_share"] == 1.0
    assert prod["infra_failure_rate"] == 0.0
    assert summary["saturation_assessed"] is False and summary["saturated_metrics"] == []
    assert manifest["git_sha"] and manifest["suite"]["hash"] == "h"
    assert manifest["judge"]["config_id"] == "stub-judge"
    assert manifest["production_config_id"] == "prod"
    history = (tmp_path / "runs" / "history.jsonl").read_text().splitlines()
    assert len(history) == 1 and json.loads(history[0])["run_id"] == "r1"


def test_second_identical_run_is_all_cache_hits_with_identical_summary(world: Path, tmp_path: Path):
    first = run_eval(spec(world, tmp_path, "c1"), clients(world), StubJudge())
    again = clients(world)
    second = run_eval(spec(world, tmp_path, "c2"), again, StubJudge())
    assert again["prod"].calls == 0
    _, _, manifest = read(second)
    assert manifest["cache"]["misses"] == 0 and manifest["cache"]["hits"] > 0
    assert (first / "summary.json").read_bytes() == (second / "summary.json").read_bytes()
    assert len((tmp_path / "runs" / "history.jsonl").read_text().splitlines()) == 2


def test_provider_error_on_one_node_is_an_infra_step_not_a_crash(world: Path, tmp_path: Path):
    fail = {"prod": (Node.FLAG_EXCEPTIONS, "s1-mixD", 1, ProviderError)}
    run_dir = run_eval(
        spec(world, tmp_path, "inf", cache=False), clients(world, fail=fail), StubJudge()
    )
    summary, results, _ = read(run_dir)
    assert len(results) == 4
    bad = next(r for r in results if (r["case_id"], r["run_index"]) == ("s1-mixD", 1))
    flag = [s for s in bad["scores"] if s["node"] == "flag_exceptions"]
    assert flag and all(s["failure_origin"] == FailureOrigin.INFRA.value for s in flag)
    assert bad["infra_failed"] is True
    [prod] = summary["configs"]
    assert prod["infra_failure_rate"] == 0.25
    for metric in CRITICAL:
        assert prod["accuracy"][metric]["value"] == 1.0, metric
    assert prod["pass_pow_n_excluded"] == ["s1-mixD"]


def test_saturation_flags_only_metrics_every_config_keeps_at_098(world: Path, tmp_path: Path):
    fail = {"shaky": (Node.FLAG_EXCEPTIONS, "s1-mixA", 0, ModelOutputError)}
    run_dir = run_eval(
        spec(world, tmp_path, "sat", "prod", "shaky", cache=False),
        clients(world, "prod", "shaky", fail=fail),
        StubJudge(),
    )
    summary, _, _ = read(run_dir)
    assert summary["saturation_assessed"] is True
    saturated = summary["saturated_metrics"]
    assert "select_samples.required_recall" in saturated
    assert "draft_workpaper.guard_pass" in saturated
    # the fault hit s1-mixA, which has no judgment items, so judgment_allowed stays saturated
    assert "flag_exceptions.planted_recall" not in saturated
    assert "flag_exceptions.false_positive_rate" not in saturated
    shaky = summary["configs"][1]
    assert shaky["accuracy"]["flag_exceptions.planted_recall"]["value"] == 0.75


def test_output_is_identical_with_one_or_four_workers(world: Path, tmp_path: Path):
    one = run_eval(
        spec(world, tmp_path / "w1", "same", workers=1, cache=False), clients(world), StubJudge()
    )
    four = run_eval(
        spec(world, tmp_path / "w4", "same", workers=4, cache=False), clients(world), StubJudge()
    )
    for name in ("results.jsonl", "summary.json"):
        assert (one / name).read_bytes() == (four / name).read_bytes(), name


def test_cli_validates_models_and_judge_before_running(
    world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base = [
        "--suite",
        "two",
        "--suites-dir",
        str(world / "suites"),
        "--engagements-dir",
        str(world / "engagements"),
        "--models-toml",
        str(world / "models.toml"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--n",
        "2",
        "--feedback-dir",
        str(tmp_path / "no-feedback"),
    ]
    with pytest.raises(ValueError, match="unknown config_id"):
        main([*base, "--models", "prod,nope"])
    monkeypatch.setenv("AUDITLAB_JUDGE_CONFIG", "prod")
    with pytest.raises(ValueError, match="also a model under test"):
        main([*base, "--models", "prod"])
    with pytest.raises(ValueError, match="N >= 2"):
        main([*base, "--models", "prod", "--n", "1"])
    assert not (tmp_path / "runs").exists()


def test_cli_runs_the_suite(world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AUDITLAB_JUDGE_CONFIG", raising=False)
    code = main(
        [
            "--suite",
            "two",
            "--suites-dir",
            str(world / "suites"),
            "--engagements-dir",
            str(world / "engagements"),
            "--models-toml",
            str(world / "models.toml"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--feedback-dir",
            str(tmp_path / "no-feedback"),  # never the repo's local from_feedback/
            "--models",
            "prod",
            "--n",
            "2",
            "--run-id",
            "cli",
        ]
    )
    assert code == 0
    summary, _, manifest = read(tmp_path / "runs" / "cli")
    assert summary["suite"] == "two"
    assert len(manifest["suite"]["hash"]) == 64


def test_production_flag_picks_the_baseline_for_this_run(
    world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("AUDITLAB_JUDGE_CONFIG", raising=False)
    toml = tmp_path / "models.toml"
    toml.write_text(
        (world / "models.toml").read_text()
        + '\n[[model]]\nconfig_id = "cand"\nprovider = "stub"\nmodel_id = "stub-cand-good"\n'
        'prompt_version = "v1"\nsimulated = true\n'
    )
    base = [
        "--suite", "two", "--suites-dir", str(world / "suites"),
        "--engagements-dir", str(world / "engagements"), "--models-toml", str(toml),
        "--runs-dir", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "cache"),
        "--feedback-dir", str(tmp_path / "no-feedback"), "--n", "2",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="--production nope is not in --models"):
        main([*base, "--models", "prod,cand", "--production", "nope"])
    assert main([*base, "--models", "cand", "--production", "cand", "--run-id", "p"]) == 0
    _, _, manifest = read(tmp_path / "runs" / "p")
    assert manifest["production_config_id"] == "cand"
    assert main([*base, "--models", "prod,cand", "--run-id", "d"]) == 0
    _, _, manifest = read(tmp_path / "runs" / "d")
    assert manifest["production_config_id"] == "prod"
