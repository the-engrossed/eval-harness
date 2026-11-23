"""The four-config matrix (prod + three candidate stubs) on three cases at N=5, run once
per test session in a temp dir. Shared by the gate and scorecard tests."""

from pathlib import Path

from cases import CASES

from auditlab.config import CONFIG_DIR
from auditlab.domain.suite import Suite
from auditlab.models.config import load_models_config
from auditlab.models.stub import StubModel
from data.generate import write_case
from evals.feedback_cases import FeedbackCase
from evals.runner import EvalSpec, run_eval
from evals.scorers.stub_judge import StubJudge

CASE_IDS = ("s1-mixA", "s1-mixB", "s1-mixC")  # cutoff, judgment, judgment
CONFIGS = ["prod", "stub-cand-good", "stub-cand-cutoff", "stub-cand-noisy"]
RUN_ID = "gate-e2e"


def build_matrix_run(root: Path) -> Path:
    for case in (c for c in CASES if c.case_id in CASE_IDS):
        write_case(case, root / "engagements")
    return run_matrix(root, RUN_ID, CONFIGS)


def run_matrix(
    root: Path,
    run_id: str,
    config_ids: list[str],
    feedback: list[FeedbackCase] | None = None,
    runs_dir: Path | None = None,
) -> Path:
    """Run `config_ids` on the three cases (engagements already under root) at N=5."""
    engagements = root / "engagements"
    chosen = [c for c in CASES if c.case_id in CASE_IDS]
    models = load_models_config(CONFIG_DIR / "models.toml")
    suite = Suite.model_validate(
        {
            "name": "gate",
            "cases": [
                {"case_id": c.case_id, "seed": 1, "mix": c.case_id[-1], "tags": ["t"]}
                for c in chosen
            ],
        }
    )
    spec = EvalSpec(
        run_id=run_id,
        suite=suite,
        suite_hash="h",
        n=5,
        seed=7,
        workers=4,
        runs_dir=runs_dir or root / "runs",
        engagements_dir=engagements,
        cache_dir=None,
        models=models,
        config_ids=config_ids,
        judge_config_id="stub-judge",
        judge_price=None,
        feedback=feedback or [],
    )
    clients = {c: StubModel(models.model(c).model_id, engagements) for c in config_ids}
    return run_eval(spec, clients, StubJudge())
