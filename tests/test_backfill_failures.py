"""Backfill replay failure entries from a recorded run, then replay it to the same scores."""

from pathlib import Path

from test_runner import CRITICAL, clients, read, spec, world  # noqa: F401  (fixture)

from auditlab.domain.enums import Node
from auditlab.models.cache import ResponseCache
from auditlab.models.errors import ProviderError
from auditlab.models.replay import ReplayModel
from evals.backfill_failures import backfill_failures
from evals.runner import run_eval
from evals.scorers.stub_judge import StubJudge


def _comparable(results: list[dict]) -> list[tuple]:
    return [
        (
            r["case_id"],
            r["run_index"],
            r["infra_failed"],
            [
                (s["metric"], s["value"], s["passed"], s["failure_origin"], s["detail"])
                for s in r["scores"]
            ],
        )
        for r in results
    ]


def test_backfilled_failures_replay_to_identical_scores(world: Path, tmp_path: Path) -> None:  # noqa: F811
    fail = {"prod": (Node.EXTRACT_EVIDENCE, "s1-mixD", 1, ProviderError)}
    recorded_spec = spec(world, tmp_path, "rec")
    run_dir = run_eval(recorded_spec, clients(world, fail=fail), StubJudge())
    _, recorded, _ = read(run_dir)
    cache = ResponseCache(tmp_path / "cache")

    written = backfill_failures(
        run_dir, cache, recorded_spec.models, recorded_spec.engagements_dir
    )
    assert [(w.config_id, w.case_id, w.run_index, w.node) for w in written] == [
        ("prod", "s1-mixD", 1, Node.EXTRACT_EVIDENCE)
    ]
    assert backfill_failures(
        run_dir, cache, recorded_spec.models, recorded_spec.engagements_dir
    ) == [], "a second backfill writes nothing new"

    replayed_stub = ReplayModel(cache, provider="stub", model_id="stub-prod")
    replayed_stub.simulated = True  # replaying a stub recording; real replays stay False
    replay = {"prod": replayed_stub}
    replay_spec = spec(world, tmp_path, "replayed", cache=False)
    _, replayed, _ = read(run_eval(replay_spec, replay, StubJudge()))
    assert _comparable(replayed) == _comparable(recorded)
