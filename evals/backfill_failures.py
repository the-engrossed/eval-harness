"""Backfill replay failure entries from a recorded run (SPEC.md §6.4).

`python -m evals.backfill_failures --run <run_id> [--cache-dir runs/demo/cache]`

A call that failed while recording (timeout, prompt too large, bad output) has no cached
response, so replaying the run would miss. For every run whose step recorded an error,
this rebuilds that exact request by replaying the run up to the failing node (every
upstream response is cached, so the request and its §6 key are the recorded ones), then
writes a FailureEntry under that key with the recorded origin and message. A request that
cannot be rebuilt exactly raises; that run must be re-recorded instead. Never overwrites a
recorded response. Judge failures are not backfilled.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from auditlab.agent.graph import run_case
from auditlab.config import ENGAGEMENTS_DIR, MODELS_TOML, RUNS_DIR
from auditlab.domain.enums import FailureOrigin, Node
from auditlab.models.base import ModelRequest, ModelResponse
from auditlab.models.cache import FailureEntry, ResponseCache, request_key
from auditlab.models.config import ModelsConfig, load_models_config
from auditlab.models.errors import ProviderError, RecordedFailure
from auditlab.models.prompt_size import estimate_prompt_tokens
from auditlab.models.replay import ReplayMissError, ReplayModel
from auditlab.models.variant import request_variant
from evals.artifacts import price_for
from evals.feedback_cases import load_feedback_cases

FEEDBACK_DIR = Path(__file__).resolve().parent / "suites" / "from_feedback"
NODE_FAILED = re.compile(r"^node failed \((agent|infra|data)\): (.*)$", re.DOTALL)
TOO_LARGE = re.compile(r"prompt too large: (\d+) tokens")


@dataclass(frozen=True)
class Backfilled:
    config_id: str
    case_id: str
    run_index: int
    node: Node
    key: str


@dataclass
class _Capture:
    """Replays cached calls; stops the run at the first call with no recorded response."""

    inner: ReplayModel
    node: Node | None = None
    key: str | None = None
    request: ModelRequest | None = None
    already: bool = False

    def __post_init__(self) -> None:
        self.provider, self.model_id = self.inner.provider, self.inner.model_id
        self.simulated = self.inner.simulated

    def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            return self.inner.complete(request)
        except (ReplayMissError, RecordedFailure) as exc:
            self.node, self.request = request.node, request
            self.key = request_key(self.provider, self.model_id, request, self.inner.variant)
            self.already = isinstance(exc, RecordedFailure)
            stop = ProviderError(self.provider, self.model_id, request.node, "backfill stop")
            raise stop from exc


def _recorded_failure(row: dict[str, object]) -> tuple[Node, FailureOrigin, str] | None:
    scores = row["scores"]
    assert isinstance(scores, list)
    for s in scores:
        match = NODE_FAILED.match(s["detail"])
        if match and not s.get("skipped"):
            return Node(s["node"]), FailureOrigin(match.group(1)), match.group(2)
    return None


def backfill_failures(
    run_dir: Path,
    cache: ResponseCache,
    models: ModelsConfig,
    engagements_dir: Path,
    feedback_dir: Path = FEEDBACK_DIR,
) -> list[Backfilled]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").open(encoding="utf-8")]
    sources = {f.case_id: f.source_case_id for f in load_feedback_cases(feedback_dir)}
    written: list[Backfilled] = []
    for row in rows:
        failed = _recorded_failure(row)
        if failed is None:
            continue
        node, origin, message = failed
        config_id, case_id, run_index = row["model_config_id"], row["case_id"], row["run_index"]
        model = models.model(config_id)
        capture = _Capture(
            ReplayModel(
                cache,
                provider=model.replay_provider or model.provider,
                model_id=model.model_id,
                variant=request_variant(model.reasoning_effort, model.max_output_tokens),
            )
        )
        capture.simulated = model.simulated  # price lookup follows the recorded config
        with tempfile.TemporaryDirectory() as tmp:
            run_case(
                sources.get(case_id, case_id),
                client=capture,
                model_config_id=config_id,
                engagements_dir=engagements_dir,
                runs_dir=Path(tmp),
                run_id="backfill",
                run_index=run_index,
                seed=manifest["seed"],
                prompt_version=model.prompt_version,
                price=price_for(models, config_id),
            )
        where = f"{config_id} {case_id} run {run_index}"
        if capture.node is not node or capture.key is None or capture.request is None:
            got = capture.node.value if capture.node else "no call"
            raise ValueError(
                f"{where}: recorded failure at {node.value} but the rebuilt run stops at "
                f"{got}; the request cannot be rebuilt exactly. Re-record this run."
            )
        too_large = TOO_LARGE.search(message)
        if too_large and model.prompt_bytes_per_token is not None:
            rebuilt = estimate_prompt_tokens(capture.request, model.prompt_bytes_per_token)
            if rebuilt != int(too_large.group(1)):
                raise ValueError(
                    f"{where}: rebuilt prompt estimates {rebuilt} tokens, recorded "
                    f"{too_large.group(1)}; not the recorded request. Re-record this run."
                )
        if capture.already:
            existing = cache.get_failure(capture.key)
            if existing is None or existing.message != message:
                raise ValueError(f"{where}: key {capture.key} holds a different failure")
            continue
        cache.put_failure(
            capture.key,
            FailureEntry(
                provider=capture.provider,
                model_id=capture.model_id,
                node=node.value,
                run_index=run_index,
                failure_origin=origin,
                message=message,
                source_run_id=manifest["run_id"],
            ),
        )
        written.append(Backfilled(config_id, case_id, run_index, node, capture.key))
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.backfill_failures")
    p.add_argument("--run", required=True)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--cache-dir", type=Path, default=RUNS_DIR / "demo" / "cache")
    p.add_argument("--models-toml", type=Path, default=MODELS_TOML)
    p.add_argument("--engagements-dir", type=Path, default=ENGAGEMENTS_DIR)
    args = p.parse_args(argv)
    written = backfill_failures(
        args.runs_dir / args.run,
        ResponseCache(args.cache_dir),
        load_models_config(args.models_toml),
        args.engagements_dir,
    )
    for w in written:
        print(f"{w.config_id} {w.case_id} run {w.run_index} {w.node.value}: {w.key}")
    print(f"backfilled {len(written)} failure entries into {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
