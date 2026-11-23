"""Eval runner (SPEC.md §10): suite × model configs × N runs, scored, written to runs/<run_id>/.

`python -m evals.runner --suite core --models prod --n 3 [--workers 4]`. Every model config
and the judge are validated before the first run. Runs execute on a bounded thread pool;
results keep (config, case, run_index) order whatever the worker count. Writes
manifest.json, results.jsonl, summary.json and spans.jsonl, and appends to
runs/history.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from auditlab.agent.graph import run_case
from auditlab.config import ENGAGEMENTS_DIR, MODELS_TOML, REPO_ROOT, RUNS_DIR
from auditlab.docs.store import DocStore
from auditlab.domain.answer_key import AnswerKey
from auditlab.domain.suite import Suite
from auditlab.models.base import ModelClient
from auditlab.models.cache import CachedClient, ResponseCache
from auditlab.models.config import load_models_config
from auditlab.models.registry import build_client
from evals.artifacts import EvalSpec, build_manifest, build_summary, history_line, price_for
from evals.feedback_cases import FeedbackCase, load_feedback_cases
from evals.results import RunResult, run_result
from evals.scorers import score_run
from evals.scorers.feedback import feedback_grade
from evals.scorers.judge import STUB_JUDGE_ID, Judge, build_judge, judge_config_id

__all__ = ["EvalSpec", "main", "run_eval"]

SUITES_DIR = REPO_ROOT / "evals" / "suites"
FEEDBACK_DIR = SUITES_DIR / "from_feedback"
CACHE_DIR = RUNS_DIR / "cache"
Task = tuple[str, str, int]  # (config_id, case_id, run_index)


def _git() -> tuple[str, bool]:
    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()

    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


def _run_one(
    spec: EvalSpec, clients: Mapping[str, ModelClient], judge: Judge, task: Task
) -> RunResult:
    config_id, case_id, run_index = task
    feedback = next((f for f in spec.feedback if f.case_id == case_id), None)
    source = feedback.source_case_id if feedback is not None else case_id
    key = f"{spec.run_id}|{config_id}|{case_id}|{run_index}"
    state = run_case(
        source,
        client=clients[config_id],
        model_config_id=config_id,
        engagements_dir=spec.engagements_dir,
        runs_dir=spec.runs_dir,
        run_id=spec.run_id,
        run_index=run_index,
        seed=spec.seed,
        prompt_version=spec.models.model(config_id).prompt_version,
        price=price_for(spec.models, config_id),
        telemetry_key=key,
        id_seed=key,
    )
    case_dir = spec.engagements_dir / source
    answer_key = AnswerKey.model_validate_json((case_dir / "answer_key.json").read_text("utf-8"))
    cited = {f.doc_id for f in state.evidence.fields} if state.evidence is not None else set()
    pages = DocStore(case_dir).pages(cited | {e.doc_id for e in answer_key.evidence})
    grade = score_run(state, answer_key, pages, judge)
    if feedback is None:
        return run_result(state, grade, spec.judge_price)
    result = run_result(state, feedback_grade(state, feedback, grade), spec.judge_price)
    return result.model_copy(update={"case_id": feedback.case_id})


def _wrap(spec: EvalSpec, clients: Mapping[str, ModelClient]) -> dict[str, ModelClient]:
    """Cache every client by the §6 key, except replay (already cache-only) and --no-cache."""
    if spec.cache_dir is None:
        return {c: clients[c] for c in spec.config_ids}
    cache = ResponseCache(spec.cache_dir)
    return {
        c: clients[c]
        if spec.models.model(c).provider == "replay"
        else CachedClient(clients[c], cache)
        for c in spec.config_ids
    }


def run_eval(spec: EvalSpec, clients: Mapping[str, ModelClient], judge: Judge) -> Path:
    """Run every task, write the artifacts, return runs/<run_id>/."""
    run_dir = spec.runs_dir / spec.run_id
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; pick another --run-id")
    run_dir.mkdir(parents=True)
    created_at = dt.datetime.now(dt.UTC)
    wrapped = _wrap(spec, clients)
    tasks: list[Task] = [
        (c, case_id, i) for c in spec.config_ids for case_id in spec.case_ids for i in range(spec.n)
    ]
    with ThreadPoolExecutor(max_workers=spec.workers) as pool:
        results = list(pool.map(lambda t: _run_one(spec, wrapped, judge, t), tasks))

    summary = build_summary(spec, results)
    sha, dirty = _git()
    manifest = build_manifest(spec, results, created_at=created_at, git_sha=sha, git_dirty=dirty)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(r.model_dump_json() + "\n" for r in results)
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n", "utf-8")
    with (spec.runs_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_line(manifest, summary), sort_keys=True) + "\n")
    return run_dir


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m evals.runner")
    p.add_argument("--suite", default="core")
    p.add_argument("--models", default="prod", help="comma-separated config_ids")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--run-id")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--engagements-dir", type=Path, default=ENGAGEMENTS_DIR)
    p.add_argument("--suites-dir", type=Path, default=SUITES_DIR)
    p.add_argument("--models-toml", type=Path, default=MODELS_TOML)
    p.add_argument("--feedback-dir", type=Path, default=FEEDBACK_DIR)
    p.add_argument("--no-feedback", action="store_true", help="skip from_feedback cases")
    p.add_argument(
        "--production",
        help="config_id the gate treats as the baseline in this run (default: [production])",
    )
    return p.parse_args(argv)


def _load_suite(path: Path, engagements_dir: Path) -> tuple[Suite, str]:
    if not path.is_file():
        raise FileNotFoundError(f"suite {path} not found")
    raw = path.read_bytes()
    suite = Suite.model_validate_json(raw)
    for case in suite.cases:
        for name in ("population.json", "answer_key.json"):
            if not (engagements_dir / case.case_id / name).is_file():
                raise FileNotFoundError(
                    f"{case.case_id}: {name} missing in {engagements_dir}. Run `make data`."
                )
    return suite, hashlib.sha256(raw).hexdigest()


def _with_feedback(suite_hash: str, feedback: list[FeedbackCase], engagements: Path) -> str:
    """Feedback cases are part of what ran, so they are part of the suite hash."""
    if not feedback:
        return suite_hash
    for f in feedback:
        if not (engagements / f.source_case_id / "answer_key.json").is_file():
            raise FileNotFoundError(f"{f.case_id}: source case {f.source_case_id} missing")
    digest = hashlib.sha256(suite_hash.encode())
    for f in feedback:
        digest.update(f.model_dump_json().encode())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse(argv)
    if args.n < 2:
        raise ValueError(f"--n {args.n}: pass^N needs N >= 2 (SPEC.md §10 asks for 3 or more)")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    models = load_models_config(args.models_toml)
    ids = [i.strip() for i in args.models.split(",") if i.strip()]
    if not ids:
        raise ValueError("--models is empty; pass config_ids from config/models.toml")
    for config_id in ids:
        models.model(config_id)
    if args.production is not None:
        if args.production not in ids:
            raise ValueError(f"--production {args.production} is not in --models {args.models}")
        models = models.model_copy(update={"production_config_id": args.production})
    clients = {i: build_client(i, models, engagements_dir=args.engagements_dir) for i in ids}
    cache_dir = None if args.no_cache else args.cache_dir
    judge = build_judge(os.environ, models, model_config_ids=ids, cache_dir=cache_dir)
    judge_id = judge_config_id(os.environ)
    suite, suite_hash = _load_suite(args.suites_dir / f"{args.suite}.json", args.engagements_dir)
    feedback = [] if args.no_feedback else load_feedback_cases(args.feedback_dir)
    suite_hash = _with_feedback(suite_hash, feedback, args.engagements_dir)
    run_id = args.run_id or f"{suite.name}-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}"
    spec = EvalSpec(
        run_id=run_id,
        suite=suite,
        suite_hash=suite_hash,
        n=args.n,
        seed=args.seed,
        workers=args.workers,
        runs_dir=args.runs_dir,
        engagements_dir=args.engagements_dir,
        cache_dir=cache_dir,
        models=models,
        config_ids=ids,
        judge_config_id=judge_id,
        judge_price=None if judge_id == STUB_JUDGE_ID else price_for(models, judge_id),
        feedback=feedback,
    )
    run_dir = run_eval(spec, clients, judge)
    print(f"run {run_id}: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
