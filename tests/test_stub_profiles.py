"""Candidate stub profiles (SPEC.md §6.3): each differs from stub-prod at one node only."""

from pathlib import Path

import pytest
from cases import CASES, each_case

import data.sampling as sampling
from auditlab.agent.graph import run_case
from auditlab.agent.nodes.select_samples import population_summary
from auditlab.agent.state import AgentState
from auditlab.docs.loader import pages_from_html
from auditlab.domain.enums import ExceptionType, Node
from auditlab.domain.models import SampleSelection
from auditlab.models import stub_profiles as sp
from auditlab.models.stub import RULE_LINES, StubModel
from data.generate import write_case
from evals.scorers import RunGrade, score_run
from evals.scorers.stub_judge import StubJudge

BY_ID = {c.case_id: c for c in CASES}
CRITICAL = {
    "required_recall",
    "citation_validity",
    "field_exact",
    "tieout_accuracy",
    "planted_recall",
    "guard_pass",
}


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("profiles") / "engagements"
    for case_id in ("s1-mixA", "s1-mixB", "s1-mixD"):
        write_case(BY_ID[case_id], root)
    return root


def run(world: Path, tmp: Path, profile: str, case_id: str, run_index: int = 0) -> AgentState:
    return run_case(
        case_id,
        client=StubModel(profile, world),
        model_config_id=profile,
        engagements_dir=world,
        runs_dir=tmp,
        run_id=f"{profile}-{case_id}-{run_index}-{tmp.name}",
        run_index=run_index,
    )


def grade(state: AgentState) -> RunGrade:
    case = BY_ID[state.case_id]
    pages = {d: {p.page_no: p.text for p in pages_from_html(h)} for d, h in case.docs.items()}
    return score_run(state, case.answer_key, pages, StubJudge())


def values(g: RunGrade) -> dict[str, float]:
    return {s.metric: s.value for s in g.scores}


def tokens(state: AgentState) -> dict[Node, int]:
    return {r.node: r.tokens_in + r.tokens_out for r in state.step_records}


def test_cand_good_is_oracle_with_fewer_tokens(world: Path, tmp_path: Path) -> None:
    prod = run(world, tmp_path, sp.STUB_PROD, "s1-mixD")
    good = run(world, tmp_path, sp.CAND_GOOD, "s1-mixD")
    v = values(grade(good))
    assert all(v[m] == 1.0 for m in CRITICAL if m in v)
    assert good.workpaper == prod.workpaper and good.exceptions == prod.exceptions
    prod_t, good_t = tokens(prod), tokens(good)
    for node in (n for n in Node if n is not Node.TIE_OUT):
        assert good_t[node] < prod_t[node], node
    assert sum(good_t.values()) < sum(prod_t.values())


def test_cand_cutoff_misses_the_cutoff_error_and_final_only_misses_it(
    world: Path, tmp_path: Path
) -> None:
    state = run(world, tmp_path, sp.CAND_CUTOFF, "s1-mixA")
    key = BY_ID["s1-mixA"].answer_key
    [cutoff] = [x.item_id for x in key.expected_exceptions if x.type is ExceptionType.CUTOFF_ERROR]
    assert state.samples is not None and cutoff not in state.samples.item_ids
    assert state.exceptions is not None
    assert cutoff not in {x.item_id for x in state.exceptions.exceptions}
    g = grade(state)
    v = values(g)
    assert v["required_recall"] < 1.0 and v["planted_recall"] < 1.0
    assert g.shadow_pass and g.missed_by_final_only


def test_cand_cutoff_on_a_case_with_no_cutoff_error(world: Path, tmp_path: Path) -> None:
    g = grade(run(world, tmp_path, sp.CAND_CUTOFF, "s1-mixB"))
    v = values(g)
    for metric in ("planted_recall", "citation_validity", "guard_pass", "precision"):
        assert v[metric] == 1.0, metric
    assert all(v[m] == 1.0 for m in v if m.startswith("judge_"))
    assert v["required_recall"] < 1.0
    failing = {s.metric for s in g.scores if not s.passed}
    assert failing <= {"required_recall", "field_exact", "tieout_accuracy"}


def test_cutoff_constants_match_the_sampling_rule() -> None:
    assert sp.KEY_ITEM_THRESHOLD_CENTS == sampling.KEY_ITEM_THRESHOLD_CENTS
    assert (sp.CUTOFF_START, sp.CUTOFF_END) == (sampling.CUTOFF_START, sampling.CUTOFF_END)


@each_case
def test_cutoff_selection_is_rules_one_and_three(case) -> None:
    pop = case.population
    full = SampleSelection(item_ids=list(case.answer_key.required_sample_ids), rationale=RULE_LINES)
    dropped = sp.drop_cutoff_window(full, population_summary(pop))
    assert set(dropped.item_ids) == sampling.key_items(pop) | sampling.credit_memo_items(pop)
    assert RULE_LINES[1] not in dropped.rationale


def test_cand_noisy_varies_judgments_by_run_index_only(world: Path, tmp_path: Path) -> None:
    prod = run(world, tmp_path, sp.STUB_PROD, "s1-mixD")
    runs = [run(world, tmp_path, sp.CAND_NOISY, "s1-mixD", i) for i in range(5)]
    outcomes = []
    for state in runs:
        assert state.exceptions is not None and prod.exceptions is not None
        assert state.exceptions.exceptions == prod.exceptions.exceptions
        assert state.samples == prod.samples and state.evidence == prod.evidence
        outcomes.append(tuple(j.outcome for j in state.exceptions.judgments))
    assert len(set(outcomes)) > 1
    again = run(world, tmp_path / "again", sp.CAND_NOISY, "s1-mixD", 3)
    assert again.exceptions == runs[3].exceptions
    assert again.workpaper == runs[3].workpaper


def test_profiles_match_prod_away_from_their_node(world: Path, tmp_path: Path) -> None:
    prod = run(world, tmp_path, sp.STUB_PROD, "s1-mixD")
    cutoff = run(world, tmp_path, sp.CAND_CUTOFF, "s1-mixD")
    assert cutoff.samples is not None and prod.samples is not None
    assert set(cutoff.samples.item_ids) < set(prod.samples.item_ids)
    good = run(world, tmp_path, sp.CAND_GOOD, "s1-mixD")
    assert (good.samples, good.evidence, good.tieouts) == (
        prod.samples,
        prod.evidence,
        prod.tieouts,
    )


def test_required_recall_detail_names_the_rule_of_the_missed_ids(world: Path, tmp_path: Path):
    g = grade(run(world, tmp_path, sp.CAND_CUTOFF, "s1-mixB"))
    [recall] = [s for s in g.scores if s.metric == "required_recall"]
    assert recall.detail.startswith("missing required ids (rules: cutoff_window): ")
