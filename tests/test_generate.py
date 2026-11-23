from pathlib import Path

import pytest

from auditlab.domain.answer_key import AnswerKey
from auditlab.domain.population import Population
from data.generate import build_case, main, write_case

FILES = ("population.json", "answer_key.json")


def test_same_seed_and_mix_give_identical_bytes(tmp_path: Path) -> None:
    a = write_case(build_case(1, "C"), tmp_path / "a")
    b = write_case(build_case(1, "C"), tmp_path / "b")
    for name in FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_different_seeds_give_different_populations(tmp_path: Path) -> None:
    a = write_case(build_case(1, "C"), tmp_path / "a")
    b = write_case(build_case(2, "C"), tmp_path / "b")
    assert (a / "population.json").read_bytes() != (b / "population.json").read_bytes()


def test_cli_writes_files_that_load_into_the_domain_models(tmp_path: Path) -> None:
    assert main(["--seed", "2", "--mix", "D", "--out", str(tmp_path)]) == 0
    case_dir = tmp_path / "s2-mixD"
    Population.model_validate_json((case_dir / "population.json").read_text(encoding="utf-8"))
    AnswerKey.model_validate_json((case_dir / "answer_key.json").read_text(encoding="utf-8"))


def test_unknown_mix_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mix: F"):
        build_case(1, "F")
    with pytest.raises(SystemExit):
        main(["--seed", "1", "--mix", "F"])
