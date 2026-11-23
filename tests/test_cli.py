from collections.abc import Iterator
from pathlib import Path

import pytest
from cases import CASE_D

import auditlab.cli
from auditlab.cli import main

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def dotenv_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    calls: list[int] = []
    monkeypatch.setattr(auditlab.cli, "load_dotenv", lambda: calls.append(1) or True)
    yield calls


def test_run_case_prints_the_workpaper_with_markers_resolved(
    engagements: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], dotenv_calls: list[int]
) -> None:
    code = main(
        [
            "run-case",
            CASE_D.case_id,
            "--model",
            "prod",
            "--engagements-dir",
            str(engagements),
            "--runs-dir",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "[[" not in out
    assert CASE_D.answer_key.required_sample_ids[0] in out
    assert "$" in out
    assert "SIMULATED" in out
    assert dotenv_calls == [1]


def test_only_the_cli_loads_dotenv() -> None:
    callers = [
        str(p.relative_to(REPO))
        for p in sorted((REPO / "auditlab").rglob("*.py"))
        if "load_dotenv" in p.read_text(encoding="utf-8")
    ]
    assert callers == ["auditlab/cli.py"]
