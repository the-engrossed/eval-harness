import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LISTING = re.compile(r"\b(iterdir|glob|rglob|listdir|scandir|walk)\b")
ANSWER_KEY = re.compile(r"answer_key")


def _sources() -> list[Path]:
    return [*sorted((REPO / "auditlab" / "agent").rglob("*.py")), REPO / "auditlab/docs/store.py"]


def test_agent_and_doc_store_never_list_a_folder() -> None:
    offenders = [
        str(p.relative_to(REPO))
        for p in _sources()
        if LISTING.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_agent_never_touches_answer_keys() -> None:
    agent_files = [
        p for p in (REPO / "auditlab" / "agent").rglob("*") if p.suffix in {".py", ".md"}
    ]
    offenders = [
        str(p.relative_to(REPO))
        for p in agent_files
        if ANSWER_KEY.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
