import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ALLOWED = {"auditlab/models/stub.py", "auditlab/domain/answer_key.py"}
ACCESS = re.compile(r"auditlab\.domain\.answer_key|domain\s+import\s+answer_key|answer_key\.json")


def test_only_the_stub_model_reads_answer_keys_inside_auditlab() -> None:
    offenders = [
        rel
        for path in sorted((REPO / "auditlab").rglob("*.py"))
        if (rel := str(path.relative_to(REPO))) not in ALLOWED
        and ACCESS.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
