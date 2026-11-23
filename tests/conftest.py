from pathlib import Path

import pytest
from cases import CASE_D
from matrix_run import build_matrix_run

from data.generate import write_case


@pytest.fixture(scope="session")
def engagements(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A temporary engagements folder holding s1-mixD."""
    root = tmp_path_factory.mktemp("data") / "engagements"
    write_case(CASE_D, root)
    return root


@pytest.fixture(scope="session")
def matrix_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """runs/<RUN_ID>/ for prod + the three candidate stubs on three cases, N=5."""
    return build_matrix_run(tmp_path_factory.mktemp("matrix"))
