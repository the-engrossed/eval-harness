"""All 20 suite cases, built once per test session."""

import pytest

from data.generate import build_case
from data.mixes import MIXES

CASES = [build_case(seed, mix) for seed in range(1, 5) for mix in MIXES]
each_case = pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
CASE_D = next(c for c in CASES if c.case_id == "s1-mixD")
