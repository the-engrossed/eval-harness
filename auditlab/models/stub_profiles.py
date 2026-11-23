"""Candidate stub fault profiles (SPEC.md §6.3). Each acts at one node; every other node
answers exactly as stub-prod does. No answer-key access here: each fault rewrites the
oracle's output using only what the node was sent.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any

from auditlab.domain.enums import JudgmentOutcome
from auditlab.domain.models import ExceptionSet, SampleSelection

STUB_PROD = "stub-prod"
CAND_GOOD = "stub-cand-good"
CAND_CUTOFF = "stub-cand-cutoff"
CAND_NOISY = "stub-cand-noisy"
PROFILES = frozenset({STUB_PROD, CAND_GOOD, CAND_CUTOFF, CAND_NOISY})

# Copies of data/sampling.py (auditlab never imports data/); a test pins them equal.
KEY_ITEM_THRESHOLD_CENTS = 2_500_000
CUTOFF_START = dt.date(2025, 12, 27)
CUTOFF_END = dt.date(2026, 1, 5)
RULE_TWO_PREFIX = "Rule two:"
GOOD_TOKEN_NUM, GOOD_TOKEN_DEN = 3, 5


def drop_cutoff_window(selection: SampleSelection, summary: dict[str, Any]) -> SampleSelection:
    """stub-cand-cutoff at select_samples: ignore rule 2. An id in the cutoff window stays
    only if rule 1 (key item) or rule 3 (credit memo) also selects it."""
    invoices = {i["invoice_id"]: i for i in summary["invoices"]}
    memo_ids = set(summary["credit_memo_invoice_ids"])

    def keep(item_id: str) -> bool:
        inv = invoices.get(item_id)
        if inv is None:
            return True
        in_window = CUTOFF_START <= dt.date.fromisoformat(inv["invoice_date"]) <= CUTOFF_END
        key_item = inv["amount_cents"] >= KEY_ITEM_THRESHOLD_CENTS
        return not in_window or key_item or item_id in memo_ids

    return SampleSelection(
        item_ids=[i for i in selection.item_ids if keep(i)],
        rationale=[r for r in selection.rationale if not r.startswith(RULE_TWO_PREFIX)],
    )


def reroll_judgments(exceptions: ExceptionSet, rng: random.Random) -> ExceptionSet:
    """stub-cand-noisy at flag_exceptions: each judgment outcome is a seeded coin flip."""
    outcomes = list(JudgmentOutcome)
    judgments = [
        j.model_copy(update={"outcome": rng.choice(outcomes)}) for j in exceptions.judgments
    ]
    return ExceptionSet(exceptions=exceptions.exceptions, judgments=judgments)


def scale_tokens(profile: str, tokens: int) -> int:
    """stub-cand-good: cheaper simulated token counts at every model node."""
    if profile == CAND_GOOD:
        return tokens * GOOD_TOKEN_NUM // GOOD_TOKEN_DEN
    return tokens
