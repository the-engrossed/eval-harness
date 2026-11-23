"""Deterministic guard on model-written workpaper prose.

G1 any digit left after removing well-formed markers is a violation.
G2 every [[...]] token must be a well-formed marker that resolves:
   [[ev:<id>]] against the extracted evidence ids, [[item:<id>]] against the sampled item ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

MARKER = re.compile(r"\[\[(ev|item):([A-Za-z0-9_.-]+)\]\]")
_BRACKET_TOKEN = re.compile(r"\[\[.*?\]\]")
_DIGIT = re.compile(r"\d")


class GuardCode(StrEnum):
    UNBOUND_NUMERAL = "G1_unbound_numeral"
    BAD_MARKER = "G2_bad_marker"


@dataclass(frozen=True)
class GuardViolation:
    code: GuardCode
    message: str


def validate_prose(
    prose: str,
    evidence_ids: frozenset[str],
    *,
    item_ids: frozenset[str] = frozenset(),
) -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    known = {"ev": evidence_ids, "item": item_ids}

    for token in _BRACKET_TOKEN.findall(prose):
        match = MARKER.fullmatch(token)
        if match is None:
            violations.append(
                GuardViolation(
                    GuardCode.BAD_MARKER, f"{token} is not an [[ev:<id>]] or [[item:<id>]] marker."
                )
            )
        elif match.group(2) not in known[match.group(1)]:
            violations.append(GuardViolation(GuardCode.BAD_MARKER, f"{token} does not resolve."))

    bare = _DIGIT.findall(MARKER.sub("", prose))
    if bare:
        violations.append(
            GuardViolation(
                GuardCode.UNBOUND_NUMERAL,
                f"bare digits in prose: {bare[:5]}; use [[ev:<id>]] for amounts and "
                "[[item:<id>]] for invoices.",
            )
        )
    return violations
