"""Closed vocabularies shared by the agent, scorers, gate and scorecard."""

from __future__ import annotations

from enum import StrEnum


class ExceptionType(StrEnum):
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"
    CUTOFF_ERROR = "CUTOFF_ERROR"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    BILL_AND_HOLD = "BILL_AND_HOLD"
    PARTIAL_SHIPMENT = "PARTIAL_SHIPMENT"


CRITICAL_TYPES: frozenset[ExceptionType] = frozenset(
    {
        ExceptionType.DUPLICATE_INVOICE,
        ExceptionType.CUTOFF_ERROR,
        ExceptionType.AMOUNT_MISMATCH,
        ExceptionType.MISSING_EVIDENCE,
    }
)
JUDGMENT_TYPES: frozenset[ExceptionType] = frozenset(ExceptionType) - CRITICAL_TYPES


class JudgmentOutcome(StrEnum):
    EXCEPTION = "exception"
    NO_EXCEPTION_WITH_RATIONALE = "no_exception_with_rationale"


class Node(StrEnum):
    """Members are declared in graph order."""

    SELECT_SAMPLES = "select_samples"
    EXTRACT_EVIDENCE = "extract_evidence"
    TIE_OUT = "tie_out"
    FLAG_EXCEPTIONS = "flag_exceptions"
    DRAFT_WORKPAPER = "draft_workpaper"


class Verdict(StrEnum):
    SHIP = "SHIP"
    DO_NOT_SHIP = "DO_NOT_SHIP"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class MetricKind(StrEnum):
    OUTCOME = "outcome"
    PROCESS = "process"


class FailureOrigin(StrEnum):
    AGENT = "agent"
    INFRA = "infra"
    DATA = "data"
