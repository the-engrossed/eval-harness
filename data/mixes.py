"""Exception mixes (SPEC.md §4.4). Together they cover every ExceptionType."""

from __future__ import annotations

from auditlab.domain.enums import ExceptionType as ET

MIXES: dict[str, tuple[ET, ...]] = {
    "A": (ET.DUPLICATE_INVOICE, ET.CUTOFF_ERROR),
    "B": (ET.AMOUNT_MISMATCH, ET.MISSING_EVIDENCE, ET.BILL_AND_HOLD),
    "C": (ET.CUTOFF_ERROR, ET.AMOUNT_MISMATCH, ET.PARTIAL_SHIPMENT),
    "D": (ET.DUPLICATE_INVOICE, ET.MISSING_EVIDENCE, ET.CUTOFF_ERROR, ET.BILL_AND_HOLD),
    "E": (ET.AMOUNT_MISMATCH, ET.PARTIAL_SHIPMENT),
}

TAGS: dict[ET, str] = {
    ET.DUPLICATE_INVOICE: "duplicate",
    ET.CUTOFF_ERROR: "cutoff",
    ET.AMOUNT_MISMATCH: "amount_mismatch",
    ET.MISSING_EVIDENCE: "missing_evidence",
    ET.BILL_AND_HOLD: "bill_and_hold",
    ET.PARTIAL_SHIPMENT: "partial_shipment",
}
JUDGMENT_TAG = "judgment"
