"""Simulated oracle models (SPEC.md §6.3). Every response is labelled simulated.

The only model code allowed to read answer_key.json. It answers each node from the answer
key, restricted to the ids it was given, so a fault upstream stays visible downstream.
"""

from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from auditlab.domain.answer_key import AnswerKey
from auditlab.domain.enums import Node
from auditlab.domain.models import (
    DroppedField,
    EvidenceField,
    EvidenceSet,
    ExceptionSet,
    FlaggedException,
    JudgmentDecision,
    SampleSelection,
    Workpaper,
)
from auditlab.models.base import ModelRequest, ModelResponse, canonical_json
from auditlab.models.stub_profiles import (
    CAND_CUTOFF,
    CAND_NOISY,
    PROFILES,
    drop_cutoff_window,
    reroll_judgments,
    scale_tokens,
)

RULE_LINES = [
    "Rule one: every invoice at or above the key-item threshold.",
    "Rule two: every invoice dated inside the cutoff window.",
    "Rule three: every invoice with an entry in the credit memo log.",
]
OPENERS = ["Procedures performed:", "We performed the following procedures:"]
CLOSERS = [
    "Conclusion: revenue is fairly stated for the items tested, subject to the matters above.",
    "Conclusion: subject to the matters above, the items tested support recorded revenue.",
]


def evidence_id(item_id: str, field: str) -> str:
    return f"ev-{item_id}-{field}"


@lru_cache(maxsize=64)
def _load_key(path: Path) -> AnswerKey:
    if not path.is_file():
        raise FileNotFoundError(f"answer key not found: {path}. Run `make data` first.")
    return AnswerKey.model_validate_json(path.read_text(encoding="utf-8"))


def _rng(request: ModelRequest) -> random.Random:
    digest = hashlib.sha256(f"{request.case_id}|{request.run_index}|{request.seed}".encode())
    return random.Random(int.from_bytes(digest.digest()[:8], "big"))


def _select(key: AnswerKey, _payload: dict[str, Any]) -> SampleSelection:
    return SampleSelection(item_ids=list(key.required_sample_ids), rationale=list(RULE_LINES))


def _extract(key: AnswerKey, payload: dict[str, Any]) -> EvidenceSet:
    wanted = list(payload["item_ids"])
    fields = [
        EvidenceField(evidence_id=evidence_id(e.item_id, e.field), **e.model_dump())
        for e in key.evidence
        if e.item_id in wanted
    ]
    found = {f.item_id for f in fields}
    dropped = [
        DroppedField(item_id=i, field="all", reason="no supporting fields located")
        for i in wanted
        if i not in found
    ]
    return EvidenceSet(fields=fields, dropped=dropped)


def _flag(key: AnswerKey, payload: dict[str, Any]) -> ExceptionSet:
    seen = {f.item_id for f in EvidenceSet.model_validate(payload["evidence"]).fields}
    exceptions = [
        FlaggedException(
            item_id=e.item_id,
            type=e.type,
            rationale=f"{e.type.value.replace('_', ' ').lower()} indicated by the evidence",
        )
        for e in key.expected_exceptions
        if e.item_id in seen
    ]
    judgments = [
        JudgmentDecision(
            item_id=j.item_id,
            type=j.type,
            outcome=j.allowed_outcomes[0],
            rationale="; ".join(j.rubric),
        )
        for j in key.judgment_items
        if j.item_id in seen
    ]
    return ExceptionSet(exceptions=exceptions, judgments=judgments)


def _amount(evidence: EvidenceSet, item_id: str) -> str:
    fields = [f for f in evidence.fields if f.item_id == item_id]
    preferred = [f for f in fields if f.field == "invoice_amount"] or fields
    return f"[[ev:{preferred[0].evidence_id}]]"


def _workpaper(rng: random.Random, payload: dict[str, Any]) -> Workpaper:
    evidence = EvidenceSet.model_validate(payload["evidence"])
    exceptions = ExceptionSet.model_validate(payload["exceptions"])
    items = sorted({f.item_id for f in evidence.fields})
    samples = ", ".join(f"[[item:{i}]] ({_amount(evidence, i)})" for i in items) or "none"
    flagged = [
        f"{e.type.value.replace('_', ' ').lower()} on [[item:{e.item_id}]]"
        for e in exceptions.exceptions
    ]
    flagged += [
        f"{j.outcome.value.replace('_', ' ')} on [[item:{j.item_id}]]" for j in exceptions.judgments
    ]
    prose = " ".join(
        [
            rng.choice(OPENERS),
            "selected items under the stated sampling rule, agreed each invoice to its bill",
            "of lading and the bank statement, and tied each invoice to the general ledger.",
            f"Samples tested: {samples}.",
            f"Matters noted: {'; '.join(flagged) if flagged else 'none'}.",
            rng.choice(CLOSERS),
        ]
    )
    return Workpaper(prose=prose, guard_violations=[])


class StubModel:
    provider = "stub"
    simulated = True

    def __init__(self, profile: str, engagements_dir: Path) -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown stub profile {profile!r}; available: {sorted(PROFILES)}")
        self.model_id = profile
        self.engagements_dir = engagements_dir

    def complete(self, request: ModelRequest) -> ModelResponse:
        key = _load_key(self.engagements_dir / request.case_id / "answer_key.json")
        payload: dict[str, Any] = dict(request.payload)
        output: BaseModel
        if request.node is Node.SELECT_SAMPLES:
            output = _select(key, payload)
            if self.model_id == CAND_CUTOFF:
                output = drop_cutoff_window(output, payload["population_summary"])
        elif request.node is Node.EXTRACT_EVIDENCE:
            output = _extract(key, payload)
        elif request.node is Node.FLAG_EXCEPTIONS:
            output = _flag(key, payload)
            if self.model_id == CAND_NOISY:
                output = reroll_judgments(output, _rng(request))
        elif request.node is Node.DRAFT_WORKPAPER:
            output = _workpaper(_rng(request), payload)
        else:
            raise ValueError(f"{request.node.value} is deterministic code and never calls a model")
        if not isinstance(output, request.output_schema):
            raise TypeError(
                f"{request.node.value} produces {type(output).__name__}, "
                f"request asked for {request.output_schema.__name__}"
            )
        tokens_in = scale_tokens(
            self.model_id, len(request.system + canonical_json(request.payload)) // 4
        )
        tokens_out = scale_tokens(self.model_id, len(output.model_dump_json()) // 4)
        return ModelResponse(
            output=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=40 + tokens_out // 20,
            provider_model_id=self.model_id,
            cached=False,
            simulated=True,
        )
