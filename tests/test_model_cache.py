from pathlib import Path
from typing import Any

import pytest

from auditlab.domain.enums import Node
from auditlab.domain.models import SampleSelection
from auditlab.models.base import ModelRequest, ModelResponse
from auditlab.models.cache import CachedClient, ResponseCache, cache_key, canonical_json


def request(**overrides: Any) -> ModelRequest:
    data: dict[str, Any] = {
        "case_id": "case-x",
        "node": Node.SELECT_SAMPLES,
        "prompt_version": "v1",
        "system": "Select the required sample.",
        "payload": {"sampling_rule": "rule text", "invoice_count": 3},
        "output_schema": SampleSelection,
        "run_index": 0,
        "seed": 7,
    }
    data.update(overrides)
    return ModelRequest(**data)


class CountingClient:
    provider = "openai"
    model_id = "fake-model"
    simulated = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, req: ModelRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            output=SampleSelection(item_ids=["INV-1"], rationale=["rule one"]),
            tokens_in=10,
            tokens_out=5,
            latency_ms=120,
            provider_model_id="fake-model-2026",
            cached=False,
            simulated=False,
        )


def key_for(req: ModelRequest, provider: str = "openai", model_id: str = "fake-model") -> str:
    return cache_key(
        provider=provider,
        model_id=model_id,
        prompt_version=req.prompt_version,
        node=req.node,
        canonical_input=req.canonical_input(),
        run_index=req.run_index,
    )


def test_canonical_json_ignores_key_order() -> None:
    assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == canonical_json(
        {"a": [1, {"c": 3, "d": 2}], "b": 1}
    )


def test_key_is_stable_for_equal_requests() -> None:
    a = request(payload={"x": 1, "y": 2})
    b = request(payload={"y": 2, "x": 1})
    assert key_for(a) == key_for(b)


@pytest.mark.parametrize(
    "change",
    [
        {"run_index": 1},
        {"prompt_version": "v2"},
        {"node": Node.EXTRACT_EVIDENCE},
        {"payload": {"sampling_rule": "other"}},
        {"system": "Different system prompt."},
        {"case_id": "case-y"},
    ],
    ids=lambda c: next(iter(c)),
)
def test_every_key_component_changes_the_key(change: dict[str, Any]) -> None:
    assert key_for(request()) != key_for(request(**change))


def test_provider_and_model_id_change_the_key() -> None:
    req = request()
    assert key_for(req) != key_for(req, provider="anthropic")
    assert key_for(req) != key_for(req, model_id="other-model")


def test_miss_calls_the_model_and_stores_then_hit_serves_from_cache(tmp_path: Path) -> None:
    inner = CountingClient()
    client = CachedClient(inner, ResponseCache(tmp_path))
    first = client.complete(request())
    assert inner.calls == 1 and first.cached is False

    second = client.complete(request())
    assert inner.calls == 1 and second.cached is True
    assert second.output == first.output
    assert (second.tokens_in, second.tokens_out, second.provider_model_id) == (
        10,
        5,
        "fake-model-2026",
    )

    client.complete(request(run_index=1))
    assert inner.calls == 2


def test_a_request_variant_changes_the_key_and_empty_keeps_old_keys() -> None:
    req = request()
    from auditlab.models.cache import request_key

    base = request_key("openai_compat", "m", req)
    assert request_key("openai_compat", "m", req, "") == base
    assert request_key("openai_compat", "m", req, "reasoning_effort=none") != base
