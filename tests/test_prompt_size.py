"""The pre-call prompt size estimate (auditlab/models/prompt_size.py)."""

from auditlab.domain.enums import Node
from auditlab.domain.models import SampleSelection
from auditlab.models.base import ModelRequest, canonical_json
from auditlab.models.prompt_size import estimate_prompt_tokens


def _request(system: str, payload: dict[str, object]) -> ModelRequest:
    return ModelRequest(
        case_id="s1-mixA",
        node=Node.SELECT_SAMPLES,
        prompt_version="v1",
        system=system,
        payload=payload,  # type: ignore[arg-type]
        output_schema=SampleSelection,
        run_index=0,
        seed=7,
    )


def test_estimate_counts_system_payload_and_schema_bytes() -> None:
    request = _request("Select the sample.", {"rule": "key items"})
    sent = (
        request.system
        + request.payload_json()
        + canonical_json(SampleSelection.model_json_schema())
    )
    assert estimate_prompt_tokens(request, 4.0) == len(sent.encode("utf-8")) // 4
    assert estimate_prompt_tokens(request, 2.5) == int(len(sent.encode("utf-8")) / 2.5)


def test_estimate_grows_with_the_payload() -> None:
    small = estimate_prompt_tokens(_request("s", {"x": "a"}), 4.0)
    big = estimate_prompt_tokens(_request("s", {"x": "a" * 40_000}), 4.0)
    assert big - small == 10_000
