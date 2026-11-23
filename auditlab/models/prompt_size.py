"""Pre-call prompt size estimate, for the context-window check in real clients.

No tokenizer is a dependency (SPEC.md §3), so the count is UTF-8 bytes of system prompt +
user payload + output schema, divided by the config's `prompt_bytes_per_token`. That ratio
is measured, the same way for every config: the bytes of the s1-mixA select_samples
request (its input does not depend on any model) over the prompt tokens the provider
reported for it. See config/models.toml for each config's measurement.
"""

from __future__ import annotations

from auditlab.models.base import ModelRequest, canonical_json


def prompt_bytes(request: ModelRequest) -> int:
    sent = (
        request.system
        + request.payload_json()
        + canonical_json(request.output_schema.model_json_schema())
    )
    return len(sent.encode("utf-8"))


def estimate_prompt_tokens(request: ModelRequest, bytes_per_token: float) -> int:
    if bytes_per_token <= 0:
        raise ValueError(f"bytes_per_token must be positive, got {bytes_per_token}")
    return int(prompt_bytes(request) / bytes_per_token)
