"""Request options that change a model's output, as one string for the cache key."""

from __future__ import annotations


def request_variant(reasoning_effort: str | None, max_output_tokens: int | None = None) -> str:
    parts = []
    if reasoning_effort is not None:
        parts.append(f"reasoning_effort={reasoning_effort}")
    if max_output_tokens is not None:
        parts.append(f"max_output_tokens={max_output_tokens}")
    return ";".join(parts)
