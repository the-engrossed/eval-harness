"""Content-addressed response cache. Keyed per run_index so N runs are N samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from auditlab.domain.enums import FailureOrigin
from auditlab.models.base import ModelClient, ModelRequest, ModelResponse, canonical_json

__all__ = [
    "CacheEntry",
    "CachedClient",
    "FailureEntry",
    "ResponseCache",
    "cache_key",
    "canonical_json",
]


def cache_key(
    *,
    provider: str,
    model_id: str,
    prompt_version: str,
    node: str,
    canonical_input: str,
    run_index: int,
    variant: str = "",
) -> str:
    """`variant` names request options that change the output (e.g. reasoning_effort=none);
    it is left out when empty, so keys without options stay what they were."""
    parts: list[object] = [provider, model_id, prompt_version, node, canonical_input, run_index]
    if variant:
        parts.append(variant)
    return hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()


def request_key(provider: str, model_id: str, request: ModelRequest, variant: str = "") -> str:
    return cache_key(
        provider=provider,
        model_id=model_id,
        prompt_version=request.prompt_version,
        node=request.node.value,
        canonical_input=request.canonical_input(),
        run_index=request.run_index,
        variant=variant,
    )


class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model_id: str
    node: str
    run_index: int
    output: dict[str, JsonValue]
    tokens_in: int
    tokens_out: int
    latency_ms: int
    provider_model_id: str
    simulated: bool

    def to_response(self, schema: type[BaseModel]) -> ModelResponse:
        return ModelResponse(
            output=schema.model_validate(self.output),
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            latency_ms=self.latency_ms,
            provider_model_id=self.provider_model_id,
            cached=True,
            simulated=self.simulated,
        )


class FailureEntry(BaseModel):
    """A call that failed while recording (SPEC.md §6.4): replay re-raises it as recorded.

    Written by `python -m evals.backfill_failures` from a run's scored step errors. A live
    CachedClient never serves one: it calls the model again and replaces it on success.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["failure"] = "failure"
    provider: str
    model_id: str
    node: str
    run_index: int
    failure_origin: FailureOrigin
    message: str
    source_run_id: str


class ResponseCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def _read(self, key: str) -> dict[str, object] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        raw: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        return raw

    def get(self, key: str) -> CacheEntry | None:
        """The recorded response; None on a miss or a failure entry."""
        raw = self._read(key)
        if raw is None or raw.get("kind") == "failure":
            return None
        return CacheEntry.model_validate(raw)

    def get_failure(self, key: str) -> FailureEntry | None:
        raw = self._read(key)
        if raw is None or raw.get("kind") != "failure":
            return None
        return FailureEntry.model_validate(raw)

    def put_failure(self, key: str, entry: FailureEntry) -> None:
        if self.get(key) is not None:
            raise FileExistsError(f"{key} holds a recorded response; a failure never replaces it")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def put(self, key: str, entry: CacheEntry) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry.model_dump_json(indent=2) + "\n", encoding="utf-8")


class CachedClient:
    """Serves from the cache when it can; otherwise calls the wrapped client and stores."""

    def __init__(self, inner: ModelClient, cache: ResponseCache) -> None:
        self.inner = inner
        self.cache = cache
        self.provider = inner.provider
        self.model_id = inner.model_id
        self.simulated = inner.simulated
        self.variant: str = getattr(inner, "variant", "")

    def complete(self, request: ModelRequest) -> ModelResponse:
        key = request_key(self.provider, self.model_id, request, self.variant)
        entry = self.cache.get(key)
        if entry is not None:
            return entry.to_response(request.output_schema)
        response = self.inner.complete(request)
        self.cache.put(
            key,
            CacheEntry(
                provider=self.provider,
                model_id=self.model_id,
                node=request.node.value,
                run_index=request.run_index,
                output=response.output.model_dump(mode="json"),
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                latency_ms=response.latency_ms,
                provider_model_id=response.provider_model_id,
                simulated=response.simulated,
            ),
        )
        return response
