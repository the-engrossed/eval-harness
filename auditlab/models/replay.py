"""Replay: serves recorded responses only. A miss raises; there is no model behind it.

A recorded failure (FailureEntry) re-raises as RecordedFailure with its recorded origin and
message, so a call that failed while recording fails the same way on replay.
"""

from __future__ import annotations

from auditlab.models.base import ModelRequest, ModelResponse
from auditlab.models.cache import ResponseCache, request_key
from auditlab.models.errors import RecordedFailure


class ReplayMissError(LookupError):
    pass


class ReplayModel:
    simulated = False

    def __init__(
        self, cache: ResponseCache, *, provider: str, model_id: str, variant: str = ""
    ) -> None:
        self.cache = cache
        self.provider = provider
        self.model_id = model_id
        self.variant = variant

    def complete(self, request: ModelRequest) -> ModelResponse:
        key = request_key(self.provider, self.model_id, request, self.variant)
        entry = self.cache.get(key)
        if entry is None:
            failure = self.cache.get_failure(key)
            if failure is not None:
                raise RecordedFailure(failure.failure_origin, failure.message)
            raise ReplayMissError(
                f"no recorded response for {self.provider}/{self.model_id} node "
                f"{request.node.value} run_index={request.run_index} (key {key}) in "
                f"{self.cache.root}. Record it by running this config against the real "
                "provider once, then commit the cache entry."
            )
        return entry.to_response(request.output_schema)
