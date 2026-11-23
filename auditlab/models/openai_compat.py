"""OpenAI-compatible chat client: Ollama, LM Studio, vLLM, hosted open-model APIs.

The `openai` SDK is imported only when a client is built, so a missing extra only
breaks this provider. No retries: one request per call, failures surface as typed errors.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from auditlab.models.base import ModelRequest, ModelResponse
from auditlab.models.config import ModelConfig
from auditlab.models.errors import ModelOutputError, PromptTooLargeError, ProviderError
from auditlab.models.prompt_size import estimate_prompt_tokens
from auditlab.models.variant import request_variant

if TYPE_CHECKING:
    from openai import OpenAI

PROVIDER = "openai_compat"
DEFAULT_TIMEOUT_S = 120

KEYLESS_API_KEY = "unused"
"""Sent when no api_key_env is configured. The SDK refuses to build a client without a
key, and keyless local servers (Ollama, LM Studio) ignore whatever is sent."""


class OpenAICompatClient:
    provider = PROVIDER
    simulated = False

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        api_key: str,
        timeout_s: int,
        http_client: Any = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        max_output_tokens_field: str = "max_tokens",
        context_window_tokens: int | None = None,
        prompt_bytes_per_token: float | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "provider openai_compat needs the openai SDK: run `uv sync --extra openai`"
            ) from exc
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_output_tokens_field = max_output_tokens_field
        if (context_window_tokens is None) != (prompt_bytes_per_token is None):
            raise ValueError("context_window_tokens and prompt_bytes_per_token go together")
        self.context_window_tokens = context_window_tokens
        self.prompt_bytes_per_token = prompt_bytes_per_token
        self.variant = request_variant(reasoning_effort, max_output_tokens)
        self._sdk: OpenAI = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(timeout_s),
            max_retries=0,
            http_client=http_client,
        )

    @classmethod
    def from_config(
        cls, model: ModelConfig, env: Mapping[str, str], *, http_client: Any = None
    ) -> OpenAICompatClient:
        if model.base_url is None:
            raise ValueError(f"{model.config_id}: openai_compat needs base_url")
        api_key = KEYLESS_API_KEY
        if model.api_key_env is not None:
            api_key = env.get(model.api_key_env, "")
            if not api_key:
                raise ValueError(
                    f"{model.config_id}: api_key_env {model.api_key_env} is not set. "
                    f"Add {model.api_key_env}=... to .env or the environment."
                )
        return cls(
            model_id=model.model_id,
            base_url=model.base_url,
            api_key=api_key,
            timeout_s=model.timeout_s or DEFAULT_TIMEOUT_S,
            http_client=http_client,
            reasoning_effort=model.reasoning_effort,
            max_output_tokens=model.max_output_tokens,
            max_output_tokens_field=model.max_output_tokens_field,
            context_window_tokens=model.context_window_tokens,
            prompt_bytes_per_token=model.prompt_bytes_per_token,
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        import openai

        if self.context_window_tokens is not None and self.prompt_bytes_per_token is not None:
            tokens = estimate_prompt_tokens(request, self.prompt_bytes_per_token)
            if tokens > self.context_window_tokens:
                raise PromptTooLargeError(
                    self.model_id, request.node, tokens, self.context_window_tokens
                )
        schema = request.output_schema
        extra: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            extra["reasoning_effort"] = self.reasoning_effort
        if self.max_output_tokens is not None:
            extra[self.max_output_tokens_field] = self.max_output_tokens
        started = time.perf_counter_ns()
        try:
            completion = self._sdk.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.payload_json()},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": False,
                    },
                },
                seed=request.seed,
                **extra,
            )
        except openai.APIStatusError as exc:
            raise self._infra(request, f"HTTP {exc.status_code}: {exc.message}") from exc
        except openai.APITimeoutError as exc:
            raise self._infra(request, f"timed out: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise self._infra(request, f"connection failed: {exc}") from exc
        latency_ms = (time.perf_counter_ns() - started) // 1_000_000

        if not completion.choices:
            raise self._infra(request, "response has no choices")
        if completion.usage is None:
            raise self._infra(request, "response has no usage (token counts)")
        choice = completion.choices[0]
        raw = choice.message.content or ""
        if choice.finish_reason == "length":
            cap = (
                f"max output tokens ({self.max_output_tokens})"
                if self.max_output_tokens
                else "length"
            )
            reason = f"output cut off at {cap} after {completion.usage.completion_tokens} tokens"
            raise ModelOutputError(self.model_id, request.node, raw, reason)
        try:
            output = schema.model_validate_json(raw)
        except ValidationError as exc:
            reason = f"{exc.error_count()} validation error(s); first: {exc.errors()[0]['msg']}"
            raise ModelOutputError(self.model_id, request.node, raw, reason) from exc
        return ModelResponse(
            output=output,
            tokens_in=completion.usage.prompt_tokens,
            tokens_out=completion.usage.completion_tokens,
            latency_ms=latency_ms,
            provider_model_id=completion.model,
            cached=False,
            simulated=False,
        )

    def _infra(self, request: ModelRequest, message: str) -> ProviderError:
        return ProviderError(PROVIDER, self.model_id, request.node, message)
