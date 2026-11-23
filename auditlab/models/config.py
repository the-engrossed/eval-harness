"""`config/models.toml` (SPEC.md §6.2). Prices become integer micros at this boundary."""

from __future__ import annotations

import datetime as dt
import tomllib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from auditlab.domain.base import DomainModel, Id, Text, require_unique

RealProvider = Literal["openai", "anthropic", "gemini", "openai_compat"]
Provider = Literal["stub", "replay", "openai", "anthropic", "gemini", "openai_compat"]
PRICED_PROVIDERS = frozenset({"openai", "anthropic", "gemini", "openai_compat", "replay"})


def usd_to_micros(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{label} must be a number, got {value!r}")
    try:
        micros = Decimal(str(value)) * 1_000_000
    except InvalidOperation as exc:
        raise ValueError(f"{label} is not a number: {value!r}") from exc
    if micros < 0 or micros != micros.to_integral_value():
        raise ValueError(f"{label} = {value!r} must be non-negative and a whole number of micros")
    return int(micros)


class ModelConfig(DomainModel):
    config_id: Id
    provider: Provider
    model_id: Id
    prompt_version: Id
    simulated: bool
    replay_provider: RealProvider | None = None
    base_url: Text | None = None
    api_key_env: Id | None = None
    timeout_s: int | None = Field(default=None, gt=0, strict=True)
    # Sent as `reasoning_effort` on /v1/chat/completions; Ollama documents "none" as thinking
    # off. Part of the cache key, so runs with different settings never share responses.
    reasoning_effort: Id | None = None
    # Per-call output cap, so a runaway generation ends as a scored agent error. The field name
    # differs by server: Ollama documents `max_tokens`; OpenAI deprecates it for
    # `max_completion_tokens` (which also counts reasoning tokens).
    max_output_tokens: int | None = Field(default=None, gt=0, strict=True)
    # Prompt limit checked before each call (auditlab/models/prompt_size.py); over it, the
    # step is a scored agent error and no request is sent. The estimate divides prompt bytes
    # by prompt_bytes_per_token, measured per config; the two are set together.
    context_window_tokens: int | None = Field(default=None, gt=0, strict=True)
    prompt_bytes_per_token: float | None = Field(default=None, gt=0)
    max_output_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.simulated != (self.provider == "stub"):
            raise ValueError(f"{self.config_id}: simulated must be true for stub models only")
        if (self.provider == "replay") != (self.replay_provider is not None):
            raise ValueError(f"{self.config_id}: replay_provider is required for replay only")
        real_only = {
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "context_window_tokens": self.context_window_tokens,
            "prompt_bytes_per_token": self.prompt_bytes_per_token,
        }
        for name, value in real_only.items():
            if value is not None and self.provider not in ("openai_compat", "replay"):
                raise ValueError(f"{self.config_id}: {name} applies to openai_compat/replay")
        if (self.context_window_tokens is None) != (self.prompt_bytes_per_token is None):
            raise ValueError(
                f"{self.config_id}: set context_window_tokens and prompt_bytes_per_token "
                "together (the pre-call check needs both)"
            )
        if self.provider == "openai_compat":
            if self.base_url is None:
                raise ValueError(f"{self.config_id}: openai_compat needs base_url")
        else:
            extras: dict[str, object] = {
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
                "timeout_s": self.timeout_s,
            }
            set_ = [k for k, v in extras.items() if v is not None]
            if set_:
                raise ValueError(
                    f"{self.config_id}: {', '.join(set_)} apply to provider openai_compat only"
                )
        return self


class Price(DomainModel):
    model_id: Id
    input_micros_per_mtok: int = Field(ge=0, strict=True)
    output_micros_per_mtok: int = Field(ge=0, strict=True)
    source_url: Text
    as_of: dt.date

    @classmethod
    def from_toml(cls, raw: dict[str, Any]) -> Price:
        fields = dict(raw)
        model_id = fields.get("model_id", "?")
        for side in ("input", "output"):
            usd = fields.pop(f"{side}_usd_per_mtok", None)
            if usd is None:
                raise ValueError(f"[[price]] {model_id}: {side}_usd_per_mtok is missing")
            label = f"[[price]] {model_id} {side}_usd_per_mtok"
            fields[f"{side}_micros_per_mtok"] = usd_to_micros(usd, label)
        return cls.model_validate(fields)


class ModelsConfig(DomainModel):
    production_config_id: Id
    models: list[ModelConfig] = Field(min_length=1)
    prices: list[Price]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        ids = require_unique("config_id", [m.config_id for m in self.models])
        if self.production_config_id not in ids:
            raise ValueError(
                f"production config_id {self.production_config_id!r} is not a [[model]]"
            )
        priced = require_unique("price model_id", [p.model_id for p in self.prices])
        for m in self.models:
            if m.provider in PRICED_PROVIDERS and m.model_id not in priced:
                raise ValueError(
                    f"model {m.model_id} ({m.config_id}, provider {m.provider}) has no "
                    "[[price]] entry; add one with source_url and as_of from the "
                    "provider's pricing page"
                )
        return self

    def model(self, config_id: str) -> ModelConfig:
        for m in self.models:
            if m.config_id == config_id:
                return m
        configured = [m.config_id for m in self.models]
        raise ValueError(f"unknown config_id: {config_id}; configured: {configured}")


def load_models_config(path: Path) -> ModelsConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    production = raw.get("production")
    if not isinstance(production, dict) or "config_id" not in production:
        raise ValueError(f"{path}: missing [production] config_id")
    return ModelsConfig(
        production_config_id=production["config_id"],
        models=raw.get("model", []),
        prices=[Price.from_toml(p) for p in raw.get("price", [])],
    )
