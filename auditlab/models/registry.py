"""Build a ModelClient for a config_id from config/models.toml."""

from __future__ import annotations

import os
from pathlib import Path

from auditlab.config import DEMO_CACHE_DIR, ENGAGEMENTS_DIR
from auditlab.models.base import ModelClient
from auditlab.models.cache import ResponseCache
from auditlab.models.config import ModelsConfig
from auditlab.models.replay import ReplayModel
from auditlab.models.stub import StubModel
from auditlab.models.variant import request_variant


def build_client(
    config_id: str,
    config: ModelsConfig,
    *,
    cache_dir: Path = DEMO_CACHE_DIR,
    engagements_dir: Path = ENGAGEMENTS_DIR,
) -> ModelClient:
    model = config.model(config_id)
    if model.provider == "stub":
        return StubModel(model.model_id, engagements_dir)
    if model.provider == "replay":
        if model.replay_provider is None:
            raise ValueError(f"{config_id}: replay config without replay_provider")
        return ReplayModel(
            ResponseCache(cache_dir),
            provider=model.replay_provider,
            model_id=model.model_id,
            variant=request_variant(model.reasoning_effort, model.max_output_tokens),
        )
    if model.provider == "openai_compat":
        from auditlab.models.openai_compat import OpenAICompatClient

        return OpenAICompatClient.from_config(model, os.environ)
    raise NotImplementedError(
        f"provider {model.provider!r} for config {config_id} has no client yet: "
        f"auditlab/models/{model.provider}_client.py lands in a later task"
    )
