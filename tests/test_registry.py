import re
from pathlib import Path

import pytest

from auditlab.config import MODELS_TOML
from auditlab.models.config import load_models_config
from auditlab.models.openai_compat import OpenAICompatClient
from auditlab.models.registry import build_client

PROD = """
[production]
config_id = "prod"

[[model]]
config_id = "prod"
provider = "stub"
model_id = "stub-prod"
prompt_version = "v1"
simulated = true
"""

REAL_MODEL = """
[[model]]
config_id = "cand-real"
provider = "openai"
model_id = "provider-model-x"
prompt_version = "v1"
simulated = false
"""

PRICE = """
[[price]]
model_id = "provider-model-x"
input_usd_per_mtok = {input}
output_usd_per_mtok = 10.0
source_url = "https://example.invalid/pricing"
as_of = "2026-11-01"
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "models.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repo_config_has_prod_and_the_three_candidate_stubs() -> None:
    config = load_models_config(MODELS_TOML)
    assert config.production_config_id == "prod"
    got = {m.config_id: (m.provider, m.model_id, m.simulated) for m in config.models}
    assert got == {
        "prod": ("stub", "stub-prod", True),
        "stub-cand-good": ("stub", "stub-cand-good", True),
        "stub-cand-cutoff": ("stub", "stub-cand-cutoff", True),
        "stub-cand-noisy": ("stub", "stub-cand-noisy", True),
        "openai-gpt-6.2-sol": ("openai_compat", "gpt-6.2-sol", False),
        "local-qwen3-8b": ("openai_compat", "qwen3:8b", False),
        "openai-gpt-6-luna-judge": ("openai_compat", "gpt-6-luna", False),
        "replay-openai-gpt-6.2-sol": ("replay", "gpt-6.2-sol", False),
        "replay-local-qwen3-8b": ("replay", "qwen3:8b", False),
        "replay-openai-gpt-6-luna-judge": ("replay", "gpt-6-luna", False),
    }
    for m in config.models:
        if m.provider == "replay":
            recorded = config.model(m.config_id.removeprefix("replay-"))
            assert m.replay_provider == recorded.provider
            assert (m.model_id, m.prompt_version) == (recorded.model_id, recorded.prompt_version)
            assert m.reasoning_effort == recorded.reasoning_effort
            assert m.max_output_tokens == recorded.max_output_tokens
    local = [m for m in config.models if m.config_id.startswith("local-")]
    assert {m.reasoning_effort for m in local} == {"none"}
    real_configs = [m for m in config.models if not m.simulated]
    assert {m.max_output_tokens for m in real_configs} == {32768}
    live = [m for m in real_configs if m.provider == "openai_compat"]
    windows = {m.config_id: m.context_window_tokens for m in live}
    assert windows == {
        "openai-gpt-6.2-sol": 1_050_000,
        "local-qwen3-8b": 32_768,
        "openai-gpt-6-luna-judge": 1_050_000,
    }
    ratios = {m.config_id: m.prompt_bytes_per_token for m in live}
    assert ratios == {
        "openai-gpt-6.2-sol": 3.10,
        "local-qwen3-8b": 2.33,
        "openai-gpt-6-luna-judge": 3.10,
    }
    prices = {
        p.model_id: (p.input_micros_per_mtok, p.output_micros_per_mtok) for p in config.prices
    }
    assert prices == {
        "stub-prod": (3_000_000, 15_000_000),
        "stub-cand-good": (1_000_000, 5_000_000),
        "stub-cand-cutoff": (3_000_000, 15_000_000),
        "stub-cand-noisy": (3_000_000, 15_000_000),
        "gpt-6.2-sol": (2_000_000, 10_000_000),
        "qwen3:8b": (0, 0),
        "gpt-6-luna": (100_000, 500_000),
    }
    stubs = {"stub-prod", "stub-cand-good", "stub-cand-cutoff", "stub-cand-noisy"}
    assert {p.source_url for p in config.prices if p.model_id in stubs} == {"simulated"}
    real = {p.model_id: p for p in config.prices if p.model_id not in stubs}
    for model_id in ("gpt-6.2-sol", "gpt-6-luna"):
        assert real[model_id].source_url == "https://developers.openai.com/api/docs/pricing"
    assert {real[m].source_url for m in ("qwen3:8b",)} == {"self-hosted"}


def test_real_provider_without_a_price_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"provider-model-x.*\[\[price\]\]"):
        load_models_config(write(tmp_path, PROD + REAL_MODEL))


def test_replay_without_a_price_is_refused(tmp_path: Path) -> None:
    replay = REAL_MODEL.replace('"openai"', '"replay"') + 'replay_provider = "openai"\n'
    with pytest.raises(ValueError, match=r"provider-model-x.*\[\[price\]\]"):
        load_models_config(write(tmp_path, PROD + replay))


def test_prices_become_integer_micros(tmp_path: Path) -> None:
    config = load_models_config(write(tmp_path, PROD + REAL_MODEL + PRICE.format(input="2.5")))
    [price] = config.prices
    assert price.input_micros_per_mtok == 2_500_000
    assert price.output_micros_per_mtok == 10_000_000
    assert type(price.input_micros_per_mtok) is int


def test_price_finer_than_a_micro_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="micro"):
        load_models_config(write(tmp_path, PROD + REAL_MODEL + PRICE.format(input="0.0000001")))


def test_duplicate_config_ids_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate config_id: prod"):
        load_models_config(write(tmp_path, PROD + PROD.split("\n", 3)[3]))


def test_production_must_name_a_configured_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="production config_id"):
        text = PROD.replace('config_id = "prod"\n\n', 'config_id = "x"\n\n', 1)
        load_models_config(write(tmp_path, text))


def test_registry_builds_the_prod_stub_and_refuses_unknowns(tmp_path: Path) -> None:
    config = load_models_config(MODELS_TOML)
    client = build_client("prod", config, cache_dir=tmp_path, engagements_dir=tmp_path)
    assert (client.provider, client.model_id, client.simulated) == ("stub", "stub-prod", True)
    with pytest.raises(ValueError, match="unknown config_id: nope"):
        build_client("nope", config, cache_dir=tmp_path, engagements_dir=tmp_path)


def test_unknown_stub_profile_and_unbuilt_provider_fail_loudly(tmp_path: Path) -> None:
    text = PROD.replace('model_id = "stub-prod"', 'model_id = "stub-mystery"')
    config = load_models_config(write(tmp_path, text))
    with pytest.raises(ValueError, match="stub-mystery"):
        build_client("prod", config, cache_dir=tmp_path, engagements_dir=tmp_path)

    real = load_models_config(write(tmp_path, PROD + REAL_MODEL + PRICE.format(input="2.5")))
    with pytest.raises(NotImplementedError, match="openai"):
        build_client("cand-real", real, cache_dir=tmp_path, engagements_dir=tmp_path)


COMPAT = REAL_MODEL.replace('"openai"', '"openai_compat"')
LOCAL_URL = 'base_url = "http://localhost:11434/v1"\n'


def test_openai_compat_needs_a_base_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="base_url"):
        load_models_config(write(tmp_path, PROD + COMPAT + PRICE.format(input="0")))


def test_base_url_and_api_key_env_belong_to_openai_compat_only(tmp_path: Path) -> None:
    for extra in (LOCAL_URL, 'api_key_env = "X"\n', "timeout_s = 30\n"):
        stub = PROD.replace("simulated = true\n", "simulated = true\n" + extra)
        with pytest.raises(ValueError, match="openai_compat"):
            load_models_config(write(tmp_path, stub))


def test_openai_compat_without_a_price_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"provider-model-x.*\[\[price\]\]"):
        load_models_config(write(tmp_path, PROD + COMPAT + LOCAL_URL))


def test_registry_builds_openai_compat_and_checks_the_key_env_at_build_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = PROD + COMPAT + LOCAL_URL + 'api_key_env = "OPEN_MODEL_KEY"\n' + PRICE.format(input="0")
    config = load_models_config(write(tmp_path, text))
    monkeypatch.delenv("OPEN_MODEL_KEY", raising=False)
    with pytest.raises(ValueError, match="OPEN_MODEL_KEY"):
        build_client("cand-real", config, cache_dir=tmp_path, engagements_dir=tmp_path)
    monkeypatch.setenv("OPEN_MODEL_KEY", "sk-test")
    client = build_client("cand-real", config, cache_dir=tmp_path, engagements_dir=tmp_path)
    assert isinstance(client, OpenAICompatClient)
    assert (client.provider, client.model_id, client.simulated) == (
        "openai_compat",
        "provider-model-x",
        False,
    )


def test_the_commented_local_examples_parse_once_filled_in(tmp_path: Path) -> None:
    text = MODELS_TOML.read_text(encoding="utf-8")
    marker = "# --- Example local configs"
    assert marker in text
    examples = text.split(marker, 1)[1]
    examples = examples[examples.index("# [[model]]") :]
    body = "\n".join(line.removeprefix("#").removeprefix(" ") for line in examples.splitlines())
    filled = re.sub(r"<([a-z ]+)>", lambda m: m.group(1).replace(" ", "-"), body)
    config = load_models_config(write(tmp_path, PROD + filled))
    local = [m for m in config.models if m.provider == "openai_compat"]
    assert [m.base_url for m in local] == ["http://localhost:11434/v1", "http://localhost:1234/v1"]
    assert all(m.timeout_s == 300 and m.api_key_env is None for m in local)
    assert all(p.input_micros_per_mtok == p.output_micros_per_mtok == 0 for p in config.prices)
    assert {p.source_url for p in config.prices} == {"self-hosted"}
    assert {str(p.as_of) for p in config.prices} == {"2026-11-17"}
