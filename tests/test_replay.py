from pathlib import Path

import pytest
from test_model_cache import CountingClient, request

from auditlab.models.cache import CachedClient, ResponseCache
from auditlab.models.replay import ReplayMissError, ReplayModel


def test_replay_serves_a_recorded_response(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    recorded = CachedClient(CountingClient(), cache).complete(request())
    replay = ReplayModel(cache, provider="openai", model_id="fake-model")
    served = replay.complete(request())
    assert served.cached is True
    assert served.output == recorded.output
    assert served.simulated is False


def test_replay_miss_raises_with_the_key_and_node(tmp_path: Path) -> None:
    replay = ReplayModel(ResponseCache(tmp_path), provider="openai", model_id="fake-model")
    with pytest.raises(ReplayMissError, match=r"select_samples.*run_index=3.*key [0-9a-f]{64}"):
        replay.complete(request(run_index=3))


def test_replay_has_no_model_to_fall_back_to() -> None:
    assert not any("client" in name or "inner" in name for name in vars(ReplayModel))


def test_replay_finds_only_responses_recorded_with_the_same_variant(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    inner = CountingClient()
    inner.variant = "reasoning_effort=none"  # type: ignore[attr-defined]
    CachedClient(inner, cache).complete(request())
    same = ReplayModel(cache, provider="openai", model_id="fake-model", variant=inner.variant)  # type: ignore[attr-defined]
    assert same.complete(request()).cached is True
    with pytest.raises(ReplayMissError):
        ReplayModel(cache, provider="openai", model_id="fake-model").complete(request())


def _failure(**overrides):  # type: ignore[no-untyped-def]
    from auditlab.domain.enums import FailureOrigin
    from auditlab.models.cache import FailureEntry

    data = {
        "provider": "openai",
        "model_id": "fake-model",
        "node": "select_samples",
        "run_index": 0,
        "failure_origin": FailureOrigin.INFRA,
        "message": "openai fake-model at select_samples: timed out: Request timed out.",
        "source_run_id": "rec-1",
    }
    data.update(overrides)
    return FailureEntry(**data)


def test_replay_reraises_a_recorded_failure_with_its_origin_and_exact_message(
    tmp_path: Path,
) -> None:
    from auditlab.domain.enums import FailureOrigin
    from auditlab.models.cache import request_key
    from auditlab.models.errors import RecordedFailure

    cache = ResponseCache(tmp_path)
    key = request_key("openai", "fake-model", request())
    cache.put_failure(key, _failure())
    replay = ReplayModel(cache, provider="openai", model_id="fake-model")
    with pytest.raises(RecordedFailure) as err:
        replay.complete(request())
    assert err.value.failure_origin is FailureOrigin.INFRA
    assert str(err.value) == _failure().message
    agent = _failure(failure_origin=FailureOrigin.AGENT, message="prompt too large: 9 tokens")
    cache.put_failure(request_key("openai", "fake-model", request(run_index=1)), agent)
    with pytest.raises(RecordedFailure) as err:
        replay.complete(request(run_index=1))
    assert err.value.failure_origin is FailureOrigin.AGENT


def test_a_live_cached_client_retries_past_a_failure_entry_and_replaces_it(tmp_path: Path) -> None:
    from auditlab.models.cache import request_key

    cache = ResponseCache(tmp_path)
    key = request_key("openai", "fake-model", request())
    cache.put_failure(key, _failure())
    assert cache.get(key) is None and cache.get_failure(key) is not None
    inner = CountingClient()
    CachedClient(inner, cache).complete(request())
    assert inner.calls == 1
    assert cache.get(key) is not None and cache.get_failure(key) is None


def test_a_failure_entry_never_overwrites_a_recorded_response(tmp_path: Path) -> None:
    from auditlab.models.cache import request_key

    cache = ResponseCache(tmp_path)
    CachedClient(CountingClient(), cache).complete(request())
    with pytest.raises(FileExistsError, match="recorded response"):
        cache.put_failure(request_key("openai", "fake-model", request()), _failure())
