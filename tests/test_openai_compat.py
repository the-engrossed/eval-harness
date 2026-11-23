import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from auditlab.domain.enums import FailureOrigin, Node
from auditlab.domain.models import SampleSelection
from auditlab.models.base import ModelRequest
from auditlab.models.config import ModelConfig
from auditlab.models.errors import ModelOutputError, ProviderError
from auditlab.models.openai_compat import KEYLESS_API_KEY, OpenAICompatClient

CASE_ID = "s1-mixD"
GOOD = {"item_ids": ["INV-1001", "INV-1002"], "rationale": ["key items", "cutoff window"]}


def _config(**extra: Any) -> ModelConfig:
    return ModelConfig(
        config_id="local",
        provider="openai_compat",
        model_id="local-model",
        prompt_version="v1",
        simulated=False,
        base_url="http://localhost:11434/v1",
        **extra,
    )


def _request() -> ModelRequest:
    return ModelRequest(
        case_id=CASE_ID,
        node=Node.SELECT_SAMPLES,
        prompt_version="v1",
        system="Select the sample.",
        payload={"sampling_rule": "key items", "population_summary": {"invoices": []}},
        output_schema=SampleSelection,
        run_index=0,
        seed=7,
    )


def _completion(content: str | None, *, usage: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "local-model:served",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    return body


class Transport:
    def __init__(self, respond: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.respond = respond
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.respond(request)


def _client(
    respond: Callable[[httpx2.Request], httpx2.Response],
    env: dict[str, str] | None = None,
    **extra: Any,
) -> tuple[OpenAICompatClient, Transport]:
    transport = Transport(respond)
    client = OpenAICompatClient.from_config(
        _config(**extra),
        env or {},
        http_client=httpx2.Client(transport=httpx2.MockTransport(transport)),
    )
    return client, transport


def _ok(content: str) -> Callable[[httpx2.Request], httpx2.Response]:
    return lambda request: httpx2.Response(200, json=_completion(content))


def test_parsed_output_token_counts_and_model_id() -> None:
    client, transport = _client(_ok(json.dumps(GOOD)))
    response = client.complete(_request())
    assert response.output == SampleSelection.model_validate(GOOD)
    assert (response.tokens_in, response.tokens_out) == (120, 30)
    assert response.provider_model_id == "local-model:served"
    assert response.latency_ms >= 0
    assert (response.cached, response.simulated) == (False, False)
    assert (client.provider, client.model_id, client.simulated) == (
        "openai_compat",
        "local-model",
        False,
    )

    [sent] = transport.requests
    assert str(sent.url) == "http://localhost:11434/v1/chat/completions"
    body = json.loads(sent.content)
    assert body["model"] == "local-model"
    assert body["seed"] == 7
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert json.loads(body["messages"][1]["content"]) == _request().payload
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "SampleSelection"
    assert fmt["json_schema"]["schema"] == SampleSelection.model_json_schema()


def test_case_id_never_appears_in_the_request() -> None:
    client, transport = _client(_ok(json.dumps(GOOD)))
    client.complete(_request())
    [sent] = transport.requests
    assert CASE_ID not in sent.content.decode()
    assert CASE_ID not in str(sent.url)
    assert all(CASE_ID not in v for v in sent.headers.values())


def test_http_500_raises_provider_error_once_without_retrying() -> None:
    client, transport = _client(lambda r: httpx2.Response(500, json={"error": "boom"}))
    with pytest.raises(ProviderError) as err:
        client.complete(_request())
    assert err.value.failure_origin is FailureOrigin.INFRA
    assert "500" in str(err.value)
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "exc", [httpx2.ConnectError("connection refused"), httpx2.ReadTimeout("too slow")]
)
def test_connection_failures_and_timeouts_raise_provider_error(exc: Exception) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        raise exc

    client, transport = _client(respond)
    with pytest.raises(ProviderError) as err:
        client.complete(_request())
    assert err.value.failure_origin is FailureOrigin.INFRA
    assert len(transport.requests) == 1


def test_a_response_without_usage_is_a_provider_error() -> None:
    client, _ = _client(lambda r: httpx2.Response(200, json=_completion("{}", usage=False)))
    with pytest.raises(ProviderError, match="usage"):
        client.complete(_request())


@pytest.mark.parametrize(
    "raw",
    [
        "Sure! Here are the samples: INV-1001",
        json.dumps({"item_ids": [], "rationale": ["none"]}),
        json.dumps({"items": ["INV-1001"]}),
        "",
    ],
    ids=["not-json", "fails-validator", "wrong-shape", "empty"],
)
def test_bad_output_raises_model_output_error_with_the_raw_text(raw: str) -> None:
    client, _ = _client(_ok(raw))
    with pytest.raises(ModelOutputError) as err:
        client.complete(_request())
    assert err.value.failure_origin is FailureOrigin.AGENT
    assert err.value.raw_text == raw
    assert err.value.node is Node.SELECT_SAMPLES


def test_a_null_content_is_a_model_output_error() -> None:
    client, _ = _client(lambda r: httpx2.Response(200, json=_completion(None)))
    with pytest.raises(ModelOutputError) as err:
        client.complete(_request())
    assert err.value.raw_text == ""


def test_api_key_env_value_is_sent_as_the_bearer_token() -> None:
    client, transport = _client(
        _ok(json.dumps(GOOD)), env={"OPEN_MODEL_KEY": "sk-test-123"}, api_key_env="OPEN_MODEL_KEY"
    )
    client.complete(_request())
    assert transport.requests[0].headers["authorization"] == "Bearer sk-test-123"


def test_keyless_servers_get_the_named_unused_key() -> None:
    client, transport = _client(_ok(json.dumps(GOOD)))
    client.complete(_request())
    assert transport.requests[0].headers["authorization"] == f"Bearer {KEYLESS_API_KEY}"


@pytest.mark.parametrize("env", [{}, {"OPEN_MODEL_KEY": ""}])
def test_a_missing_api_key_env_var_raises_at_build_time(env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="OPEN_MODEL_KEY"):
        OpenAICompatClient.from_config(_config(api_key_env="OPEN_MODEL_KEY"), env)


def test_reasoning_effort_is_sent_only_when_configured() -> None:
    client, transport = _client(_ok(json.dumps(GOOD)), reasoning_effort="none")
    client.complete(_request())
    [sent] = transport.requests
    assert json.loads(sent.content)["reasoning_effort"] == "none"
    assert client.variant == "reasoning_effort=none"
    plain, plain_transport = _client(_ok(json.dumps(GOOD)))
    plain.complete(_request())
    [sent] = plain_transport.requests
    assert "reasoning_effort" not in json.loads(sent.content)
    assert plain.variant == ""


def test_max_output_tokens_uses_the_configured_field_and_a_cutoff_is_an_agent_error() -> None:
    from auditlab.models.errors import ModelOutputError

    def cut(request: httpx2.Request) -> httpx2.Response:
        body = _completion('{"item_ids": ["INV-1')
        body["choices"][0]["finish_reason"] = "length"
        return httpx2.Response(200, json=body)

    client, transport = _client(cut, max_output_tokens=64)
    with pytest.raises(ModelOutputError, match=r"cut off at max output tokens \(64\)"):
        client.complete(_request())
    assert json.loads(transport.requests[0].content)["max_tokens"] == 64
    openai_style, t2 = _client(
        _ok(json.dumps(GOOD)), max_output_tokens=64, max_output_tokens_field="max_completion_tokens"
    )
    openai_style.complete(_request())
    sent = json.loads(t2.requests[0].content)
    assert sent["max_completion_tokens"] == 64 and "max_tokens" not in sent
    assert openai_style.variant == "max_output_tokens=64"


def test_a_prompt_over_the_context_window_is_an_agent_error_and_no_call_is_made() -> None:
    from auditlab.models.errors import PromptTooLargeError
    from auditlab.models.prompt_size import estimate_prompt_tokens

    estimate = estimate_prompt_tokens(_request(), 2.5)
    client, transport = _client(
        _ok(json.dumps(GOOD)), context_window_tokens=estimate - 1, prompt_bytes_per_token=2.5
    )
    with pytest.raises(PromptTooLargeError) as err:
        client.complete(_request())
    assert err.value.failure_origin is FailureOrigin.AGENT
    assert err.value.node is Node.SELECT_SAMPLES
    assert str(err.value).endswith(
        f"prompt too large: {estimate} tokens (estimated) > context window {estimate - 1}"
    )
    assert transport.requests == []


def test_a_prompt_that_fits_the_context_window_is_sent() -> None:
    from auditlab.models.prompt_size import estimate_prompt_tokens

    estimate = estimate_prompt_tokens(_request(), 2.5)
    client, transport = _client(
        _ok(json.dumps(GOOD)), context_window_tokens=estimate, prompt_bytes_per_token=2.5
    )
    client.complete(_request())
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "extra", [{"context_window_tokens": 1000}, {"prompt_bytes_per_token": 3.1}]
)
def test_context_window_and_bytes_per_token_come_together(extra: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="context_window_tokens and prompt_bytes_per_token"):
        _config(**extra)
