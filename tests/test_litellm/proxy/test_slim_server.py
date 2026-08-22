from __future__ import annotations

import json
from collections.abc import AsyncIterator
from textwrap import dedent
from types import SimpleNamespace

from fastapi.testclient import TestClient

from litellm.proxy.slim_server import (
    _anthropic_error_response,
    _anthropic_sse_chunk,
    _apply_litellm_settings,
    _strip_tools_for_no_tools_models,
    create_slim_app,
)


class DumpableResponse:
    def model_dump(self, exclude_none: bool = False):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "customer-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }
            ],
        }


class RouterError(Exception):
    status_code = 429


class FakeRouter:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    async def acompletion(self, **kwargs):
        return await self._dispatch(kwargs)

    async def anthropic_messages(self, **kwargs):
        return await self._dispatch(kwargs)

    async def _dispatch(self, kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


async def fake_stream() -> AsyncIterator[dict]:
    yield {"choices": [{"delta": {"content": "pon"}}]}
    yield {"choices": [{"delta": {"content": "g"}}]}


async def fake_anthropic_stream() -> AsyncIterator[object]:
    yield b"event: message_start\ndata: {}\n\n"
    yield {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}


def write_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        dedent("""
            general_settings:
              master_key: sk-master
            model_list:
              - model_name: customer-model
                litellm_params:
                  model: openai/gpt-4.1-mini
                  api_key: sk-one
              - model_name: customer-model
                litellm_params:
                  model: openai/gpt-4.1-mini
                  api_key: sk-two
            router_settings:
              routing_strategy: simple-shuffle
              num_retries: 1
              cooldown_time: 30
            """),
        encoding="utf-8",
    )
    return config_path


def make_client(tmp_path, router: FakeRouter):
    app = create_slim_app(
        config_path=write_config(tmp_path),
        router_factory=lambda _config: router,
    )
    return TestClient(app), router


def test_apply_litellm_settings_sets_scalars_and_skips_complex() -> None:
    target = SimpleNamespace()

    skipped = _apply_litellm_settings(
        {
            "drop_params": True,
            "request_timeout": 120,
            "telemetry": False,
            "success_callback": ["prometheus"],
        },
        target,
    )

    assert target.drop_params is True
    assert target.request_timeout == 120
    assert target.telemetry is False
    assert not hasattr(target, "success_callback")
    assert skipped == ("success_callback",)


def test_anthropic_error_response_builds_anthropic_body() -> None:
    response = _anthropic_error_response(429, "rate limited")

    assert response.status_code == 429
    assert json.loads(response.body) == {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "rate limited"},
    }


def test_strip_tools_removes_tools_and_tool_choice_for_no_tools_model() -> None:
    data = {
        "model": "ox-alpha-free",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "x"}],
        "tool_choice": {"type": "auto"},
    }

    _strip_tools_for_no_tools_models(data)

    assert "tools" not in data
    assert "tool_choice" not in data
    assert data["model"] == "ox-alpha-free"
    assert data["messages"]


def test_strip_tools_leaves_other_models_untouched() -> None:
    data = {
        "model": "deepseek-v4-flash-go",
        "tools": [{"name": "x"}],
        "tool_choice": {"type": "auto"},
    }

    _strip_tools_for_no_tools_models(data)

    assert "tools" in data
    assert "tool_choice" in data


def test_strip_tools_noop_when_no_tools_present() -> None:
    data = {"model": "ox-alpha-free", "messages": [{"role": "user", "content": "hi"}]}

    _strip_tools_for_no_tools_models(data)

    assert "tools" not in data
    assert data["messages"]


def test_anthropic_sse_chunk_passes_through_bytes() -> None:
    chunk = b"event: message_start\ndata: {}\n\n"

    assert _anthropic_sse_chunk(chunk) == "event: message_start\ndata: {}\n\n"


def test_anthropic_sse_chunk_frames_dicts_with_event_type() -> None:
    chunk = {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "hi"},
    }

    assert _anthropic_sse_chunk(chunk) == (
        'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
    )


def test_health_routes_do_not_require_auth(tmp_path):
    client, _router = make_client(tmp_path, FakeRouter(DumpableResponse()))

    with client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/health/liveliness").json() == {"status": "ok"}
        assert client.get("/health/readiness").json() == {"status": "ok"}


def test_default_router_factory_accepts_configured_simple_shuffle(tmp_path):
    app = create_slim_app(config_path=write_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_models_route_requires_auth(tmp_path):
    client, _router = make_client(tmp_path, FakeRouter(DumpableResponse()))

    with client:
        response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_models_route_accepts_x_api_key_header(tmp_path):
    """Anthropic SDK sends `x-api-key` by default; the slim proxy must accept
    it alongside Authorization: Bearer / x-litellm-api-key."""
    client, _router = make_client(tmp_path, FakeRouter(DumpableResponse()))

    with client:
        good = client.get("/v1/models", headers={"x-api-key": "sk-master"})
        bad = client.get("/v1/models", headers={"x-api-key": "sk-wrong"})

    assert good.status_code == 200
    assert bad.status_code == 401


def test_models_route_returns_single_public_model_for_multiple_deployments(tmp_path):
    client, _router = make_client(tmp_path, FakeRouter(DumpableResponse()))

    with client:
        response = client.get(
            "/v1/models", headers={"Authorization": "Bearer sk-master"}
        )

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "customer-model", "object": "model"}],
    }


def test_models_route_allows_anonymous_when_master_key_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        dedent("""
            model_list:
              - model_name: customer-model
                litellm_params:
                  model: openai/gpt-4.1-mini
            """),
        encoding="utf-8",
    )
    app = create_slim_app(
        config_path=config_path,
        router_factory=lambda _config: FakeRouter(DumpableResponse()),
    )

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200


def test_chat_completion_defaults_to_configured_public_model(tmp_path):
    router = FakeRouter(DumpableResponse())
    client, _router = make_client(tmp_path, router)

    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-master"},
            json={"messages": [{"role": "user", "content": "ping"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "pong"
    assert router.calls == [
        {"messages": [{"role": "user", "content": "ping"}], "model": "customer-model"}
    ]


def test_chat_completion_preserves_requested_public_model(tmp_path):
    router = FakeRouter({"id": "chatcmpl-test", "choices": []})
    client, _router = make_client(tmp_path, router)

    with client:
        response = client.post(
            "/chat/completions",
            headers={"x-litellm-api-key": "sk-master"},
            json={"model": "customer-model", "messages": []},
        )

    assert response.status_code == 200
    assert router.calls == [{"model": "customer-model", "messages": []}]


def test_models_route_returns_multiple_models_when_config_has_multi_model(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        dedent("""
            model_list:
              - model_name: agent-model
                litellm_params:
                  model: anthropic/deepseek-v4-flash
                  api_key: os.environ/ARK_KEY_1
                  api_base: https://example.com/api/plan
              - model_name: coding-model
                litellm_params:
                  model: anthropic/deepseek-v4-flash
                  api_key: os.environ/ARK_KEY_2
                  api_base: https://example.com/api/coding
        """),
        encoding="utf-8",
    )
    app = create_slim_app(
        config_path=config_path,
        router_factory=lambda _config: FakeRouter(DumpableResponse()),
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/models", headers={"Authorization": "Bearer sk-master"}
        )

    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert set(ids) == {"agent-model", "coding-model"}


def test_chat_completion_requires_model_when_multi_model_config(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")
    router = FakeRouter(DumpableResponse())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        dedent("""
            model_list:
              - model_name: agent-model
                litellm_params:
                  model: anthropic/deepseek-v4-flash
                  api_key: os.environ/ARK_KEY_1
                  api_base: https://example.com/api/plan
              - model_name: coding-model
                litellm_params:
                  model: anthropic/deepseek-v4-flash
                  api_key: os.environ/ARK_KEY_2
                  api_base: https://example.com/api/coding
        """),
        encoding="utf-8",
    )
    app = create_slim_app(
        config_path=config_path,
        router_factory=lambda _config: router,
    )

    with TestClient(app) as client:
        # Missing model -> 400
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-master"},
            json={"messages": [{"role": "user", "content": "ping"}]},
        )
        assert response.status_code == 400

        # Explicit model -> 200
        response2 = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-master"},
            json={
                "model": "agent-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        assert response2.status_code == 200
        assert router.calls[-1]["model"] == "agent-model"


def test_chat_completion_maps_router_exception_to_openai_error(tmp_path):
    router = FakeRouter(RouterError("rate limited"))
    client, _router = make_client(tmp_path, router)

    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-master"},
            json={"model": "customer-model", "messages": []},
        )

    assert response.status_code == 429
    assert "rate limited" in response.json()["error"]["message"]


def test_streaming_chat_completion_serializes_sse_chunks(tmp_path):
    router = FakeRouter(fake_stream())
    client, _router = make_client(tmp_path, router)

    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-master"},
            json={"model": "customer-model", "stream": True, "messages": []},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"choices":[{"delta":{"content":"pon"}}]}' in response.text
    assert 'data: {"choices":[{"delta":{"content":"g"}}]}' in response.text
    assert response.text.rstrip().endswith("data: [DONE]")
    assert router.calls == [{"model": "customer-model", "stream": True, "messages": []}]


def test_messages_non_streaming_returns_response_and_defaults_model(tmp_path):
    anthropic_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "pong"}],
        "model": "customer-model",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    router = FakeRouter(anthropic_response)
    client, _router = make_client(tmp_path, router)

    with client:
        response = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer sk-master"},
            json={"max_tokens": 10, "messages": [{"role": "user", "content": "ping"}]},
        )

    assert response.status_code == 200
    assert response.json() == anthropic_response
    assert router.calls == [
        {
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "ping"}],
            "model": "customer-model",
        }
    ]


def test_messages_non_streaming_strips_total_tokens_when_enabled(tmp_path, monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "strip_anthropic_total_tokens", True)
    anthropic_response = {
        "id": "msg_1",
        "type": "message",
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    }
    router = FakeRouter(anthropic_response)
    client, _router = make_client(tmp_path, router)

    with client:
        response = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer sk-master"},
            json={"max_tokens": 10, "messages": []},
        )

    assert response.status_code == 200
    assert "total_tokens" not in response.json()["usage"]


def test_messages_non_streaming_maps_exception_to_anthropic_error(tmp_path):
    router = FakeRouter(RouterError("rate limited"))
    client, _router = make_client(tmp_path, router)

    with client:
        response = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer sk-master"},
            json={"max_tokens": 10, "messages": []},
        )

    assert response.status_code == 429
    assert response.json() == {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "rate limited"},
    }


def test_messages_streaming_passes_through_sse_without_done(tmp_path):
    router = FakeRouter(fake_anthropic_stream())
    client, _router = make_client(tmp_path, router)

    with client:
        response = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer sk-master"},
            json={
                "model": "customer-model",
                "max_tokens": 10,
                "stream": True,
                "messages": [],
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message_start\ndata: {}\n\n" in response.text
    assert "event: content_block_delta" in response.text
    assert "[DONE]" not in response.text
    assert router.calls == [
        {
            "model": "customer-model",
            "max_tokens": 10,
            "stream": True,
            "messages": [],
        }
    ]


def test_messages_no_auth_returns_anthropic_shaped_401(tmp_path):
    client, _router = make_client(tmp_path, FakeRouter(DumpableResponse()))

    with client:
        response = client.post("/v1/messages", json={"max_tokens": 10, "messages": []})

    assert response.status_code == 401
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "authentication_error",
            "message": "Invalid or missing API key",
        },
    }


def test_messages_count_tokens_returns_input_tokens(tmp_path, monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: 42)
    client, _router = make_client(tmp_path, FakeRouter(DumpableResponse()))

    with client:
        response = client.post(
            "/v1/messages/count_tokens",
            headers={"Authorization": "Bearer sk-master"},
            json={
                "model": "customer-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 42}


def test_messages_count_tokens_requires_messages(tmp_path):
    client, _router = make_client(tmp_path, FakeRouter(DumpableResponse()))

    with client:
        response = client.post(
            "/v1/messages/count_tokens",
            headers={"Authorization": "Bearer sk-master"},
            json={"model": "customer-model", "messages": []},
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
