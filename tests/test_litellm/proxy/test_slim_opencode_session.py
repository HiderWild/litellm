from __future__ import annotations

import json
from textwrap import dedent

from fastapi.testclient import TestClient

from litellm.proxy.slim_server import create_slim_app


class DumpableResponse:
    def model_dump(self, exclude_none: bool = False):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "go-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }
            ],
        }


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


ANTHROPIC_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}

SESSION_HEADER = "x-opencode-session"


def write_config(tmp_path, extra_models: str = ""):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        dedent(f"""
            general_settings:
              master_key: sk-master
            model_list:
              - model_name: go-model
                litellm_params:
                  model: openai/deepseek-v4-flash
                  api_key: sk-go-one
                  api_base: https://opencode.ai/zen/go/v1
              - model_name: anthropic-go-model
                litellm_params:
                  model: anthropic/deepseek-v4-pro
                  api_key: sk-go-two
                  api_base: https://opencode.ai/zen/go
              - model_name: plain-model
                litellm_params:
                  model: openai/gpt-4.1-mini
                  api_key: sk-plain
                  api_base: https://one.example.test/v1
              {extra_models}
            """),
        encoding="utf-8",
    )
    return config_path


def make_client(tmp_path, response=None, extra_models: str = ""):
    router = FakeRouter(response or DumpableResponse())
    app = create_slim_app(
        config_path=write_config(tmp_path, extra_models),
        router_factory=lambda _config: router,
    )
    return TestClient(app), router


def chat_kwargs(router: FakeRouter) -> dict:
    assert len(router.calls) == 1
    return router.calls[0]


def test_chat_forwards_inbound_opencode_session_header(tmp_path):
    client, router = make_client(tmp_path)

    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-master",
                SESSION_HEADER: "opencode-cli-abc123",
            },
            json={
                "model": "go-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 200
    assert chat_kwargs(router)["extra_headers"] == {
        SESSION_HEADER: "opencode-cli-abc123"
    }


def test_messages_forwards_inbound_opencode_session_header(tmp_path):
    client, router = make_client(tmp_path, response=ANTHROPIC_RESPONSE)

    with client:
        response = client.post(
            "/v1/messages",
            headers={
                "Authorization": "Bearer sk-master",
                SESSION_HEADER: "opencode-cli-def456",
            },
            json={
                "model": "anthropic-go-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            },
        )

    assert response.status_code == 200
    assert chat_kwargs(router)["extra_headers"] == {
        SESSION_HEADER: "opencode-cli-def456"
    }


def test_metadata_user_id_session_used_when_no_header(tmp_path):
    client, router = make_client(tmp_path, response=ANTHROPIC_RESPONSE)

    with client:
        response = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer sk-master"},
            json={
                "model": "anthropic-go-model",
                "max_tokens": 10,
                "metadata": {
                    "user_id": json.dumps(
                        {"device_id": "dev-1", "session_id": "zcode-session-42"}
                    )
                },
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert (
        chat_kwargs(router)["extra_headers"][SESSION_HEADER] == "zcode-session-42"
    )


def test_derived_session_stable_within_conversation_and_distinct_across(tmp_path):
    client, router = make_client(tmp_path)
    first_turn = [{"role": "user", "content": "Refactor the parser module"}]
    second_turn = [
        {"role": "user", "content": "Refactor the parser module"},
        {"role": "assistant", "content": "Sure, which part?"},
        {"role": "user", "content": "Start with the tokenizer"},
    ]
    other_conversation = [{"role": "user", "content": "Write release notes instead"}]

    with client:
        for messages in (first_turn, second_turn, other_conversation):
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-master"},
                json={"model": "go-model", "messages": messages},
            )

    session_ids = [call["extra_headers"][SESSION_HEADER] for call in router.calls]
    assert all(sid.startswith("litellm-") for sid in session_ids)
    assert session_ids[0] == session_ids[1]
    assert session_ids[2] != session_ids[0]


def test_plain_model_gets_no_injection(tmp_path):
    client, router = make_client(tmp_path)

    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-master",
                SESSION_HEADER: "opencode-cli-abc123",
            },
            json={
                "model": "plain-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 200
    assert "extra_headers" not in chat_kwargs(router)


def test_no_user_message_falls_back_to_static_session(tmp_path):
    client, router = make_client(tmp_path, response=ANTHROPIC_RESPONSE)

    with client:
        response = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer sk-master"},
            json={
                "model": "anthropic-go-model",
                "max_tokens": 10,
                "messages": [{"role": "assistant", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert (
        chat_kwargs(router)["extra_headers"][SESSION_HEADER]
        == "litellm-unattributed"
    )


def test_client_extra_headers_preserved_on_injection(tmp_path):
    client, router = make_client(tmp_path)

    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-master",
                SESSION_HEADER: "opencode-cli-xyz",
            },
            json={
                "model": "go-model",
                "messages": [{"role": "user", "content": "ping"}],
                "extra_headers": {"X-Custom": "abc"},
            },
        )

    assert response.status_code == 200
    assert chat_kwargs(router)["extra_headers"] == {
        "X-Custom": "abc",
        SESSION_HEADER: "opencode-cli-xyz",
    }


def test_invalid_inbound_header_falls_through_to_derivation(tmp_path):
    client, router = make_client(tmp_path)

    with client:
        client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-master",
                SESSION_HEADER: "x" * 200,
            },
            json={
                "model": "go-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-master",
                SESSION_HEADER: "bad\x01value",
            },
            json={
                "model": "go-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    session_ids = [call["extra_headers"][SESSION_HEADER] for call in router.calls]
    assert all(sid.startswith("litellm-") for sid in session_ids)
    assert session_ids[0] == session_ids[1]


def test_mixed_model_group_targets_whole_group(tmp_path):
    mixed_group = """
              - model_name: mixed-model
                litellm_params:
                  model: openai/some-model
                  api_key: sk-go
                  api_base: https://opencode.ai/zen/go/v1
              - model_name: mixed-model
                litellm_params:
                  model: openai/some-model
                  api_key: sk-ark
                  api_base: https://ark.cn-beijing.volces.com/api/plan
    """
    client, router = make_client(tmp_path, extra_models=mixed_group)

    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-master"},
            json={
                "model": "mixed-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 200
    assert chat_kwargs(router)["extra_headers"][SESSION_HEADER].startswith(
        "litellm-"
    )
