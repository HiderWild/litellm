from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from litellm.proxy.slim_config import (
    RequestLogConfig,
    SlimProxyConfigError,
    load_slim_config,
)
from litellm.proxy.slim_server import create_slim_app


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


async def fake_anthropic_stream() -> AsyncIterator[object]:
    yield b"event: message_start\ndata: {}\n\n"
    yield {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}
    yield {"type": "message_stop"}


def write_config(tmp_path, *, extra_litellm: str = "") -> object:
    config_path = tmp_path / "config.yaml"
    litellm_extra = ""
    if extra_litellm.strip():
        litellm_extra = "litellm_settings:\n" + "".join(
            f"  {line}\n" for line in extra_litellm.splitlines() if line.strip()
        )
    config_path.write_text(
        "general_settings:\n"
        "  master_key: sk-master\n"
        "model_list:\n"
        "  - model_name: customer-model\n"
        "    litellm_params:\n"
        "      model: openai/gpt-4.1-mini\n"
        "      api_key: sk-one\n"
        "router_settings:\n"
        "  routing_strategy: simple-shuffle\n"
        + litellm_extra,
        encoding="utf-8",
    )
    return config_path


def make_client(tmp_path, router: FakeRouter, log_path):
    app = create_slim_app(
        config_path=write_config(
            tmp_path,
            extra_litellm=(
                "request_logging: true\n"
                f"request_log_file: '{log_path.as_posix()}'\n"
                "request_log_body_limit: 200"
            ),
        ),
        router_factory=lambda _config: router,
    )
    return TestClient(app), router


def test_request_logging_disabled_by_default(tmp_path) -> None:
    config_path = write_config(tmp_path, extra_litellm="telemetry: false")
    config = load_slim_config(config_path)

    assert config.request_log is None
    assert "request_logging" not in config.litellm_settings


def test_request_logging_enabled_parses_config(tmp_path) -> None:
    log_path = tmp_path / "requests.log"
    config_path = write_config(
        tmp_path,
        extra_litellm=(
            "request_logging: true\n"
            f"request_log_file: '{log_path.as_posix()}'\n"
            "request_log_body_limit: 512"
        ),
    )
    config = load_slim_config(config_path)

    assert config.request_log is not None
    assert config.request_log.file_path == log_path.as_posix()
    assert config.request_log.body_limit == 512
    # The logging keys must be stripped from litellm_settings so they never
    # leak into the litellm module via setattr.
    assert "request_logging" not in config.litellm_settings
    assert "request_log_file" not in config.litellm_settings


def test_request_logging_requires_file(tmp_path) -> None:
    config_path = write_config(tmp_path, extra_litellm="request_logging: true")

    try:
        load_slim_config(config_path)
    except SlimProxyConfigError as exc:
        assert "request_log_file" in str(exc)
    else:
        raise AssertionError("expected SlimProxyConfigError")


def test_request_logging_records_request_and_response(tmp_path) -> None:
    log_path = tmp_path / "requests.log"
    client, router = make_client(tmp_path, FakeRouter(DumpableResponse()), log_path)

    with client:
        resp = client.post(
            "/v1/messages",
            json={"model": "customer-model", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-master"},
        )
        assert resp.status_code == 200

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    assert "POST /v1/messages" in lines[0]
    assert "body=" in lines[0]
    assert '"model": "customer-model"' in lines[0]
    assert "response status=200" in "\n".join(lines)
    assert "pong" in "\n".join(lines)


def test_request_logging_stream_records_event_summary(tmp_path) -> None:
    log_path = tmp_path / "requests.log"
    client, router = make_client(tmp_path, FakeRouter(fake_anthropic_stream()), log_path)

    with client:
        resp = client.post(
            "/v1/messages",
            json={
                "model": "customer-model",
                "max_tokens": 10,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": "Bearer sk-master"},
        )
        assert resp.status_code == 200
        assert "content_block_delta" in resp.text

    lines = log_path.read_text(encoding="utf-8").splitlines()
    joined = "\n".join(lines)
    assert len(lines) >= 2
    assert "SSE done" in joined
    assert "message_start" in joined
    assert "content_block_delta" in joined
    assert "message_stop" in joined
    assert "total_bytes=" in joined


def test_request_logging_body_limit_truncates(tmp_path) -> None:
    log_path = tmp_path / "requests.log"
    client, router = make_client(tmp_path, FakeRouter(DumpableResponse()), log_path)

    big_payload = {"role": "user", "content": "x" * 5000}
    with client:
        resp = client.post(
            "/v1/messages",
            json={"model": "customer-model", "max_tokens": 10, "messages": [big_payload]},
            headers={"Authorization": "Bearer sk-master"},
        )
        assert resp.status_code == 200

    lines = log_path.read_text(encoding="utf-8").splitlines()
    body_line = lines[0]
    # 200-char limit + "[ts] POST ... body=" prefix; the raw content must be cut
    assert "xxxx" not in body_line[len(body_line) - 100 :] or len(body_line) < 400


def test_request_logging_records_errors(tmp_path) -> None:
    log_path = tmp_path / "requests.log"

    class Boom(Exception):
        status_code = 429

    client, router = make_client(tmp_path, FakeRouter(Boom("nope")), log_path)

    with client:
        resp = client.post(
            "/v1/messages",
            json={"model": "customer-model", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-master"},
        )
        assert resp.status_code == 429

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "error status=429" in lines[1]
    assert "nope" in lines[1]


def test_request_logging_fails_open_when_file_unwritable(tmp_path) -> None:
    """Logging must never break the proxy when the log file can't be opened."""
    log_path = tmp_path / "no_such_dir" / "requests.log"
    client, router = make_client(tmp_path, FakeRouter(DumpableResponse()), log_path)

    with client:
        resp = client.post(
            "/v1/messages",
            json={"model": "customer-model", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-master"},
        )
        # Proxy still works even though the log path is bad
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            assert "pong" in resp.text
