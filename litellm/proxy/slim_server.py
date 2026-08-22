from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from litellm.anthropic_interface.exceptions import AnthropicExceptionMapping
from litellm.proxy.slim_config import (
    RequestLogConfig,
    SlimProxyConfig,
    load_slim_config,
)

if TYPE_CHECKING:
    from litellm.router import Router

RouterFactory: TypeAlias = Callable[[SlimProxyConfig], object]

_MESSAGES_PATH_PREFIX = "/v1/messages"


@dataclass(frozen=True)
class SlimProxyState:
    config: SlimProxyConfig
    router: object


def _default_model_for_request(state: SlimProxyState) -> str | None:
    """Single-model config: default to the one public model. Multi-model:
    client MUST specify model explicitly."""
    if state.config.public_model_name is not None:
        return state.config.public_model_name
    return None


def _ensure_model_in_request(request_data: dict, state: SlimProxyState) -> None:
    """Fill in the default model when the config has exactly one public
    model; otherwise require the client to specify one."""
    if "model" not in request_data or not request_data.get("model"):
        default = _default_model_for_request(state)
        if default is not None:
            request_data["model"] = default
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": (
                        "This proxy exposes multiple models; "
                        "the 'model' field is required"
                    ),
                    "type": "invalid_request_error",
                },
            )


def create_slim_app(
    config_path: str | Path | None = None,
    router_factory: RouterFactory | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = load_slim_config(_resolve_config_path(config_path))
        _apply_config_to_litellm(config)
        app.state.slim_proxy_state = SlimProxyState(
            config=config,
            router=(router_factory or _default_router_factory)(config),
        )
        yield

    app = FastAPI(title="LiteLLM Slim Proxy", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return _http_exception_response(request, exc)

    @app.get("/health")
    @app.get("/health/liveliness")
    @app.get("/health/readiness")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    def _model_entry(model_id: str, known_ids: set[str]) -> dict[str, object] | None:
        """Build a /v1/models entry, attaching reasoning (thinking-level)
        capability metadata so clients like zcode can offer an "off" variant
        for deepseek-family models instead of forcing thinking on.
        Returns None when the model id is not part of this gateway's config."""
        if model_id not in known_ids:
            return None
        entry: dict[str, object] = {"id": model_id, "object": "model"}
        base = model_id.lower()
        if "deepseek" in base:
            # deepseek-family reasoning models: allow off/high/max
            entry["reasoning"] = {
                "enabled": True,
                "variants": ["off", "high", "max"],
                "defaultVariant": "max",
            }
        elif "mimo" in base:
            entry["reasoning"] = {
                "enabled": True,
                "variants": ["enabled", "off"],
                "defaultVariant": "enabled",
            }
        elif "glm" in base:
            entry["reasoning"] = {
                "enabled": True,
                "variants": ["low", "max", "high"],
                "defaultVariant": "max",
            }
        # hy3 / kimi / minimax / qwen / others: no reasoning metadata
        return entry

    @app.get("/v1/models", dependencies=[Depends(_require_master_key)])
    @app.get("/models", dependencies=[Depends(_require_master_key)])
    async def models(request: Request) -> dict[str, object]:
        state = _get_state(request)
        model_ids = (
            state.config.model_names
            if state.config.model_names
            else {state.config.public_model_name}
        )
        return {
            "object": "list",
            "data": [
                e
                for m in sorted(model_ids)
                if (e := _model_entry(m, model_ids)) is not None
            ],
        }

    @app.get("/v1/models/{model_id}", dependencies=[Depends(_require_master_key)])
    @app.get("/models/{model_id}", dependencies=[Depends(_require_master_key)])
    async def model_detail(model_id: str, request: Request) -> dict[str, object]:
        state = _get_state(request)
        known = (
            state.config.model_names
            if state.config.model_names
            else {state.config.public_model_name}
        )
        entry = _model_entry(model_id, known)
        if entry is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return entry

    @app.post("/v1/chat/completions", dependencies=[Depends(_require_master_key)])
    @app.post("/chat/completions", dependencies=[Depends(_require_master_key)])
    async def chat_completions(request: Request) -> Response:
        state = _get_state(request)
        request_data = await _read_json_request(request)
        _ensure_model_in_request(request_data, state)
        log = state.config.request_log
        if log is not None:
            _log_request(log, request, request_data)
        try:
            response = await _call_router(state.router, "acompletion", request_data)
        except Exception as exc:
            if log is not None:
                _append_log_line(
                    log, f"error status={_exception_status_code(exc)} message={exc}"
                )
            return _exception_response(exc)
        if request_data.get("stream") is True:
            return StreamingResponse(
                _logged_sse(log, _sse_events(response))
                if log is not None
                else _sse_events(response),
                media_type="text/event-stream",
            )
        body = _serialize_response(response)
        if log is not None:
            _log_response(log, status.HTTP_200_OK, body)
        return JSONResponse(content=body)

    @app.post("/v1/messages", dependencies=[Depends(_require_master_key)])
    async def anthropic_messages(request: Request) -> Response:
        state = _get_state(request)
        request_data = await _read_json_request(request)
        _ensure_model_in_request(request_data, state)
        log = state.config.request_log
        if log is not None:
            _log_request(log, request, request_data)
        try:
            response = await _call_router(
                state.router, "anthropic_messages", request_data
            )
        except Exception as exc:
            if log is not None:
                _append_log_line(
                    log, f"error status={_exception_status_code(exc)} message={exc}"
                )
            return _anthropic_error_response(_exception_status_code(exc), str(exc))
        if request_data.get("stream") is True:
            return StreamingResponse(
                _logged_sse(log, _anthropic_sse_events(response))
                if log is not None
                else _anthropic_sse_events(response),
                media_type="text/event-stream",
            )
        body = _anthropic_response_body(response)
        if log is not None:
            _log_response(log, status.HTTP_200_OK, body)
        return JSONResponse(content=body)

    @app.post("/v1/messages/count_tokens", dependencies=[Depends(_require_master_key)])
    async def anthropic_count_tokens(request: Request) -> JSONResponse:
        state = _get_state(request)
        request_data = await _read_json_request(request)
        messages = request_data.get("messages")
        if not messages:
            return _anthropic_error_response(
                status.HTTP_400_BAD_REQUEST, "messages parameter is required"
            )
        model = request_data.get("model") or _default_model_for_request(state)
        try:
            import litellm

            tokens = litellm.token_counter(
                model=model,
                messages=messages,
                tools=request_data.get("tools"),
            )
        except Exception as exc:
            return _anthropic_error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)
            )
        return JSONResponse(content={"input_tokens": tokens})

    return app


def _apply_config_to_litellm(config: SlimProxyConfig) -> None:
    import litellm
    from litellm._logging import verbose_proxy_logger

    skipped = _apply_litellm_settings(config.litellm_settings, litellm)
    if skipped:
        verbose_proxy_logger.info(
            "Slim proxy skipped non-scalar litellm_settings: %s",
            ", ".join(skipped),
        )
    if config.master_key is None:
        verbose_proxy_logger.warning(
            "Slim proxy has no master_key; all requests are unauthenticated"
        )


def _apply_litellm_settings(
    settings: dict[str, object], target: object
) -> tuple[str, ...]:
    skipped: list[str] = []
    for key, value in settings.items():
        if key == "json_logs" and value is True:
            setattr(target, "json_logs", True)
            turn_on = getattr(target, "_turn_on_json", None)
            if callable(turn_on):
                turn_on()
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            setattr(target, key, value)
        else:
            skipped.append(key)
    return tuple(skipped)


def _resolve_config_path(config_path: str | Path | None) -> Path:
    resolved = config_path or os.getenv("CONFIG_FILE_PATH")
    if resolved is None:
        raise RuntimeError("Slim proxy requires --config or CONFIG_FILE_PATH")
    return Path(resolved)


def _default_router_factory(config: SlimProxyConfig) -> "Router":
    import litellm
    from litellm.types.router import RouterGeneralSettings

    router_settings = {
        "routing_strategy": "simple-shuffle",
        **config.router_settings_for_router,
    }
    return litellm.Router(
        model_list=list(config.model_list_for_router),
        router_general_settings=RouterGeneralSettings(async_only_mode=True),
        ignore_invalid_deployments=True,
        **router_settings,
    )


async def _require_master_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_litellm_api_key: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    state = _get_state(request)
    if state.config.master_key is None:
        return
    token = _extract_bearer_token(authorization) or x_litellm_api_key or x_api_key
    if token != state.config.master_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "Invalid or missing API key",
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key",
            },
        )


def _get_state(request: Request) -> SlimProxyState:
    return request.app.state.slim_proxy_state


def _extract_bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    scheme, separator, token = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and token:
        return token
    return None


async def _read_json_request(request: Request) -> dict[str, object]:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Request body must be a JSON object",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_request",
            },
        )
    return dict(body)


# Models that reject tool definitions entirely (verified 2026-08: opencode
# Console Go returns HTTP 400 [1210]/[1214] on any request carrying a
# non-empty tools array; empty/no tools works fine). Claude Code and Hermes
# always attach tools, so strip them before forwarding.
_NO_TOOLS_MODELS = ("ox-alpha-free",)


def _strip_tools_for_no_tools_models(request_data: dict[str, object]) -> None:
    model = request_data.get("model")
    if not model:
        return
    model_str = str(model)
    if not any(m in model_str for m in _NO_TOOLS_MODELS):
        return
    removed = [k for k in ("tools", "tool_choice") if k in request_data]
    if removed:
        for k in removed:
            request_data.pop(k, None)
        print(
            f"[strip-tools] {model_str}: stripped {removed} before upstream call",
            flush=True,
        )


async def _call_router(
    router: object, method_name: str, request_data: dict[str, object]
) -> object:
    _strip_tools_for_no_tools_models(request_data)
    method = getattr(router, method_name)
    return await method(**request_data)


def _http_exception_response(request: Request, exc: HTTPException) -> JSONResponse:
    if request.url.path.startswith(_MESSAGES_PATH_PREFIX):
        return _anthropic_error_response(exc.status_code, _http_exception_message(exc))
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": _openai_error_from_detail(exc.detail)},
    )


def _http_exception_message(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict):
        return str(detail.get("message") or "HTTP error")
    return str(detail)


def _openai_error_from_detail(detail: Any) -> dict[str, object]:
    if isinstance(detail, dict):
        return {
            "message": str(detail.get("message", "HTTP error")),
            "type": str(detail.get("type", "invalid_request_error")),
            "param": detail.get("param"),
            "code": detail.get("code"),
        }
    return {
        "message": str(detail),
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }


def _anthropic_error_response(status_code: int, message: str) -> JSONResponse:
    body = AnthropicExceptionMapping.transform_to_anthropic_error(
        status_code=status_code, raw_message=message
    )
    return JSONResponse(status_code=status_code, content=body)


def _exception_response(exc: Exception) -> JSONResponse:
    status_code = _exception_status_code(exc)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": str(exc),
                "type": "api_error" if status_code >= 500 else "invalid_request_error",
                "param": None,
                "code": getattr(exc, "code", None),
            }
        },
    )


def _exception_status_code(exc: Exception) -> int:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status_code, int) and 100 <= status_code <= 599:
        return status_code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


async def _sse_events(response: object) -> AsyncIterator[str]:
    async for chunk in _iterate_response(response):
        yield f"data: {_json_dumps(_serialize_response(chunk))}\n\n"
    yield "data: [DONE]\n\n"


async def _anthropic_sse_events(response: object) -> AsyncIterator[str]:
    async for chunk in _iterate_response(response):
        yield _anthropic_sse_chunk(chunk)


def _anthropic_sse_chunk(chunk: object) -> str:
    if isinstance(chunk, (bytes, bytearray)):
        return chunk.decode("utf-8", errors="ignore")
    if isinstance(chunk, dict):
        event_type = str(chunk.get("type", "message"))
        return f"event: {event_type}\ndata: {_json_dumps(chunk)}\n\n"
    return str(chunk)


async def _iterate_response(response: object) -> AsyncIterator[object]:
    if hasattr(response, "__aiter__"):
        async for chunk in response:  # type: ignore[attr-defined]
            yield chunk
        return
    if isinstance(response, Iterable) and not isinstance(response, (str, bytes, dict)):
        for chunk in response:
            yield chunk
        return
    yield response


def _serialize_response(response: object) -> object:
    if isinstance(response, BaseModel):
        return response.model_dump(exclude_none=True)
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return response


def _anthropic_response_body(response: object) -> object:
    import litellm

    body = _serialize_response(response)
    if litellm.strip_anthropic_total_tokens and isinstance(body, dict):
        usage = body.get("usage")
        if isinstance(usage, dict):
            usage.pop("total_tokens", None)
    return body


def _json_dumps(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Request/response logging (diagnostics)
#
# Mirrors the external log_proxy.py behavior: full request body (truncated),
# status + response body summary for non-streaming, and event-count + tail
# summary for streaming SSE responses. Enabled via litellm_settings:
#
#   litellm_settings:
#     request_logging: true
#     request_log_file: C:/path/to/requests.log
#     request_log_body_limit: 3000
# ---------------------------------------------------------------------------

_LOG_LOCK = threading.Lock()


def _append_log_line(log: RequestLogConfig, line: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _LOG_LOCK:
        try:
            with open(log.file_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {line}\n")
        except OSError:
            # Logging must never break the proxy; drop the line silently.
            pass


def _log_request(
    log: RequestLogConfig, request: Request, request_data: dict[str, object]
) -> None:
    try:
        body = json.dumps(request_data, ensure_ascii=False)[: log.body_limit]
    except (TypeError, ValueError):
        body = "<unserializable request body>"
    client = request.client.host if request.client is not None else "?"
    _append_log_line(
        log,
        f"{request.method} {request.url.path} client={client} body={body}",
    )


def _log_response(log: RequestLogConfig, status_code: int, body: object) -> None:
    try:
        summary = json.dumps(body, ensure_ascii=False)[: log.body_limit]
    except (TypeError, ValueError):
        summary = "<unserializable response body>"
    _append_log_line(log, f"response status={status_code} body={summary}")


async def _logged_sse(
    log: RequestLogConfig, generator: AsyncIterator[str]
) -> AsyncIterator[str]:
    """Wrap an SSE generator, recording event counts + tail when it ends."""
    counter: Counter[str] = Counter()
    tail: deque[str] = deque(maxlen=40)
    total_bytes = 0
    async for chunk in generator:
        total_bytes += len(chunk)
        for line in chunk.splitlines():
            if line.startswith("event:"):
                counter[line[6:].strip()] += 1
            elif line.startswith("data:"):
                counter["data"] += 1
        tail.append(chunk)
        yield chunk
    tail_text = "".join(tail)[-600:]
    _append_log_line(
        log,
        f"SSE done events={dict(counter)} total_bytes={total_bytes} "
        f"tail={tail_text}",
    )


app = create_slim_app()
