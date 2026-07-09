from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from litellm.anthropic_interface.exceptions import AnthropicExceptionMapping
from litellm.proxy.slim_config import SlimProxyConfig, load_slim_config

if TYPE_CHECKING:
    from litellm.router import Router

RouterFactory: TypeAlias = Callable[[SlimProxyConfig], object]

_MESSAGES_PATH_PREFIX = "/v1/messages"


@dataclass(frozen=True)
class SlimProxyState:
    config: SlimProxyConfig
    router: object


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

    @app.get("/v1/models", dependencies=[Depends(_require_master_key)])
    @app.get("/models", dependencies=[Depends(_require_master_key)])
    async def models(request: Request) -> dict[str, object]:
        state = _get_state(request)
        return {
            "object": "list",
            "data": [{"id": state.config.public_model_name, "object": "model"}],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(_require_master_key)])
    @app.post("/chat/completions", dependencies=[Depends(_require_master_key)])
    async def chat_completions(request: Request) -> Response:
        state = _get_state(request)
        request_data = await _read_json_request(request)
        request_data.setdefault("model", state.config.public_model_name)
        try:
            response = await _call_router(state.router, "acompletion", request_data)
        except Exception as exc:
            return _exception_response(exc)
        if request_data.get("stream") is True:
            return StreamingResponse(
                _sse_events(response), media_type="text/event-stream"
            )
        return JSONResponse(content=_serialize_response(response))

    @app.post("/v1/messages", dependencies=[Depends(_require_master_key)])
    async def anthropic_messages(request: Request) -> Response:
        state = _get_state(request)
        request_data = await _read_json_request(request)
        request_data.setdefault("model", state.config.public_model_name)
        try:
            response = await _call_router(
                state.router, "anthropic_messages", request_data
            )
        except Exception as exc:
            return _anthropic_error_response(_exception_status_code(exc), str(exc))
        if request_data.get("stream") is True:
            return StreamingResponse(
                _anthropic_sse_events(response), media_type="text/event-stream"
            )
        return JSONResponse(content=_anthropic_response_body(response))

    @app.post("/v1/messages/count_tokens", dependencies=[Depends(_require_master_key)])
    async def anthropic_count_tokens(request: Request) -> JSONResponse:
        state = _get_state(request)
        request_data = await _read_json_request(request)
        messages = request_data.get("messages")
        if not messages:
            return _anthropic_error_response(
                status.HTTP_400_BAD_REQUEST, "messages parameter is required"
            )
        model = request_data.get("model") or state.config.public_model_name
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
) -> None:
    state = _get_state(request)
    if state.config.master_key is None:
        return
    token = _extract_bearer_token(authorization) or x_litellm_api_key
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


async def _call_router(
    router: object, method_name: str, request_data: dict[str, object]
) -> object:
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


app = create_slim_app()
