from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class SlimProxyConfigError(ValueError):
    pass


class GeneralSettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    master_key: object | None = None


class ModelListItemModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    litellm_params: dict[str, object]
    model_info: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_litellm_model(self) -> ModelListItemModel:
        if not isinstance(self.litellm_params.get("model"), str):
            raise ValueError("Each model_list item must include litellm_params.model")
        _validate_json_mapping(self.litellm_params)
        if self.model_info is not None:
            _validate_json_mapping(self.model_info)
        return self


class SlimConfigFileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_list: tuple[ModelListItemModel, ...] = Field(min_length=1)
    general_settings: GeneralSettingsModel = Field(default_factory=GeneralSettingsModel)
    router_settings: dict[str, object] = Field(default_factory=dict)
    litellm_settings: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_json_shapes(self) -> SlimConfigFileModel:
        _validate_json_mapping(self.router_settings)
        _validate_json_mapping(self.litellm_settings)
        _validate_json_value(self.general_settings.master_key)
        return self


@dataclass(frozen=True)
class SlimProxyConfig:
    public_model_name: str
    model_list_for_router: tuple[dict[str, JsonValue], ...]
    router_settings_for_router: dict[str, JsonValue]
    litellm_settings: dict[str, JsonValue]
    master_key: str | None


def load_slim_config(config_path: str | Path) -> SlimProxyConfig:
    path = Path(config_path)
    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = SlimConfigFileModel.model_validate(raw_config)
    except (OSError, ValidationError, ValueError) as exc:
        raise SlimProxyConfigError(str(exc)) from exc

    model_names = frozenset(item.model_name for item in config.model_list)
    if len(model_names) != 1:
        raise SlimProxyConfigError(
            "Slim proxy requires all model_list entries to use a single model_name"
        )

    master_key_value = _resolve_json_value(config.general_settings.master_key)
    if master_key_value is None:
        master_key_value = os.getenv("LITELLM_MASTER_KEY")
    if master_key_value is not None and not (
        isinstance(master_key_value, str) and master_key_value
    ):
        raise SlimProxyConfigError(
            "general_settings.master_key must be a non-empty string"
        )

    router_settings = _filter_router_settings(config.router_settings)
    routing_strategy = router_settings.get("routing_strategy")
    if routing_strategy is not None and routing_strategy != "simple-shuffle":
        raise SlimProxyConfigError(
            "Slim proxy only supports routing_strategy=simple-shuffle"
        )

    return SlimProxyConfig(
        public_model_name=next(iter(model_names)),
        model_list_for_router=tuple(
            _model_list_item_to_router_dict(item) for item in config.model_list
        ),
        router_settings_for_router=_resolve_json_mapping(router_settings),
        litellm_settings=_resolve_json_mapping(config.litellm_settings),
        master_key=master_key_value if isinstance(master_key_value, str) else None,
    )


def _filter_router_settings(raw_settings: dict[str, object]) -> dict[str, object]:
    import litellm
    from litellm._logging import verbose_proxy_logger

    valid_args = frozenset(litellm.Router.get_valid_args()) - {
        "model_list",
        "search_tools",
    }
    filtered: dict[str, object] = {}
    unknown: list[str] = []
    for key, value in raw_settings.items():
        if key in valid_args:
            filtered[key] = value
        else:
            unknown.append(key)
    if unknown:
        verbose_proxy_logger.warning(
            "Slim proxy ignoring unknown router_settings: %s",
            ", ".join(sorted(unknown)),
        )
    return filtered


def _model_list_item_to_router_dict(item: ModelListItemModel) -> dict[str, JsonValue]:
    router_item: dict[str, JsonValue] = {
        "model_name": item.model_name,
        "litellm_params": _resolve_json_mapping(item.litellm_params),
    }
    if item.model_info is not None:
        router_item["model_info"] = _resolve_json_mapping(item.model_info)
    return router_item


def _resolve_json_mapping(value: dict[str, object]) -> dict[str, JsonValue]:
    return {key: _resolve_json_value(item) for key, item in value.items()}


def _resolve_json_value(value: object) -> JsonValue:
    if isinstance(value, str):
        return _resolve_string(value)
    if isinstance(value, list):
        return [_resolve_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise SlimProxyConfigError(
                "Slim proxy config mappings must use string keys"
            )
        return {key: _resolve_json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise SlimProxyConfigError(
        f"Slim proxy config values must be JSON-compatible, got {type(value).__name__}"
    )


def _validate_json_mapping(value: dict[str, object]) -> None:
    for item in value.values():
        _validate_json_value(item)


def _validate_json_value(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Slim proxy config mappings must use string keys")
        for item in value.values():
            _validate_json_value(item)
        return
    raise ValueError(
        f"Slim proxy config values must be JSON-compatible, got {type(value).__name__}"
    )


def _resolve_string(value: str) -> str | None:
    prefix = "os.environ/"
    if not value.startswith(prefix):
        return value
    return os.getenv(value.removeprefix(prefix))
