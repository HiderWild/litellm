from __future__ import annotations

from textwrap import dedent

import pytest

from litellm.proxy.slim_config import SlimProxyConfigError, load_slim_config


def write_config(tmp_path, content: str):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(dedent(content), encoding="utf-8")
    return config_path


def test_load_slim_config_accepts_one_public_model_with_multiple_deployments(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLIM_MASTER_KEY", "sk-master")
    monkeypatch.setenv("UPSTREAM_KEY_ONE", "sk-one")
    monkeypatch.setenv("UPSTREAM_KEY_TWO", "sk-two")
    config_path = write_config(
        tmp_path,
        """
        general_settings:
          master_key: os.environ/SLIM_MASTER_KEY
        model_list:
          - model_name: customer-model
            litellm_params:
              model: openai/gpt-4.1-mini
              api_key: os.environ/UPSTREAM_KEY_ONE
              api_base: https://one.example.test/v1
          - model_name: customer-model
            litellm_params:
              model: openai/gpt-4.1-mini
              api_key: os.environ/UPSTREAM_KEY_TWO
              api_base: https://two.example.test/v1
        router_settings:
          routing_strategy: simple-shuffle
          num_retries: 1
          cooldown_time: 30
        """,
    )

    config = load_slim_config(config_path)

    assert config.public_model_name == "customer-model"
    assert config.master_key == "sk-master"
    assert len(config.model_list_for_router) == 2
    assert config.model_list_for_router[0]["litellm_params"]["api_key"] == "sk-one"
    assert config.model_list_for_router[1]["litellm_params"]["api_key"] == "sk-two"
    assert config.router_settings_for_router == {
        "routing_strategy": "simple-shuffle",
        "num_retries": 1,
        "cooldown_time": 30,
    }


def test_load_slim_config_uses_environment_master_key_when_config_omits_it(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-env-master")
    config_path = write_config(
        tmp_path,
        """
        model_list:
          - model_name: customer-model
            litellm_params:
              model: openai/gpt-4.1-mini
        """,
    )

    config = load_slim_config(config_path)

    assert config.master_key == "sk-env-master"


def test_load_slim_config_master_key_optional_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    config_path = write_config(
        tmp_path,
        """
        model_list:
          - model_name: customer-model
            litellm_params:
              model: openai/gpt-4.1-mini
        """,
    )

    config = load_slim_config(config_path)

    assert config.master_key is None


def test_load_slim_config_rejects_multiple_public_model_names(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")
    config_path = write_config(
        tmp_path,
        """
        model_list:
          - model_name: customer-model
            litellm_params:
              model: openai/gpt-4.1-mini
          - model_name: other-model
            litellm_params:
              model: openai/gpt-4.1-mini
        """,
    )

    with pytest.raises(SlimProxyConfigError, match="single model_name"):
        load_slim_config(config_path)


def test_load_slim_config_rejects_unsupported_routing_strategy(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")
    config_path = write_config(
        tmp_path,
        """
        model_list:
          - model_name: customer-model
            litellm_params:
              model: openai/gpt-4.1-mini
        router_settings:
          routing_strategy: latency-based-routing
        """,
    )

    with pytest.raises(SlimProxyConfigError, match="simple-shuffle"):
        load_slim_config(config_path)


def test_load_slim_config_filters_unknown_router_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")
    config_path = write_config(
        tmp_path,
        """
        model_list:
          - model_name: customer-model
            litellm_params:
              model: openai/gpt-4.1-mini
        router_settings:
          routing_strategy: simple-shuffle
          latency_window_size: 10
        """,
    )

    config = load_slim_config(config_path)

    assert "latency_window_size" not in config.router_settings_for_router
    assert config.router_settings_for_router["routing_strategy"] == "simple-shuffle"


def test_load_slim_config_accepts_real_world_router_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")
    config_path = write_config(
        tmp_path,
        """
        model_list:
          - model_name: ark-code
            litellm_params:
              model: anthropic/glm-5.2
              api_key: sk-one
          - model_name: ark-code
            litellm_params:
              model: anthropic/glm-5.2
              api_key: sk-two
        router_settings:
          routing_strategy: simple-shuffle
          enable_pre_call_checks: false
          model_group_alias:
            glm-5.2: ark-code
          retry_policy:
            RateLimitError:
              num_retries: 3
        """,
    )

    config = load_slim_config(config_path)

    assert "enable_pre_call_checks" in config.router_settings_for_router
    assert "model_group_alias" in config.router_settings_for_router
    assert "retry_policy" in config.router_settings_for_router


def test_load_slim_config_accepts_litellm_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")
    config_path = write_config(
        tmp_path,
        """
        model_list:
          - model_name: customer-model
            litellm_params:
              model: openai/gpt-4.1-mini
        litellm_settings:
          drop_params: true
          request_timeout: 120
          telemetry: false
        """,
    )

    config = load_slim_config(config_path)

    assert config.litellm_settings == {
        "drop_params": True,
        "request_timeout": 120,
        "telemetry": False,
    }
