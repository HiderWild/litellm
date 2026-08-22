
"""Verify slim /v1/models returns reasoning capability metadata and the
new per-model detail endpoint, without breaking the existing list shape."""
from __future__ import annotations

import json
from textwrap import dedent

from fastapi.testclient import TestClient

from litellm.proxy.slim_server import create_slim_app

# Minimal config with a multi-model setup (mirrors the real gateway config
# shape: deepseek-go, hy3, glm, mimo).
CONFIG_YAML = dedent("""
    model_list:
      - model_name: deepseek-v4-flash-go
        litellm_params:
          model: anthropic/deepseek-v4-flash
          api_base: https://opencode.ai/zen/go
          api_key: sk-test-go
        model_info: {id: deepseek-v4-flash-go-SoZl}
      - model_name: deepseek-v4-pro-go
        litellm_params:
          model: anthropic/deepseek-v4-pro
          api_base: https://opencode.ai/zen/go
          api_key: sk-test-go
        model_info: {id: deepseek-v4-pro-go-SoZl}
      - model_name: hy3
        litellm_params:
          model: anthropic/hy3
          api_base: https://opencode.ai/zen/go
          api_key: sk-test-go
        model_info: {id: hy3-SoZl}
      - model_name: mimo-v2.5
        litellm_params:
          model: anthropic/mimo-v2.5
          api_base: https://opencode.ai/zen/go
          api_key: sk-test-go
        model_info: {id: mimo-v2.5-SoZl}
      - model_name: glm-5.3
        litellm_params:
          model: anthropic/glm-5.3
          api_base: https://ark.cn-beijing.volces.com/api/plan
          api_key: ark-test
        model_info: {id: glm-5.3-SoZl}
    general_settings:
      master_key: sk-test-master
    router_settings:
      routing_strategy: simple-shuffle
""")


def _client():
    app = create_slim_app(config_path=None)
    # Override the config the app would load: easier to point at a real file.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(CONFIG_YAML)
        cfg = f.name
    app = create_slim_app(config_path=cfg)
    return TestClient(app)


def test_models_list_reasoning_metadata():
    with _client() as c:
        r = c.get("/v1/models", headers={"Authorization": "Bearer sk-test-master"})
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        by_id = {m["id"]: m for m in data}
        # deepseek entries advertise off/high/max
        ds = by_id["deepseek-v4-flash-go"]
        assert ds["reasoning"] == {
            "enabled": True,
            "variants": ["off", "high", "max"],
            "defaultVariant": "max",
        }
        assert by_id["deepseek-v4-pro-go"]["reasoning"]["variants"] == ["off", "high", "max"]
        # hy3: no reasoning advertised
        assert "reasoning" not in by_id["hy3"]
        # mimo: enabled/off
        assert by_id["mimo-v2.5"]["reasoning"]["variants"] == ["enabled", "off"]
        # glm: low/max/high
        assert by_id["glm-5.3"]["reasoning"]["variants"] == ["low", "max", "high"]


def test_model_detail_endpoint():
    with _client() as c:
        r = c.get("/v1/models/deepseek-v4-flash-go",
                  headers={"Authorization": "Bearer sk-test-master"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["id"] == "deepseek-v4-flash-go"
        assert body["reasoning"]["variants"] == ["off", "high", "max"]

        r2 = c.get("/v1/models/hy3",
                   headers={"Authorization": "Bearer sk-test-master"})
        assert r2.status_code == 200
        assert "reasoning" not in r2.json()

def test_unknown_model_detail_returns_404():
    with _client() as c:
        r = c.get("/v1/models/does-not-exist",
                  headers={"Authorization": "Bearer sk-test-master"})
        assert r.status_code == 404, r.text

def test_models_requires_auth():
    with _client() as c:
        r = c.get("/v1/models")
        assert r.status_code == 401, r.text
