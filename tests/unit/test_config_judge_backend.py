"""Judge backend selection (vLLM vs. API) and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bridge_steering.config import (
    ApiJudgeBackend,
    AzureApiJudgeBackend,
    ConfigError,
    JudgeConfig,
    VllmJudgeBackend,
    load_run_config,
    validate,
)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _base_yaml() -> str:
    return textwrap.dedent("""\
        run_id: r
        output_root: /tmp/out
        model_name: m
        seed: 1
        aut:
          abcd_data: /tmp/x.json
        steering: {}
        decoding_aut: {}
        aut_judge:
          prompt_id: aut_judge_novelty_usability_v5
          backend:
        """)


def test_vllm_backend_loads(tmp_path):
    cfg = _write_yaml(tmp_path, _base_yaml() + "    type: vllm\n    base_url: http://x/v1\n    model: q\n")
    Path("/tmp/x.json").touch()
    rc = load_run_config(cfg)
    assert isinstance(rc.aut_judge.backend, VllmJudgeBackend)
    assert rc.aut_judge.backend.base_url == "http://x/v1"


def test_api_backend_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = _write_yaml(tmp_path, _base_yaml() + "    type: api\n    base_url: https://api.openai.com/v1\n    model: gpt-4o-mini\n    api_key_env: OPENAI_API_KEY\n")
    Path("/tmp/x.json").touch()
    rc = load_run_config(cfg)
    assert isinstance(rc.aut_judge.backend, ApiJudgeBackend)
    assert rc.aut_judge.backend.model == "gpt-4o-mini"


def test_api_backend_missing_env_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("NEVER_SET_ME", raising=False)
    cfg = _write_yaml(tmp_path, _base_yaml() + "    type: api\n    api_key_env: NEVER_SET_ME\n")
    Path("/tmp/x.json").touch()
    with pytest.raises(ConfigError, match="NEVER_SET_ME"):
        load_run_config(cfg)


def test_unknown_backend_type_raises(tmp_path):
    cfg = _write_yaml(tmp_path, _base_yaml() + "    type: anthropic-sdk\n")
    Path("/tmp/x.json").touch()
    with pytest.raises(ConfigError, match="unknown 'type'"):
        load_run_config(cfg)


def test_backend_missing_type_raises(tmp_path):
    cfg = _write_yaml(tmp_path, _base_yaml() + "    base_url: http://x/v1\n    model: q\n")
    Path("/tmp/x.json").touch()
    with pytest.raises(ConfigError, match="requires a 'type'"):
        load_run_config(cfg)


def test_validate_rejects_empty_api_key_env(base_run_cfg, monkeypatch):
    base_run_cfg.aut_judge = JudgeConfig(
        backend=ApiJudgeBackend(api_key_env=""),
        prompt_id="aut_judge_novelty_usability_v5",
    )
    with pytest.raises(ConfigError, match="api_key_env must name an env var"):
        validate(base_run_cfg)


def test_validate_rejects_vllm_with_empty_base_url(base_run_cfg):
    base_run_cfg.aut_judge = JudgeConfig(
        backend=VllmJudgeBackend(base_url=""),
        prompt_id="aut_judge_novelty_usability_v5",
    )
    with pytest.raises(ConfigError, match="base_url must be set"):
        validate(base_run_cfg)


def test_validate_passes_for_set_api_key(base_run_cfg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    base_run_cfg.aut_judge = JudgeConfig(
        backend=ApiJudgeBackend(api_key_env="OPENAI_API_KEY"),
        prompt_id="aut_judge_novelty_usability_v5",
    )
    validate(base_run_cfg)


def _set_azure_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-10-21")


def test_azure_backend_loads(tmp_path, monkeypatch):
    _set_azure_env(monkeypatch)
    cfg = _write_yaml(
        tmp_path,
        _base_yaml() + "    type: azure\n    model: my-deployment\n",
    )
    Path("/tmp/x.json").touch()
    rc = load_run_config(cfg)
    assert isinstance(rc.aut_judge.backend, AzureApiJudgeBackend)
    assert rc.aut_judge.backend.model == "my-deployment"
    assert rc.aut_judge.backend.api_key_env == "AZURE_OPENAI_API_KEY"
    assert rc.aut_judge.backend.endpoint_env == "AZURE_OPENAI_ENDPOINT"
    assert rc.aut_judge.backend.api_version_env == "OPENAI_API_VERSION"


def test_azure_backend_missing_endpoint_raises(tmp_path, monkeypatch):
    _set_azure_env(monkeypatch)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    cfg = _write_yaml(
        tmp_path,
        _base_yaml() + "    type: azure\n    model: my-deployment\n",
    )
    Path("/tmp/x.json").touch()
    with pytest.raises(ConfigError, match="AZURE_OPENAI_ENDPOINT"):
        load_run_config(cfg)


def test_azure_backend_missing_api_version_raises(tmp_path, monkeypatch):
    _set_azure_env(monkeypatch)
    monkeypatch.delenv("OPENAI_API_VERSION", raising=False)
    cfg = _write_yaml(
        tmp_path,
        _base_yaml() + "    type: azure\n    model: my-deployment\n",
    )
    Path("/tmp/x.json").touch()
    with pytest.raises(ConfigError, match="OPENAI_API_VERSION"):
        load_run_config(cfg)


def test_azure_backend_custom_env_var_names(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_AZ_KEY", "k")
    monkeypatch.setenv("MY_AZ_EP", "https://x.openai.azure.com/")
    monkeypatch.setenv("MY_AZ_VER", "2024-10-21")
    cfg = _write_yaml(
        tmp_path,
        _base_yaml()
        + "    type: azure\n"
        + "    model: my-deployment\n"
        + "    api_key_env: MY_AZ_KEY\n"
        + "    endpoint_env: MY_AZ_EP\n"
        + "    api_version_env: MY_AZ_VER\n",
    )
    Path("/tmp/x.json").touch()
    rc = load_run_config(cfg)
    assert isinstance(rc.aut_judge.backend, AzureApiJudgeBackend)
    assert rc.aut_judge.backend.api_key_env == "MY_AZ_KEY"
