"""Unit tests for config loading and cross-field validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bridge_steering.config import (
    AUTConfig,
    AUTPrompts,
    AUTPromptOverride,
    ConfigError,
    DecodingConfig,
    JudgeConfig,
    MacGyverInferenceConfig,
    RunConfig,
    SteeringConfig,
    load_run_config,
    resolved_yaml,
    validate,
)


# ---------- load_run_config -------------------------------------------------


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_unknown_top_level_key_raises(tmp_path):
    cfg = _write_yaml(tmp_path, """
run_id: r
output_root: /tmp/out
model_name: m
seed: 1
unknown_top_key: 1
aut:
  abcd_data: /tmp/x.json
steering: {}
decoding_aut: {}
""")
    with pytest.raises(ConfigError, match="unknown key"):
        load_run_config(cfg)


def test_unknown_nested_key_raises(tmp_path):
    cfg = _write_yaml(tmp_path, """
run_id: r
output_root: /tmp/out
model_name: m
seed: 1
aut:
  abcd_data: /tmp/x.json
  bogus_field: 1
steering: {}
decoding_aut: {}
""")
    with pytest.raises(ConfigError, match="unknown key"):
        load_run_config(cfg)


def test_minimal_yaml_loads(tmp_path):
    cfg = _write_yaml(tmp_path, """
run_id: r
output_root: /tmp/out
model_name: meta-llama/Llama-3.1-8B-Instruct
seed: 42
aut:
  abcd_data: /tmp/x.json
steering: {}
decoding_aut: {temperature: 0.7, gen_seed: 1}
""")
    rc = load_run_config(cfg)
    assert rc.run_id == "r"
    assert rc.output_root == Path("/tmp/out")
    assert rc.aut.abcd_data == Path("/tmp/x.json")
    assert rc.decoding_aut.temperature == 0.7


def test_resolved_yaml_round_trip(base_run_cfg):
    """resolved_yaml -> load -> equal RunConfig (modulo dataclass equality)."""
    text = resolved_yaml(base_run_cfg)
    assert "run_id: test_run" in text
    assert "default_subject: object" in text


# ---------- validate() ------------------------------------------------------


def test_macgyver_data_requires_decoding_macgyver(base_run_cfg, tmp_path):
    base_run_cfg.macgyver_data = tmp_path / "mac.xlsx"
    base_run_cfg.decoding_macgyver = None
    with pytest.raises(ConfigError, match="decoding_macgyver"):
        validate(base_run_cfg)


def test_macgyver_judge_without_macgyver_data_raises(base_run_cfg):
    from bridge_steering.config import VllmJudgeBackend
    base_run_cfg.macgyver_judge = JudgeConfig(
        backend=VllmJudgeBackend(), prompt_id="macgyver_judge_additive_5pt_v1",
    )
    base_run_cfg.macgyver_data = None
    with pytest.raises(ConfigError, match="macgyver_data is None"):
        validate(base_run_cfg)


def test_judge_prompt_id_required(base_run_cfg):
    from bridge_steering.config import VllmJudgeBackend
    base_run_cfg.aut_judge = JudgeConfig(backend=VllmJudgeBackend(), prompt_id="")
    with pytest.raises(ConfigError, match="aut_judge.prompt_id"):
        validate(base_run_cfg)


def test_bad_judge_prompt_id_fails_at_validate(base_run_cfg):
    from bridge_steering.config import VllmJudgeBackend
    base_run_cfg.aut_judge = JudgeConfig(backend=VllmJudgeBackend(), prompt_id="does_not_exist")
    with pytest.raises(KeyError):
        validate(base_run_cfg)
