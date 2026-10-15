"""Regression tests for the AUT prompt decoupling rule.

Covers the bug originally called out: activation collection used one
prompt while AUT inference used another. The new design makes coupling
the default and decoupling explicit (allow_decoupling: true + reason).
"""

from __future__ import annotations

import pytest

from bridge_steering.config import (
    AUTPromptOverride,
    AUTPrompts,
    ConfigError,
    aut_prompt_pair,
    validate,
)


# --- forbidden by default ---------------------------------------------------


def test_decoupling_forbidden_by_default_via_subject(base_run_cfg):
    base_run_cfg.aut.prompts = AUTPrompts(
        steering=AUTPromptOverride(subject="full_instruction"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    with pytest.raises(ConfigError, match="decoupling"):
        validate(base_run_cfg)


def test_decoupling_detected_by_id_not_subject(base_run_cfg):
    """Different prompt ids: still a decoupling.

    Future-proof: when an ``aut_user_object_only_v2`` exists, mismatching
    versions across steering/inference must be caught even though both have
    ``subject="object_only"``. Today we only have one spec per subject, so
    we use cross-subject ids to verify the id+hash comparison fires.
    """
    base_run_cfg.aut.prompts = AUTPrompts(
        steering=AUTPromptOverride(prompt_id="aut_user_object_only_v1"),
        inference=AUTPromptOverride(prompt_id="aut_user_a_bare"),
    )
    with pytest.raises(ConfigError, match="decoupling"):
        validate(base_run_cfg)


# --- allowed when explicit --------------------------------------------------


def test_decoupling_allowed_when_explicit(base_run_cfg):
    base_run_cfg.aut.prompts = AUTPrompts(
        allow_decoupling=True,
        decoupling_reason="ablation: train on full_instruction, eval on object_only",
        steering=AUTPromptOverride(subject="full_instruction"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    validate(base_run_cfg)
    s, i, dec = aut_prompt_pair(base_run_cfg)
    assert dec is True
    assert s.id != i.id


def test_allow_decoupling_requires_reason(base_run_cfg):
    base_run_cfg.aut.prompts = AUTPrompts(
        allow_decoupling=True,
        decoupling_reason="",
        steering=AUTPromptOverride(subject="full_instruction"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    with pytest.raises(ConfigError, match="decoupling_reason"):
        validate(base_run_cfg)
