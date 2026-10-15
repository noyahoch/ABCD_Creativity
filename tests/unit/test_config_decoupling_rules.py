"""Exhaustive coverage of the AUT prompt decoupling rule."""

from __future__ import annotations

import pytest

from bridge_steering.config import (
    AUTPromptOverride,
    AUTPrompts,
    ConfigError,
    aut_prompt_pair,
    validate,
)


def _set_aut_prompts(cfg, **kw):
    cfg.aut.prompts = AUTPrompts(**kw)
    return cfg


def test_coupled_default_validates(base_run_cfg):
    validate(base_run_cfg)
    s, i, dec = aut_prompt_pair(base_run_cfg)
    assert s.id == i.id and not dec


def test_coupled_via_explicit_same_subject(base_run_cfg):
    _set_aut_prompts(
        base_run_cfg,
        default_subject="object_only",
        steering=AUTPromptOverride(subject="object_only"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    validate(base_run_cfg)
    _, _, dec = aut_prompt_pair(base_run_cfg)
    assert not dec


def test_decoupled_by_subject_without_allow_raises(base_run_cfg):
    _set_aut_prompts(
        base_run_cfg,
        default_subject="object_only",
        steering=AUTPromptOverride(subject="full_instruction"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    with pytest.raises(ConfigError, match="decoupling detected"):
        validate(base_run_cfg)


def test_decoupled_with_allow_and_reason_validates(base_run_cfg):
    _set_aut_prompts(
        base_run_cfg,
        default_subject="object_only",
        allow_decoupling=True,
        decoupling_reason="ablation: full_a train, object eval",
        steering=AUTPromptOverride(subject="full_instruction"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    validate(base_run_cfg)
    _, _, dec = aut_prompt_pair(base_run_cfg)
    assert dec is True


def test_allow_decoupling_without_reason_raises(base_run_cfg):
    _set_aut_prompts(
        base_run_cfg,
        allow_decoupling=True,
        decoupling_reason="   ",  # whitespace counts as empty
        steering=AUTPromptOverride(subject="full_instruction"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    with pytest.raises(ConfigError, match="decoupling_reason"):
        validate(base_run_cfg)


def test_per_role_override_cannot_set_both_id_and_subject(base_run_cfg):
    _set_aut_prompts(
        base_run_cfg,
        steering=AUTPromptOverride(prompt_id="aut_user_object_only_v1", subject="object_only"),
    )
    with pytest.raises(ConfigError, match="EITHER prompt_id OR subject"):
        validate(base_run_cfg)


# --------------------------------------------------------------------------- #
# Olson runs bypass the decoupling check
#
# When cfg.olson is set, aut.prompts.steering is never consumed (the activation
# collection happens via cfg.olson.{creative,boring}_prompt_id in olson_collect).
# The decoupling check between aut.prompts.steering and aut.prompts.inference is
# therefore meaningless for Olson runs and is skipped.
# --------------------------------------------------------------------------- #


def test_olson_run_skips_decoupling_check(base_run_cfg):
    """An Olson config with `steering` left unset must validate cleanly,
    even though it would otherwise look decoupled."""
    from bridge_steering.config import OlsonConfig

    _set_aut_prompts(
        base_run_cfg,
        default_subject="object_only",
        steering=None,                                       # ← left unset
        inference=AUTPromptOverride(prompt_id="aut_user_a_paper_infer"),
    )
    base_run_cfg.olson = OlsonConfig()
    validate(base_run_cfg)                                   # must not raise


def test_olson_run_skips_decoupling_even_with_obvious_mismatch(base_run_cfg):
    """Even when steering and inference resolve to genuinely different prompts,
    the Olson path skips the check — aut.prompts.steering is never read on
    Olson runs."""
    from bridge_steering.config import OlsonConfig

    _set_aut_prompts(
        base_run_cfg,
        default_subject="object_only",
        steering=AUTPromptOverride(subject="full_instruction"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    base_run_cfg.olson = OlsonConfig()
    validate(base_run_cfg)                                   # must not raise


def test_non_olson_run_still_enforces_decoupling(base_run_cfg):
    """The bypass is gated on cfg.olson — without it, the existing rule fires."""
    _set_aut_prompts(
        base_run_cfg,
        default_subject="object_only",
        steering=AUTPromptOverride(subject="full_instruction"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    base_run_cfg.olson = None                                # ← not Olson
    with pytest.raises(ConfigError, match="decoupling detected"):
        validate(base_run_cfg)
