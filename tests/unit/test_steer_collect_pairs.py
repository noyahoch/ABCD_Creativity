"""Unit tests for the pure-Python helpers in stages/steer_collect.

Specifically:
* ``format_d_completion`` — D bullets from ``D_banal`` (or configured field).
* ``build_pairs`` — resolves the AUT user prompt via
  ``bridge_steering.prompts.aut.build_user_prompt`` (the SAME function aut_infer
  will call). This is the bug-fix payload — verified per-subject.
"""

from __future__ import annotations

import pytest

from bridge_steering.config import AUTPromptOverride, AUTPrompts
from bridge_steering.stages.steer_collect import build_pairs, format_d_completion


def test_format_d_completion_default():
    item = {"D_banal": ["use1", "use2"]}
    assert format_d_completion(item) == "- use1\n- use2"


def test_format_d_completion_missing_field_raises():
    with pytest.raises(ValueError, match="D_banal"):
        format_d_completion({"id": "x"}, "D_banal")


def test_format_d_completion_string_source_returns_prose_with_warning():
    """D_generic-style sources (single prose string) bypass bullet-join."""
    item = {"id": "AUT_001", "D_generic": "The afternoon light came in."}
    with pytest.warns(UserWarning, match="prose as-is"):
        out = format_d_completion(item, "D_generic")
    assert out == "The afternoon light came in."
    assert not out.startswith("- ")


def test_build_pairs_object_only(base_run_cfg, abcd_tiny_items):
    """Default subject = object_only — prompt should contain only the object."""
    pairs = build_pairs(abcd_tiny_items, base_run_cfg)
    assert len(pairs) == 3
    p0 = pairs[0]
    assert "brick" in p0["user_prompt"]
    assert "kitchen setting" not in p0["user_prompt"]   # full A would include "kitchen"
    assert p0["_aut_prompt_spec"].id == "aut_user_object_only_v1"
    # Row IDs propagated
    assert p0["id"] == "AUT_TEST_001"
    assert p0["source_id"] in (p0["id"], "AUT_TEST_001")
    # D completion built from D_banal
    assert p0["d_completion"].startswith("- Doorstop")
    # B_list copied
    assert isinstance(p0["b_list"], list) and len(p0["b_list"]) == 1


def test_build_pairs_full_instruction_subject(base_run_cfg, abcd_tiny_items):
    """When subject=full_instruction, the entire A field is in the prompt."""
    base_run_cfg.aut.prompts = AUTPrompts(default_subject="full_instruction")
    pairs = build_pairs(abcd_tiny_items, base_run_cfg)
    assert "kitchen setting" in pairs[0]["user_prompt"]
    assert pairs[0]["_aut_prompt_spec"].id == "aut_user_a_bare"


def test_build_pairs_steering_override(base_run_cfg, abcd_tiny_items):
    """A per-role steering override changes the steering prompt only."""
    base_run_cfg.aut.prompts = AUTPrompts(
        default_subject="object_only",
        allow_decoupling=True,
        decoupling_reason="ablation",
        steering=AUTPromptOverride(subject="full_instruction"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    pairs = build_pairs(abcd_tiny_items, base_run_cfg)
    # steering side picks full_instruction
    assert "kitchen setting" in pairs[0]["user_prompt"]
    assert pairs[0]["_aut_prompt_spec"].id == "aut_user_a_bare"


def test_build_pairs_missing_b_list_raises(base_run_cfg):
    with pytest.raises(ValueError, match="B_list"):
        build_pairs(
            [{"id": "x", "A": "Give 8 uses for a brick.", "D_banal": ["d"]}],
            base_run_cfg,
        )


def test_build_pairs_empty_b_list_raises(base_run_cfg):
    with pytest.raises(ValueError, match="B_list"):
        build_pairs(
            [{"id": "x", "A": "x", "D_banal": ["d"], "B_list": []}],
            base_run_cfg,
        )


# --------------------------------------------------------------------------- #
# CAA-style C-vs-D baseline: build_pairs attaches c_list; c_creative_bullets
# source formats the gold creative completions into a bullet list.
# --------------------------------------------------------------------------- #


def test_build_pairs_attaches_c_list_when_present(base_run_cfg, abcd_tiny_items):
    """When item has a non-empty list C field, build_pairs attaches it as c_list."""
    items = [dict(abcd_tiny_items[0], C=["use_a", "use_b", "use_c"])]
    pairs = build_pairs(items, base_run_cfg)
    assert pairs[0]["c_list"] == ["use_a", "use_b", "use_c"]


def test_build_pairs_omits_c_list_when_C_empty_or_string(base_run_cfg, abcd_tiny_items):
    """When item's C is empty/missing/non-list, c_list is NOT attached.

    The legacy fixture stores ``C: ""`` (empty string) for items where the gold
    creative completions were dropped — those should not produce a c_list.
    """
    # Empty string (current fixture state)
    pairs = build_pairs(abcd_tiny_items, base_run_cfg)
    assert "c_list" not in pairs[0]
    # Empty list
    items_empty = [dict(abcd_tiny_items[0], C=[])]
    pairs_empty = build_pairs(items_empty, base_run_cfg)
    assert "c_list" not in pairs_empty[0]


def test_c_creative_bullets_source_formats_bullets():
    """c_creative_bullets returns one bullet-list completion matching D's register."""
    from bridge_steering.steering.completion_sources import SOURCE_REGISTRY
    source = SOURCE_REGISTRY["c_creative_bullets"]({})
    pair = {"c_list": ["use1", "use2", "use3"]}
    completions = source.get_b_completions(pair, model=None, tokenizer=None)
    assert completions == ["- use1\n- use2\n- use3"]


def test_c_creative_bullets_source_raises_on_missing_c_list():
    """Missing or empty c_list is a hard error (caller used the wrong dataset)."""
    from bridge_steering.steering.completion_sources import SOURCE_REGISTRY
    source = SOURCE_REGISTRY["c_creative_bullets"]({})
    with pytest.raises(ValueError, match="c_list"):
        source.get_b_completions({}, model=None, tokenizer=None)
    with pytest.raises(ValueError, match="c_list"):
        source.get_b_completions({"c_list": []}, model=None, tokenizer=None)


def test_c_creative_bullets_source_rejects_empty_strings():
    """Empty / whitespace-only C entries are a hard error (broken data)."""
    from bridge_steering.steering.completion_sources import SOURCE_REGISTRY
    source = SOURCE_REGISTRY["c_creative_bullets"]({})
    with pytest.raises(ValueError, match="non-empty string"):
        source.get_b_completions(
            {"c_list": ["a", ""]}, model=None, tokenizer=None,
        )
