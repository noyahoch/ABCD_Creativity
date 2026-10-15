"""Unit tests for MacGyver prompt builders."""

from __future__ import annotations

import pytest

from bridge_steering.prompts.macgyver import (
    MACGYVER_USER_DEFAULT_V1,
    MACGYVER_USER_NOVELTY_INSTRUCT_V1,
    MACGYVER_JUDGE_ADDITIVE_5PT_V1,
    build_judge_prompt,
    build_user_prompt,
)


def test_user_default_style():
    msgs, spec, user = build_user_prompt({"problem": "Cross a river."}, "default")
    assert spec is MACGYVER_USER_DEFAULT_V1
    assert msgs[0]["role"] == "system"  # default has a system message
    assert "Cross a river." in user
    assert "Step 1" in user


def test_user_novelty_instruct_style():
    msgs, spec, user = build_user_prompt(
        {"problem": "Cross a river."}, "novelty_instruct"
    )
    assert spec is MACGYVER_USER_NOVELTY_INSTRUCT_V1
    assert msgs[0]["role"] == "user"  # paper E.2.3: user-only
    assert "Cross a river." in user


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        build_user_prompt({"problem": "x"}, "weird")  # type: ignore[arg-type]


def test_judge_prompt_truncates_long_response():
    long_resp = "A" * 5000
    msgs, spec, user = build_judge_prompt(
        MACGYVER_JUDGE_ADDITIVE_5PT_V1.id, "problem", long_resp
    )
    # Default cap = 2000 chars; should NOT contain the tail of the input
    assert "A" * 2000 in user
    assert "A" * 2001 not in user


def test_judge_prompt_renders_required_substrings():
    _, _, user = build_judge_prompt(
        MACGYVER_JUDGE_ADDITIVE_5PT_V1.id, "problem statement", "solution text"
    )
    assert "Score:" in user
    assert "problem statement" in user
    assert "solution text" in user


def test_judge_prompt_unknown_id_raises():
    with pytest.raises(KeyError):
        build_judge_prompt("nope_v99", "p", "s")
