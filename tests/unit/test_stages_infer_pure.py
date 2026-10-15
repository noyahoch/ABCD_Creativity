"""Pure-Python tests for the inference stages' prompt-building and parsing.

The model-backed half of each stage is covered by the e2e tests; here we
verify the pieces that don't need a GPU.
"""

from __future__ import annotations

import pytest

from bridge_steering.stages.aut_infer import (
    AUT_INFER_COLUMNS, _build_user_messages as aut_msgs, parse_uses_from_reply,
)
from bridge_steering.stages.macgyver_infer import (
    MAC_INFER_COLUMNS, _build_user_messages as mac_msgs,
)


# --- AUT --------------------------------------------------------------------


def test_aut_messages_default_subject(base_run_cfg, abcd_tiny_items):
    msgs, spec, user = aut_msgs(abcd_tiny_items[0], base_run_cfg, "inference")
    assert spec.id == "aut_user_object_only_v1"
    assert "brick" in user
    assert msgs[-1]["content"] == user


def test_parse_uses_drops_preamble_and_short_lines():
    reply = (
        "Here are 8 unconventional uses for a brick:\n"
        "- store thermal mass for a slow oven\n"
        "- use as a doorstop\n"      # too short ("- use as a doorstop" → 4 words OK)
        "* mark a tennis-ball wall target\n"
        "- ok\n"                       # too short
        "1. press flowers between two bricks\n"
    )
    uses = parse_uses_from_reply(reply)
    assert "store thermal mass for a slow oven" in uses
    assert "press flowers between two bricks" in uses
    assert "ok" not in uses
    assert not any(u.lower().startswith("here are") for u in uses)


def test_parse_uses_handles_empty():
    assert parse_uses_from_reply("") == []
    assert parse_uses_from_reply(None) == []  # type: ignore[arg-type]


def test_aut_csv_columns_have_all_audit_fields():
    """Per the v3 schema, the audit columns must all be present."""
    required = {
        "item_id", "sample_idx", "seed",
        "prompt_id", "prompt_sha256", "prompt_input_hash",
        "raw_response", "parsed_response", "output_hash",
    }
    assert required.issubset(set(AUT_INFER_COLUMNS))


# --- MacGyver ---------------------------------------------------------------


def test_macgyver_messages_default_style(base_run_cfg):
    item = {"problem": "Cross a river with only X and Y."}
    msgs, spec, user = mac_msgs(item, base_run_cfg)
    assert spec.id == "macgyver_user_default_v1"
    assert "Cross a river" in user
    assert msgs[0]["role"] == "system"


def test_macgyver_messages_novelty_instruct_style(base_run_cfg):
    from bridge_steering.config import MacGyverInferenceConfig
    base_run_cfg.macgyver = MacGyverInferenceConfig(prompt_style="novelty_instruct")
    item = {"problem": "Cross a river with only X and Y."}
    msgs, spec, user = mac_msgs(item, base_run_cfg)
    assert spec.id == "macgyver_user_novelty_instruct_v1"
    # Paper E.2.3 prompt is user-only (no system message).
    assert msgs[0]["role"] == "user"


def test_macgyver_csv_columns_have_required_extras():
    required = {
        "item_id", "problem_id", "problem_text", "solvable", "unconventional",
        "ref_solution", "ref_label", "prompt_id", "prompt_sha256", "output_hash",
    }
    assert required.issubset(set(MAC_INFER_COLUMNS))
