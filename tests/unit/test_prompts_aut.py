"""Unit tests for AUT prompt builders."""

from __future__ import annotations

import pytest

from bridge_steering.prompts.aut import (
    AUT_USER_OBJECT_ONLY_V1,
    AUT_USER_A_BARE,
    AUT_BRAINSTORM_V1,
    AUT_BS_ADVANTAGES_V1,
    AUT_BS_DRAWBACKS_V1,
    AUT_BS_UTIL_EVAL_V1,
    AUT_BS_ORIG_EVAL_V1,
    BRAINSTORM_SELECT_REGISTRY,
    JUDGE_NOVELTY_USABILITY_V5,
    aut_object,
    build_judge_prompt,
    build_user_prompt,
    spec_for_subject,
)


def test_aut_object_prefers_A_OBJECT_field():
    obj, src = aut_object({"A_OBJECT": "brick", "A": "Give 8 uses for a paperclip."})
    assert obj == "brick" and src == "A_OBJECT"


def test_aut_object_falls_back_to_parsing_A():
    obj, src = aut_object({"A": "Give 8 uses for a paperclip in an office."})
    assert "paperclip" in obj
    assert src == "parsed_from_A"


def test_spec_for_subject():
    assert spec_for_subject("object_only") is AUT_USER_OBJECT_ONLY_V1
    assert spec_for_subject("full_instruction") is AUT_USER_A_BARE
    with pytest.raises(ValueError):
        spec_for_subject("wrong")  # type: ignore[arg-type]


def test_build_user_prompt_object_only():
    msgs, spec, user = build_user_prompt({"A_OBJECT": "brick", "A": "x"}, "object_only")
    assert spec is AUT_USER_OBJECT_ONLY_V1
    assert msgs[-1]["role"] == "user"
    assert "brick" in user
    assert 'starting with "-"' in user


def test_build_user_prompt_full_instruction():
    msgs, spec, user = build_user_prompt({"A": "Uses for a brick."}, "full_instruction")
    assert spec is AUT_USER_A_BARE
    assert "Uses for a brick." in user


def test_judge_prompt_renders_without_keyerror():
    msgs, spec, user = build_judge_prompt(
        JUDGE_NOVELTY_USABILITY_V5.id, "brick", "use as a paperweight"
    )
    assert spec is JUDGE_NOVELTY_USABILITY_V5
    # Parser dependencies — these substrings MUST appear or the parser breaks.
    assert "ONE valid JSON object" in user
    assert "novelty" in user.lower()
    assert "usability" in user.lower()
    assert "brick" in user
    assert "use as a paperweight" in user
    # v4/v5 reordering invariants: data block comes AFTER the rubric, the
    # schema/output-spec comes AFTER the data, and there's no dangling
    # "object+setting" wording (only "this object").
    rubric_pos = user.find("NOVELTY (1-10)")
    data_pos = user.find("Object: brick")
    schema_pos = user.find('"novelty": integer')
    assert rubric_pos < data_pos < schema_pos
    assert "object+setting" not in user
    assert "this object." in user


def test_judge_prompt_unknown_id_raises():
    with pytest.raises(KeyError):
        build_judge_prompt("nope_v99", "brick", "use")


def test_judge_prompt_archived_blocked_by_default():
    # v3 is now archived too — same opt-in behavior.
    with pytest.raises(ValueError, match="archived"):
        build_judge_prompt("aut_judge_novelty_usability_v3", "brick", "use")
    # opt-in unlocks it
    _, spec, _ = build_judge_prompt(
        "aut_judge_novelty_usability_v3", "brick", "use", allow_archived=True
    )
    assert spec.archived


# --- Brainstorm-then-Select prompts (Summers-Stay 2022) --------------------


def test_brainstorm_select_registry_has_all_five():
    expected = {"brainstorm", "advantages", "drawbacks",
                "utility_eval", "originality_eval"}
    assert set(BRAINSTORM_SELECT_REGISTRY) == expected


def test_brainstorm_v1_renders_with_object_and_includes_count():
    [_, msg] = AUT_BRAINSTORM_V1.render(object="a brick")
    user = msg["content"]
    assert "a brick" in user
    assert "List 10 creative uses" in user
    assert "clever, unusual, interesting, uncommon, humorous" in user


def test_advantages_drawbacks_prompts_have_qa_shape():
    for spec in (AUT_BS_ADVANTAGES_V1, AUT_BS_DRAWBACKS_V1):
        [_, msg] = spec.render(object="a brick", purpose="use it as a doorstop")
        u = msg["content"]
        assert u.startswith("Q: Name one or more")
        assert u.rstrip().endswith("A:")
        assert "a brick" in u
        assert "use it as a doorstop" in u


def test_util_eval_renders_all_four_placeholders_and_ends_with_a_colon():
    [_, msg] = AUT_BS_UTIL_EVAL_V1.render(
        object="a brick", purpose="use it as a doorstop",
        advantages="It is heavy.", drawbacks="It scratches the floor.",
    )
    u = msg["content"]
    assert "Advantages: It is heavy." in u
    assert "Drawbacks: It scratches the floor." in u
    assert "use it as a doorstop" in u
    assert u.rstrip().endswith("A:")        # next-token probe site


def test_orig_eval_renders_and_ends_with_a_colon():
    [_, msg] = AUT_BS_ORIG_EVAL_V1.render(
        object="a brick", purpose="use it as a doorstop",
    )
    u = msg["content"]
    assert "would you be surprised" in u
    assert "novel idea" in u
    assert u.rstrip().endswith("A:")
