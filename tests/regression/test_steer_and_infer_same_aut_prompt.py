"""Regression test: steer_collect and aut_infer build the SAME messages.

This is THE bug the new design exists to prevent — different prompts for
activation collection vs. AUT evaluation. The check is byte-identical
equality of the full chat-messages list (system + user), per subject and
per role override.
"""

from __future__ import annotations

from bridge_steering.config import AUTPromptOverride, AUTPrompts
from bridge_steering.stages.aut_infer import _build_user_messages as infer_msgs
from bridge_steering.stages.steer_collect import build_pairs


def _steer_messages(item, cfg):
    """The full chat-messages list steer_collect would feed to the model."""
    pair = build_pairs([item], cfg)[0]
    return pair["prompt_msgs"]


def test_coupled_object_only(base_run_cfg, abcd_tiny_items):
    """Default coupled mode: both stages render identically."""
    for item in abcd_tiny_items:
        s = _steer_messages(item, base_run_cfg)
        i_msgs, _, _ = infer_msgs(item, base_run_cfg, "inference")
        assert s == i_msgs


def test_coupled_full_instruction(base_run_cfg, abcd_tiny_items):
    """Subject swap is honoured by both stages identically."""
    base_run_cfg.aut.prompts = AUTPrompts(default_subject="full_instruction")
    for item in abcd_tiny_items:
        s = _steer_messages(item, base_run_cfg)
        i_msgs, _, _ = infer_msgs(item, base_run_cfg, "inference")
        assert s == i_msgs


def test_decoupled_with_allow_does_diverge(base_run_cfg, abcd_tiny_items):
    """When the user explicitly decouples, messages differ — and we can see it.

    This is the supported escape hatch; the validate() rule already requires
    a reason. The point of this test is to prove that decoupling actually
    affects the messages produced (i.e. the overrides are honoured).
    """
    base_run_cfg.aut.prompts = AUTPrompts(
        allow_decoupling=True,
        decoupling_reason="ablation",
        steering=AUTPromptOverride(subject="full_instruction"),
        inference=AUTPromptOverride(subject="object_only"),
    )
    diverged = 0
    for item in abcd_tiny_items:
        s = _steer_messages(item, base_run_cfg)
        i_msgs, _, _ = infer_msgs(item, base_run_cfg, "inference")
        if s != i_msgs:
            diverged += 1
    assert diverged > 0, (
        "Explicit decoupling didn't change any messages — overrides not honoured."
    )


def test_full_chat_list_compared_not_only_user(base_run_cfg, abcd_tiny_items):
    """If two prompts share the user string but differ in system, fail loudly."""
    s = _steer_messages(abcd_tiny_items[0], base_run_cfg)
    i_msgs, _, _ = infer_msgs(abcd_tiny_items[0], base_run_cfg, "inference")
    # Both have a 'system' role even when content is empty — i.e. the system
    # message is part of the comparison, not silently dropped.
    assert s[0]["role"] == i_msgs[0]["role"]
    assert s == i_msgs
