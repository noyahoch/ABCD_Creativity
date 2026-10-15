"""Regression test: archived prompts cannot be selected for new runs.

Keeping old prompts resolvable lets us replay old runs, but selecting one
for a new experiment should be a deliberate, audited choice.
"""

from __future__ import annotations

import pytest

from bridge_steering.config import ConfigError, JudgeConfig, VllmJudgeBackend, validate
from bridge_steering.prompts import resolve


def _judge(prompt_id: str) -> JudgeConfig:
    return JudgeConfig(backend=VllmJudgeBackend(), prompt_id=prompt_id)


def test_resolve_archived_blocked_by_default():
    with pytest.raises(ValueError, match="archived"):
        resolve("aut_judge_novelty_usability_v0_5pt")


def test_resolve_archived_allowed_with_opt_in():
    spec = resolve("aut_judge_novelty_usability_v0_5pt", allow_archived=True)
    assert spec.archived is True


def test_validate_blocks_archived_judge_prompt(base_run_cfg):
    base_run_cfg.aut_judge = _judge("aut_judge_novelty_usability_v0_5pt")
    with pytest.raises(ValueError, match="archived"):
        validate(base_run_cfg)


def test_validate_allows_archived_judge_prompt_with_opt_in(base_run_cfg):
    base_run_cfg.aut_judge = _judge("aut_judge_novelty_usability_v0_5pt")
    base_run_cfg.allow_archived_prompts = True
    validate(base_run_cfg)
