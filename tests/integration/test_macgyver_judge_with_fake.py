"""End-to-end test of macgyver_judge using a fake judge call."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bridge_steering.config import JudgeConfig, MacGyverInferenceConfig, DecodingConfig, VllmJudgeBackend
from bridge_steering.prompts.macgyver import MACGYVER_JUDGE_ADDITIVE_5PT_V1
from bridge_steering.repro import build_run_context


def _seed_macgyver_infer(run_dir: Path) -> None:
    d = run_dir / "macgyver_infer"
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_id": "541", "task_type": "macgyver", "domain": "macgyver",
            "split": "eval", "source_id": "541", "sample_idx": 0, "seed": 42,
            "problem_id": "541", "problem_text": "Cross a river using only X.",
            "method": "baseline",
            "prompt_id": "macgyver_user_default_v1", "prompt_sha256": "sha256:00",
            "prompt_input_hash": "sha256:00",
            "model_name": "fake", "model_revision": "", "decoding_json": "{}",
            "raw_response": "Step 1: Use X to do Y.",
            "parsed_response": "Step 1: Use X to do Y.",
            "output_hash": "sha256:00",
            "solvable": True, "unconventional": True,
            "ref_solution": "", "ref_label": "",
        },
    ]
    p = d / "baseline_results.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_macgyver_judge_writes_quality_score(base_run_cfg, tmp_path):
    from bridge_steering.stages.macgyver_judge import run

    base_run_cfg.macgyver_data = tmp_path / "fake.xlsx"  # not actually opened
    base_run_cfg.macgyver_data.touch()
    base_run_cfg.decoding_macgyver = DecodingConfig()
    base_run_cfg.macgyver = MacGyverInferenceConfig()
    base_run_cfg.macgyver_judge = JudgeConfig(
        backend=VllmJudgeBackend(),
        prompt_id=MACGYVER_JUDGE_ADDITIVE_5PT_V1.id,
        max_concurrent=1,
    )

    ctx = build_run_context(base_run_cfg)
    _seed_macgyver_infer(ctx.run_dir)

    async def fake_judge(item_row):
        return {
            "quality_score": 4,
            "raw_response": "All steps reasonable.\nScore: 4",
            "spec": MACGYVER_JUDGE_ADDITIVE_5PT_V1,
            "prompt_input": "fake prompt",
            "error": "",
        }

    res = run(ctx, judge_async=fake_judge)
    assert res.name == "macgyver_judge"

    out = ctx.stage_dir("macgyver_judge") / "baseline_judge_scores.csv"
    rows = list(csv.DictReader(out.open("r", encoding="utf-8")))
    assert len(rows) == 1
    r = rows[0]
    assert int(r["quality_score"]) == 4
    assert r["judge_prompt_id"] == MACGYVER_JUDGE_ADDITIVE_5PT_V1.id
    assert r["inference_prompt_id"] == "macgyver_user_default_v1"
