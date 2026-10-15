"""End-to-end test of the aut_judge stage using a fake judge call.

Confirms the per-row CSV schema (incl. the forwarded inference_prompt_*
columns required by the v3 PLAN) and that the manifest will pick up the
judge_prompt_id.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bridge_steering.config import JudgeConfig, VllmJudgeBackend
from bridge_steering.prompts.aut import (
    AUT_USER_OBJECT_ONLY_V1, JUDGE_NOVELTY_USABILITY_V5,
)
from bridge_steering.repro import build_run_context


def _seed_aut_infer_csv(run_dir: Path) -> Path:
    infer_dir = run_dir / "aut_infer"
    infer_dir.mkdir(parents=True, exist_ok=True)
    p = infer_dir / "baseline_results.csv"
    # Two items, one use each, full audit columns from aut_infer.
    rows = [
        {
            "item_id": "AUT_1", "task_type": "aut", "domain": "AUT", "split": "eval",
            "source_id": "AUT_1", "sample_idx": 0, "seed": 42,
            "object": "brick", "object_source": "A_OBJECT",
            "method": "baseline",
            "prompt_id": AUT_USER_OBJECT_ONLY_V1.id,
            "prompt_sha256": "sha256:" + AUT_USER_OBJECT_ONLY_V1.sha256(),
            "prompt_input_hash": "sha256:deadbeef",
            "model_name": "fake-model", "model_revision": "",
            "decoding_json": json.dumps({"temperature": 0.0}),
            "raw_response": "- store thermal mass for an oven",
            "parsed_response": json.dumps(["store thermal mass for an oven"]),
            "output_hash": "sha256:cafebabe",
        },
    ]
    cols = list(rows[0])
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return p


def test_aut_judge_writes_required_columns(base_run_cfg, monkeypatch):
    from bridge_steering.stages.aut_judge import run

    base_run_cfg.aut_judge = JudgeConfig(
        backend=VllmJudgeBackend(),
        prompt_id=JUDGE_NOVELTY_USABILITY_V5.id,
        max_concurrent=1,
    )

    ctx = build_run_context(base_run_cfg)
    _seed_aut_infer_csv(ctx.run_dir)

    async def fake_judge(item_row):
        return {
            "novelty": 7, "usability": 6, "overall": 7,
            "raw_response": '{"scores": {"novelty": 7, "usability": 6}}',
            "spec": JUDGE_NOVELTY_USABILITY_V5,
            "prompt_input": "fake prompt text",
            "error": "",
        }

    result = run(ctx, judge_async=fake_judge)
    assert result.name == "aut_judge"

    out = ctx.stage_dir("aut_judge") / "baseline_judge_scores.csv"
    assert out.is_file()
    rows = list(csv.DictReader(out.open("r", encoding="utf-8")))
    assert len(rows) == 1
    r = rows[0]

    # Inference-prompt columns forwarded from the upstream CSV — the bug fix.
    assert r["inference_prompt_id"] == AUT_USER_OBJECT_ONLY_V1.id
    assert r["inference_prompt_sha256"].startswith("sha256:")
    # Judge-prompt columns recorded from the judge spec.
    assert r["judge_prompt_id"] == JUDGE_NOVELTY_USABILITY_V5.id
    assert r["judge_prompt_sha256"].startswith("sha256:")
    assert r["judge_prompt_input_hash"].startswith("sha256:")
    # Scores parsed from the canned JSON.
    assert int(r["novelty"]) == 7
    assert int(r["usability"]) == 6
    assert int(r["overall"]) == 7
    # Reproducibility tail.
    assert r["judge_model"] == "Qwen/Qwen2.5-32B-Instruct-AWQ"
    assert r["output_hash"].startswith("sha256:")
