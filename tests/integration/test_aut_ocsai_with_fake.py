"""End-to-end test of aut_ocsai using an injected fake score_batch."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from bridge_steering.config import OCSAIConfig
from bridge_steering.repro import build_run_context


def _seed_aut_infer(run_dir: Path) -> None:
    d = run_dir / "aut_infer"
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_id": "AUT_1", "task_type": "aut", "domain": "AUT", "split": "eval",
            "source_id": "AUT_1", "sample_idx": 0, "seed": 42,
            "object": "brick",
            "method": "baseline",
            "prompt_id": "aut_user_object_only_v1", "prompt_sha256": "sha256:00",
            "prompt_input_hash": "sha256:00", "model_name": "fake",
            "model_revision": "", "decoding_json": "{}",
            "raw_response": "- use as a paperweight",
            "parsed_response": json.dumps(["use as a paperweight"]),
            "output_hash": "sha256:00",
            "object_source": "A_OBJECT",
        },
    ]
    p = d / "baseline_results.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_aut_ocsai_writes_score_with_fake_scorer(base_run_cfg):
    from bridge_steering.stages.aut_ocsai import run

    base_run_cfg.ocsai = OCSAIConfig()
    ctx = build_run_context(base_run_cfg)
    _seed_aut_infer(ctx.run_dir)

    def fake_score_batch(rows, oc):
        return [(3.42, '{"scores": [3.42]}') for _ in rows]

    res = run(ctx, score_batch=fake_score_batch)
    assert res.name == "aut_ocsai"

    out = ctx.stage_dir("aut_ocsai") / "baseline_ocsai_scores.csv"
    rows = list(csv.DictReader(out.open("r", encoding="utf-8")))
    assert len(rows) == 1
    r = rows[0]
    assert float(r["ocsai_originality"]) == 3.42
    assert r["ocsai_model"] == "ocsai-davinci3"
    assert r["ocsai_request_hash"].startswith("sha256:")
    assert r["ocsai_request_prompt_id"] == "ocsai_openscoring_v1"
