"""End-to-end test of aut_brainstorm_select with an injected fake scorer.

Verifies the stage's plumbing without loading a model: seed aut_infer with a
two-item CSV of brainstormed uses, inject a deterministic ``scorer``, run
the stage, and assert the output is filtered + relabeled correctly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from bridge_steering.config import BrainstormSelectConfig, TopKPerItemStrategy
from bridge_steering.repro import build_run_context


# Two items, three brainstormed uses each (so top-K-per-item with K=2 must
# drop the worst from each item).
USES_PER_ITEM = {
    "AUT_1": ["use as a paperweight", "use as a hammer", "use as a spoon"],
    "AUT_2": ["use as a doorstop",    "use as a ladder", "use as a hat"],
}

# Scorer rigged so that within each item, the two best are positions 0 and 1
# under rank_by="sum" (u_conf + o_conf).
FAKE_SCORES: dict[tuple[str, str], dict] = {
    # item AUT_1
    ("AUT_1", "use as a paperweight"): {"u_yes": 1.0, "u_no": -1.0, "o_yes": 0.5, "o_no": -0.5},  # sum=3.0
    ("AUT_1", "use as a hammer"):       {"u_yes": 0.7, "u_no": -0.7, "o_yes": 0.7, "o_no": -0.7},  # sum=2.8
    ("AUT_1", "use as a spoon"):        {"u_yes": -1.0, "u_no": 1.0, "o_yes": -0.5, "o_no": 0.5}, # sum=-3.0
    # item AUT_2
    ("AUT_2", "use as a doorstop"):     {"u_yes": 0.9, "u_no": -0.9, "o_yes": 0.4, "o_no": -0.4}, # sum=2.6
    ("AUT_2", "use as a ladder"):       {"u_yes": 0.6, "u_no": -0.6, "o_yes": 0.6, "o_no": -0.6}, # sum=2.4
    ("AUT_2", "use as a hat"):          {"u_yes": -0.8, "u_no": 0.8, "o_yes": -0.4, "o_no": 0.4},# sum=-2.4
}


def _seed_aut_infer(run_dir: Path) -> None:
    d = run_dir / "aut_infer"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for item_id, uses in USES_PER_ITEM.items():
        rows.append({
            "item_id": item_id, "task_type": "aut", "domain": "AUT", "split": "eval",
            "source_id": item_id, "sample_idx": 0, "seed": 42,
            "object": "brick", "object_source": "A_OBJECT",
            "method": "baseline",
            "prompt_id": "aut_brainstorm_v1", "prompt_sha256": "sha256:00",
            "prompt_input_hash": "sha256:00", "model_name": "fake",
            "model_revision": "", "decoding_json": "{}",
            "raw_response": "\n".join(f"- {u}" for u in uses),
            "parsed_response": json.dumps(uses),
            "output_hash": "sha256:00",
        })
    p = d / "baseline_results.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fake_scorer(row: dict) -> dict:
    s = FAKE_SCORES[(row["item_id"], row["use_text"])]
    return {
        "advantages_text": f"adv:{row['use_text']}",
        "drawbacks_text":  f"drw:{row['use_text']}",
        "utility_logit_yes": s["u_yes"], "utility_logit_no": s["u_no"],
        "utility_conf": s["u_yes"] - s["u_no"],
        "originality_logit_yes": s["o_yes"], "originality_logit_no": s["o_no"],
        "originality_conf": s["o_yes"] - s["o_no"],
    }


def test_brainstorm_select_writes_top_k_per_item_with_renamed_slug(base_run_cfg):
    from bridge_steering.stages.aut_brainstorm_select import run

    base_run_cfg.brainstorm_select = BrainstormSelectConfig(
        strategy=TopKPerItemStrategy(
            k_per_item=2, rank_by="sum", tiebreaker="originality",
        ),
        skip_chain=True,
        method_slug="brainstorm",
    )
    ctx = build_run_context(base_run_cfg)
    _seed_aut_infer(ctx.run_dir)

    result = run(ctx, scorer=_fake_scorer)

    # Output file is renamed via method_slug.
    out_path = ctx.run_dir / "aut_brainstorm_select" / "brainstorm_selected_results.csv"
    assert out_path.is_file()

    with out_path.open() as f:
        kept = list(csv.DictReader(f))

    # Top-2 per item × 2 items = 4 kept rows, no losers written.
    assert len(kept) == 4
    by_item: dict[str, list[str]] = {}
    for r in kept:
        by_item.setdefault(r["item_id"], []).append(r["use_text"])
    assert set(by_item["AUT_1"]) == {"use as a paperweight", "use as a hammer"}
    assert set(by_item["AUT_2"]) == {"use as a doorstop",    "use as a ladder"}

    # Method slug is rewritten on every row.
    assert all(r["method"] == "brainstorm" for r in kept)
    # Score columns are populated and confidence is logit_yes - logit_no.
    for r in kept:
        u_y = float(r["utility_logit_yes"]); u_n = float(r["utility_logit_no"])
        assert abs(float(r["utility_conf"]) - (u_y - u_n)) < 1e-9
        assert r["kept"] in ("True", "1", "true")
        assert r["keep_reason"].startswith("top2_per_item_rank_sum")

    # Manifest summary numbers reflect the filter.
    notes = result.notes["brainstorm_select"]
    assert notes["n_input_rows"] == 6
    assert notes["n_kept"] == 4
    assert abs(notes["kept_fraction_observed"] - (4 / 6)) < 1e-9
