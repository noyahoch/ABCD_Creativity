"""End-to-end test for the standalone ``macgyver_novelty`` stage.

Builds a tiny n-gram corpus, seeds a ``macgyver_infer`` CSV with two
solutions, runs ``macgyver_novelty``, and asserts the per-row originality
columns line up with the corpus.
"""

from __future__ import annotations

import csv
from pathlib import Path

from bridge_steering.config import OriginalityConfig
from bridge_steering.repro import build_run_context


def _build_tiny_index(tmp_path: Path) -> Path:
    from bridge_steering.novelty import NgramCorpusIndex
    idx = NgramCorpusIndex.build_from_texts(
        [
            "stack the bricks to climb up the wall",
            "tie the rope around the heavy stone",
            "wedge the door open with a folded shoe",
        ],
        n_values=[2, 3],
        progress=False,
    )
    path = tmp_path / "tiny_macgyver.npz"
    idx.save(path)
    return path


def _seed_macgyver_infer_csv(run_dir: Path, solutions: list[str]) -> Path:
    d = run_dir / "macgyver_infer"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sol in enumerate(solutions):
        rows.append({
            "item_id": f"MAC_{i}", "task_type": "macgyver", "domain": "macgyver",
            "split": "eval", "source_id": f"MAC_{i}", "sample_idx": 0, "seed": 42,
            "problem_id": f"PROB_{i}", "problem_text": "fake problem",
            "method": "baseline",
            "prompt_id": "fake", "prompt_sha256": "sha256:00",
            "prompt_input_hash": "sha256:00",
            "model_name": "fake", "model_revision": "", "decoding_json": "{}",
            "raw_response": sol,
            "parsed_response": sol,
            "output_hash": "sha256:00",
            "solvable": True, "unconventional": True,
            "ref_solution": "", "ref_label": "",
        })
    p = d / "baseline_results.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


def test_macgyver_novelty_writes_per_solution_columns(base_run_cfg, tmp_path):
    from bridge_steering.stages.macgyver_novelty import run

    index_path = _build_tiny_index(tmp_path)
    base_run_cfg.originality = OriginalityConfig(
        ngram_index=index_path, n_values=[2, 3],
    )

    ctx = build_run_context(base_run_cfg)
    # Solution 1: identical to a corpus document → originality 0.
    # Solution 2: text disjoint from the corpus → originality 1.
    _seed_macgyver_infer_csv(ctx.run_dir, [
        "stack the bricks to climb up the wall",
        "blow gently on the candle until it sings",
    ])

    res = run(ctx)
    assert res.name == "macgyver_novelty"

    out = ctx.stage_dir("macgyver_novelty") / "baseline_novelty.csv"
    rows = list(csv.DictReader(out.open("r", encoding="utf-8")))
    assert len(rows) == 2

    assert "originality_ngram_2" in rows[0]
    assert "originality_ngram_3" in rows[0]
    assert rows[0]["item_id"] == "MAC_0"
    assert rows[0]["problem_id"] == "PROB_0"

    # Row 1 same as corpus → 0 ; row 2 disjoint → 1.
    assert float(rows[0]["originality_ngram_2"]) == 0.0
    assert float(rows[0]["originality_ngram_3"]) == 0.0
    assert float(rows[1]["originality_ngram_2"]) == 1.0
    assert float(rows[1]["originality_ngram_3"]) == 1.0

    # Provenance.
    assert "ngram_index" in res.inputs
    assert res.inputs["ngram_index"]["path"].endswith("tiny_macgyver.npz")
    assert res.notes["stage_config"]["n_values"] == [2, 3]
