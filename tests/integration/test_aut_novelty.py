"""End-to-end test for the standalone ``aut_novelty`` stage.

Builds a tiny in-memory n-gram corpus, seeds an ``aut_infer`` CSV with two
uses, runs ``aut_novelty``, and asserts that:
* a per-method novelty CSV is produced with one row per use
* ``originality_ngram_<N>`` columns are present for every requested N
* values are 0.0 for a use whose n-grams ARE in the corpus, 1.0 for a use
  whose n-grams are NOT
* the n-gram index is recorded as a stage input (sha + path)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from bridge_steering.config import OriginalityConfig
from bridge_steering.repro import build_run_context


def _build_tiny_index(tmp_path: Path) -> Path:
    """Build an n-gram index over a tiny corpus and save to .npz."""
    from bridge_steering.novelty import NgramCorpusIndex
    idx = NgramCorpusIndex.build_from_texts(
        [
            "use a brick as a paperweight",
            "use a brick to prop open a door",
            "use a brick as a bookend",
        ],
        n_values=[2, 3],
        progress=False,
    )
    path = tmp_path / "tiny.npz"
    idx.save(path)
    return path


def _seed_aut_infer_csv(run_dir: Path, uses: list[str]) -> Path:
    """Materialize an aut_infer/baseline_results.csv with the given uses."""
    d = run_dir / "aut_infer"
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_id": "AUT_1", "task_type": "aut", "domain": "AUT", "split": "eval",
            "source_id": "AUT_1", "sample_idx": 0, "seed": 42,
            "object": "brick", "object_source": "A_OBJECT",
            "method": "baseline",
            "prompt_id": "aut_user_object_only_v1", "prompt_sha256": "sha256:00",
            "prompt_input_hash": "sha256:00",
            "model_name": "fake", "model_revision": "", "decoding_json": "{}",
            "raw_response": "\n".join(f"- {u}" for u in uses),
            "parsed_response": json.dumps(uses),
            "output_hash": "sha256:00",
        }
    ]
    p = d / "baseline_results.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


def test_aut_novelty_columns_match_corpus(base_run_cfg, tmp_path):
    from bridge_steering.stages.aut_novelty import run

    index_path = _build_tiny_index(tmp_path)
    base_run_cfg.originality = OriginalityConfig(
        ngram_index=index_path, n_values=[2, 3],
    )

    ctx = build_run_context(base_run_cfg)
    # Use 1: identical to a corpus document → originality should be 0.
    # Use 2: text whose 2-grams and 3-grams do NOT appear in the corpus.
    _seed_aut_infer_csv(ctx.run_dir, [
        "use a brick as a paperweight",
        "cats fly into the cold cold space",
    ])

    res = run(ctx)
    assert res.name == "aut_novelty"

    out = ctx.stage_dir("aut_novelty") / "baseline_novelty.csv"
    rows = list(csv.DictReader(out.open("r", encoding="utf-8")))
    assert len(rows) == 2

    # Columns present for every requested N.
    assert "originality_ngram_2" in rows[0]
    assert "originality_ngram_3" in rows[0]
    # Base columns we depend on for joining downstream are populated.
    assert rows[0]["use_text"] == "use a brick as a paperweight"
    assert rows[0]["method"] == "baseline"

    # Row 1 (same as corpus): all n-grams seen → originality 0.
    assert float(rows[0]["originality_ngram_2"]) == 0.0
    assert float(rows[0]["originality_ngram_3"]) == 0.0
    # Row 2: text disjoint from corpus → originality 1.
    assert float(rows[1]["originality_ngram_2"]) == 1.0
    assert float(rows[1]["originality_ngram_3"]) == 1.0

    # The StageResult records the n-gram index as a stage input (the
    # pipeline.run_stage wrapper persists this to the manifest).
    assert "ngram_index" in res.inputs
    assert res.inputs["ngram_index"]["path"].endswith("tiny.npz")
    assert res.inputs["ngram_index"]["sha256"].startswith("sha256:")
    # And reproducibility note records which Ns were used.
    assert res.notes["stage_config"]["n_values"] == [2, 3]
