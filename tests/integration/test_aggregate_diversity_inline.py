"""Inline diversity columns in aggregate (uses a stubbed embedding fn).

The semantic-diversity computation in the production stage loads a real
sentence-transformer; here we monkeypatch the helper that does the work
so the test stays fast and deterministic.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bridge_steering.config import DiversityConfig
from bridge_steering.repro import build_run_context


def _seed_aut_infer_and_ocsai(run_dir: Path) -> None:
    """Materialize the minimum upstream outputs the aggregate stage reads."""
    aut_infer = run_dir / "aut_infer"
    aut_infer.mkdir(parents=True, exist_ok=True)
    (aut_infer / "baseline_results.csv").write_text(
        "item_id,method,parsed_response\n"
        'AUT_1,baseline,"[""use1""]"\n',
        encoding="utf-8",
    )
    # OCSAI: one row per reply (post-fix schema with reply_text).
    aut_ocsai = run_dir / "aut_ocsai"
    aut_ocsai.mkdir(parents=True, exist_ok=True)
    (aut_ocsai / "baseline_ocsai_scores.csv").write_text(
        "item_id,method,reply_text,ocsai_originality\n"
        'AUT_1,baseline,multi-use reply text,3.5\n',
        encoding="utf-8",
    )
    # Judge: per-use rows — aggregate reads use_text + use_count from here now.
    aut_judge = run_dir / "aut_judge"
    aut_judge.mkdir(parents=True, exist_ok=True)
    (aut_judge / "baseline_judge_scores.csv").write_text(
        "item_id,method,use_text,novelty,usability\n"
        "AUT_1,baseline,use a brick as a paperweight,5,6\n"
        "AUT_1,baseline,prop open a heavy door,7,5\n"
        "AUT_1,baseline,grind it to dust for masonry,6,6\n",
        encoding="utf-8",
    )


def test_diversity_columns_present_and_populated(base_run_cfg, monkeypatch):
    from bridge_steering.stages import aggregate as agg

    base_run_cfg.diversity = DiversityConfig()
    ctx = build_run_context(base_run_cfg)
    _seed_aut_infer_and_ocsai(ctx.run_dir)

    # Stub the diversity computation so the test doesn't load
    # sentence-transformers (which would be slow and require the package).
    # Asserts the aggregate stage threads this helper's output into the row.
    def fake_diversity(uses_by_item, div_cfg):
        return {"aut_semantic_diversity_mean_pairwise_cosine": 0.42}
    monkeypatch.setattr(agg, "_compute_within_item_diversity", fake_diversity)

    res = agg.run(ctx)
    assert res.name == "aggregate"

    out = ctx.stage_dir("aggregate") / "cross_stage_summary.csv"
    rows = list(csv.DictReader(out.open("r", encoding="utf-8")))
    # One row per method now (no more task_type column).
    [row] = [r for r in rows if r["method"] == "baseline"]

    # Single "Sem. div." column threaded from the helper.
    assert float(row["aut_semantic_diversity_mean_pairwise_cosine"]) == 0.42

    # aut_n_uses reflects the per-use CSV size, not the inference CSV size.
    assert int(row["aut_n_uses"]) == 3


def test_diversity_columns_absent_when_disabled(base_run_cfg, monkeypatch):
    from bridge_steering.stages import aggregate as agg

    base_run_cfg.diversity = None
    ctx = build_run_context(base_run_cfg)
    _seed_aut_infer_and_ocsai(ctx.run_dir)

    agg.run(ctx)
    out = ctx.stage_dir("aggregate") / "cross_stage_summary.csv"
    rows = list(csv.DictReader(out.open("r", encoding="utf-8")))
    assert "aut_semantic_diversity_mean_pairwise_cosine" not in rows[0]
