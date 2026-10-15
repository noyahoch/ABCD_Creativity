"""Pure-Python strategy selectors (no model needed)."""

from __future__ import annotations

import pytest

from bridge_steering.config import (
    BrainstormSelectConfig, ConfigError,
    ThresholdStrategy, TopFractionStrategy, TopFractionIntersectionStrategy,
    TopKPerItemStrategy,
    validate as cfg_validate,
)
from bridge_steering.stages.aut_brainstorm_select import apply_strategy


def _row(u, o, item_id="i0", **extra):
    return {
        "utility_conf": u, "originality_conf": o,
        "item_id": item_id,
        **extra,
    }


# --- Threshold --------------------------------------------------------------


def test_threshold_combine_and():
    rows = [_row(0.6, 0.6), _row(0.4, 0.9), _row(0.9, 0.4), _row(0.9, 0.9)]
    s = ThresholdStrategy(utility_threshold=0.5, originality_threshold=0.5, combine="and")
    kept = [r for r in apply_strategy(rows, s) if r["kept"]]
    assert len(kept) == 2  # only (0.6,0.6) and (0.9,0.9)


def test_threshold_combine_or():
    rows = [_row(0.6, 0.0), _row(0.0, 0.6), _row(0.4, 0.4)]
    s = ThresholdStrategy(utility_threshold=0.5, originality_threshold=0.5, combine="or")
    kept = [r for r in apply_strategy(rows, s) if r["kept"]]
    assert len(kept) == 2  # first two pass; third doesn't


def test_threshold_single_axis_ignores_other():
    rows = [_row(0.6, 0.0), _row(0.0, 0.6), _row(0.4, 0.9)]
    s = ThresholdStrategy(utility_threshold=0.5)  # originality_threshold=None
    kept = [r for r in apply_strategy(rows, s) if r["kept"]]
    assert len(kept) == 1  # only row 0 has utility>=0.5


def test_threshold_keep_reason_records_active_rule():
    rows = [_row(0.9, 0.9)]
    s = ThresholdStrategy(utility_threshold=0.5, originality_threshold=0.5, combine="and")
    [r] = apply_strategy(rows, s)
    assert r["kept"] is True
    assert "u>=0.5" in r["keep_reason"] and "and" in r["keep_reason"]


# --- TopFraction ------------------------------------------------------------


def test_top_fraction_keeps_correct_count():
    rows = [_row(i / 100, (100 - i) / 100) for i in range(100)]
    s = TopFractionStrategy(keep_fraction=0.25, rank_by="originality", tiebreaker="utility")
    kept = [r for r in apply_strategy(rows, s) if r["kept"]]
    assert len(kept) == 25


def test_top_fraction_rank_by_originality_tiebreak_utility():
    # Two rows tied on originality (0.5), different utilities → utility breaks the tie.
    rows = [
        _row(0.1, 0.5),  # tied
        _row(0.9, 0.5),  # tied — should win the tiebreak
        _row(0.5, 0.4),  # ranked lower
    ]
    s = TopFractionStrategy(keep_fraction=2 / 3, rank_by="originality", tiebreaker="utility")
    selected = apply_strategy(rows, s)
    kept_indexes = [i for i, r in enumerate(selected) if r["kept"]]
    # Should keep the two highest-originality, breaking the tie via utility.
    assert 1 in kept_indexes and 0 in kept_indexes
    assert 2 not in kept_indexes


def test_top_fraction_tiebreaker_none_uses_stable_sort():
    rows = [_row(0.1, 0.5), _row(0.9, 0.5)]
    s = TopFractionStrategy(keep_fraction=0.5, rank_by="originality", tiebreaker="none")
    out = apply_strategy(rows, s)
    # Both tied on originality and tiebreaker=none ⇒ first encountered wins.
    assert out[0]["kept"] is True
    assert out[1]["kept"] is False


def test_top_fraction_keep_reason_stable():
    rows = [_row(0.5, 0.5)]
    s = TopFractionStrategy(keep_fraction=1.0, rank_by="originality", tiebreaker="utility")
    [r] = apply_strategy(rows, s)
    assert r["kept"] is True
    assert "rank_originality" in r["keep_reason"]
    assert "tb_utility" in r["keep_reason"]


# --- TopFractionIntersection ----------------------------------------------


def test_intersection_can_be_smaller_than_k():
    # Top-K on utility = {0,1,2}; top-K on originality = {3,4,5} ⇒ ∅
    rows = [
        _row(0.9, 0.1), _row(0.8, 0.1), _row(0.7, 0.1),
        _row(0.1, 0.9), _row(0.1, 0.8), _row(0.1, 0.7),
    ]
    s = TopFractionIntersectionStrategy(keep_fraction=0.5)
    kept = [r for r in apply_strategy(rows, s) if r["kept"]]
    assert kept == []


def test_intersection_keeps_overlap():
    # Rows that are top on BOTH axes — kept count should equal that overlap.
    rows = [
        _row(0.9, 0.9), _row(0.8, 0.8),    # top on both
        _row(0.7, 0.1), _row(0.1, 0.7),    # only top on one
    ]
    s = TopFractionIntersectionStrategy(keep_fraction=0.5)
    kept = [i for i, r in enumerate(apply_strategy(rows, s)) if r["kept"]]
    assert kept == [0, 1]


# --- Empty input ------------------------------------------------------------


def test_empty_input_returns_empty():
    for s in (
        ThresholdStrategy(utility_threshold=0.5),
        TopFractionStrategy(keep_fraction=0.5),
        TopFractionIntersectionStrategy(keep_fraction=0.5),
        TopKPerItemStrategy(k_per_item=3),
    ):
        assert apply_strategy([], s) == []


# --- TopKPerItem ------------------------------------------------------------


def test_top_k_per_item_picks_k_per_group():
    # Two items, 4 candidates each, varying utility; rank_by=utility picks
    # the top 2 per item.
    rows = [
        _row(0.9, 0.0, item_id="a"),  # a: top
        _row(0.8, 0.0, item_id="a"),  # a: second
        _row(0.2, 0.0, item_id="a"),
        _row(0.1, 0.0, item_id="a"),
        _row(0.7, 0.0, item_id="b"),  # b: top
        _row(0.6, 0.0, item_id="b"),  # b: second
        _row(0.2, 0.0, item_id="b"),
        _row(0.1, 0.0, item_id="b"),
    ]
    s = TopKPerItemStrategy(k_per_item=2, rank_by="utility", tiebreaker="none")
    kept_idx = [i for i, r in enumerate(apply_strategy(rows, s)) if r["kept"]]
    assert kept_idx == [0, 1, 4, 5]


def test_top_k_per_item_rank_by_sum_combines_axes():
    # Sum-rank should pick the row with highest u+o, even when it's not max
    # on either axis alone.
    rows = [
        _row(0.5, 0.5, item_id="a"),   # sum 1.0 — wins overall
        _row(0.9, 0.0, item_id="a"),   # sum 0.9
        _row(0.0, 0.8, item_id="a"),   # sum 0.8
    ]
    s = TopKPerItemStrategy(k_per_item=1, rank_by="sum", tiebreaker="none")
    [kept] = [i for i, r in enumerate(apply_strategy(rows, s)) if r["kept"]]
    assert kept == 0


def test_top_k_per_item_keeps_all_when_fewer_than_k():
    # Item has 2 candidates, k=8 → keep both, do not fabricate.
    rows = [_row(0.1, 0.1, item_id="a"), _row(0.2, 0.2, item_id="a")]
    s = TopKPerItemStrategy(k_per_item=8, rank_by="sum", tiebreaker="none")
    out = apply_strategy(rows, s)
    assert all(r["kept"] for r in out)


def test_top_k_per_item_tiebreaker_resolves_equal_primary():
    # Two rows tied on primary (utility=0.5); tiebreaker=originality picks
    # the one with higher originality.
    rows = [
        _row(0.5, 0.1, item_id="a"),    # tied on utility
        _row(0.5, 0.9, item_id="a"),    # tied on utility — wins via tiebreaker
        _row(0.4, 0.5, item_id="a"),    # lower utility
    ]
    s = TopKPerItemStrategy(k_per_item=1, rank_by="utility", tiebreaker="originality")
    [kept] = [i for i, r in enumerate(apply_strategy(rows, s)) if r["kept"]]
    assert kept == 1


def test_top_k_per_item_keep_reason_stable():
    rows = [_row(0.5, 0.5, item_id="a")]
    s = TopKPerItemStrategy(k_per_item=1, rank_by="sum", tiebreaker="originality")
    [r] = apply_strategy(rows, s)
    assert r["kept"] is True
    assert "top1_per_item_rank_sum" in r["keep_reason"]
    assert "tb_originality" in r["keep_reason"]


def test_top_k_per_item_invalid_k_raises(base_run_cfg):
    base_run_cfg.brainstorm_select = BrainstormSelectConfig(
        strategy=TopKPerItemStrategy(k_per_item=0)
    )
    with pytest.raises(ConfigError, match="k_per_item must be"):
        cfg_validate(base_run_cfg)


def test_top_k_per_item_tiebreaker_cannot_equal_rank_by(base_run_cfg):
    base_run_cfg.brainstorm_select = BrainstormSelectConfig(
        strategy=TopKPerItemStrategy(rank_by="utility", tiebreaker="utility")
    )
    with pytest.raises(ConfigError, match="tiebreaker .* cannot equal rank_by"):
        cfg_validate(base_run_cfg)


# --- Strategy validation rules (config-level, not row-level) ---------------


def test_threshold_requires_at_least_one_axis(base_run_cfg):
    base_run_cfg.brainstorm_select = BrainstormSelectConfig(
        strategy=ThresholdStrategy()
    )
    with pytest.raises(ConfigError, match="utility_threshold / originality_threshold"):
        cfg_validate(base_run_cfg)


def test_top_fraction_zero_keep_fraction_raises(base_run_cfg):
    base_run_cfg.brainstorm_select = BrainstormSelectConfig(
        strategy=TopFractionStrategy(keep_fraction=0.0)
    )
    with pytest.raises(ConfigError, match="keep_fraction must be in"):
        cfg_validate(base_run_cfg)


def test_top_fraction_over_one_raises(base_run_cfg):
    base_run_cfg.brainstorm_select = BrainstormSelectConfig(
        strategy=TopFractionStrategy(keep_fraction=1.5)
    )
    with pytest.raises(ConfigError, match="keep_fraction must be in"):
        cfg_validate(base_run_cfg)


def test_top_fraction_tiebreaker_cannot_equal_rank_by(base_run_cfg):
    base_run_cfg.brainstorm_select = BrainstormSelectConfig(
        strategy=TopFractionStrategy(rank_by="utility", tiebreaker="utility")
    )
    with pytest.raises(ConfigError, match="tiebreaker .* cannot equal rank_by"):
        cfg_validate(base_run_cfg)
