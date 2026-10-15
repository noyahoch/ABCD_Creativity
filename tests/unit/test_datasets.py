"""Unit tests for dataset loaders and split determinism."""

from __future__ import annotations

import pytest

from bridge_steering.datasets import (
    ABCDDataset,
    compute_eval_indices,
    split_items_by_eval_indices,
)


def test_compute_eval_indices_deterministic():
    a = compute_eval_indices(100, 7, 10)
    b = compute_eval_indices(100, 7, 10)
    assert a == b
    assert len(a) == 10 and a == sorted(a)


def test_compute_eval_indices_clamps_holdout_above_n():
    idx = compute_eval_indices(5, 1, 100)
    assert sorted(idx) == [0, 1, 2, 3, 4]


def test_compute_eval_indices_matches_legacy_fixture():
    # Locked in: deterministic across Python versions.
    # If random.Random's algorithm changes, this fixture must be re-pinned.
    assert compute_eval_indices(100, 7, 10) == [6, 9, 12, 19, 41, 46, 50, 68, 74, 83]


def test_compute_eval_indices_holdout_zero():
    assert compute_eval_indices(10, 5, 0) == []


def test_compute_eval_indices_n_zero():
    assert compute_eval_indices(0, 5, 10) == []


def test_compute_eval_indices_holdout_negative_raises():
    with pytest.raises(ValueError):
        compute_eval_indices(10, 1, -1)


def test_split_items_by_eval_indices():
    data = [{"id": i} for i in range(5)]
    ev, tr, ev_set = split_items_by_eval_indices(data, [1, 3])
    assert [it["id"] for it in ev] == [1, 3]
    assert [it["id"] for it in tr] == [0, 2, 4]
    assert ev_set == {1, 3}


def test_abcd_dataset_load_stamps_ids(abcd_tiny_path):
    ds = ABCDDataset.load(abcd_tiny_path)
    assert len(ds.items) == 3
    assert all("source_id" in it for it in ds.items)
    assert all(it["task_type"] == "aut" for it in ds.items)
    assert ds.sha256


def test_abcd_split(abcd_tiny_path):
    ds = ABCDDataset.load(abcd_tiny_path)
    split = ds.split(eval_seed=7, eval_holdout=2)
    assert len(split.eval_items) == 2
    assert len(split.train_items) == 1
    assert set(split.train_indices) | set(split.eval_indices) == {0, 1, 2}
