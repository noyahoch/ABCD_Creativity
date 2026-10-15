"""Loads the real MacGyver xlsx. Skipped if the file isn't present so CI
elsewhere still passes."""

from __future__ import annotations

from pathlib import Path

import pytest

from bridge_steering.datasets import MacGyverDataset, MacGyverSubset


XLSX = Path(__file__).resolve().parents[2] / "data" / "macgyver" / "problem_solution_pair.xlsx"


pytestmark = pytest.mark.skipif(
    not XLSX.is_file(), reason=f"MacGyver xlsx missing: {XLSX}"
)


# Locked counts from the canonical xlsx (sha256 stamped on the dataset).
# If these numbers change, either the file changed or the filter changed.
EXPECTED_COUNTS = {
    MacGyverSubset.ALL: 1683,
    MacGyverSubset.SOLVABLE: 1306,
    MacGyverSubset.SOLVABLE_UNCONVENTIONAL: 956,
    MacGyverSubset.SOLVABLE_CONVENTIONAL: 350,
    MacGyverSubset.UNSOLVABLE: 377,
}


@pytest.mark.parametrize("subset,expected", list(EXPECTED_COUNTS.items()))
def test_subset_counts_match_locked_fixture(subset, expected):
    ds = MacGyverDataset.load(XLSX, subset.value)
    assert len(ds.items) == expected, (
        f"{subset.value}: expected {expected} items, got {len(ds.items)}. "
        f"xlsx sha256={ds.sha256[:12]}..."
    )


def test_solvable_and_unsolvable_partition_all():
    """SOLVABLE + UNSOLVABLE should sum to ALL (no third value of Solvable?)."""
    all_n = EXPECTED_COUNTS[MacGyverSubset.ALL]
    s = EXPECTED_COUNTS[MacGyverSubset.SOLVABLE]
    u = EXPECTED_COUNTS[MacGyverSubset.UNSOLVABLE]
    assert s + u == all_n


def test_solvable_unconventional_plus_conventional_equals_solvable():
    s = EXPECTED_COUNTS[MacGyverSubset.SOLVABLE]
    su = EXPECTED_COUNTS[MacGyverSubset.SOLVABLE_UNCONVENTIONAL]
    sc = EXPECTED_COUNTS[MacGyverSubset.SOLVABLE_CONVENTIONAL]
    assert su + sc == s


def test_first_item_has_required_row_id_columns():
    ds = MacGyverDataset.load(XLSX)
    item = ds.items[0]
    # Stable row IDs (see PLAN.md: every stochastic-stage CSV must carry these)
    for col in ("id", "source_id", "task_type", "domain",
                "problem_id", "problem_text"):
        assert col in item, f"missing required field: {col}"
    assert item["task_type"] == "macgyver"


def test_max_items_truncates():
    ds = MacGyverDataset.load(XLSX, max_items=10)
    assert len(ds.items) == 10


def test_unknown_subset_raises():
    with pytest.raises(ValueError):
        MacGyverDataset.load(XLSX, "not_a_subset")


def test_dataset_sha_is_stable_across_loads():
    """Two loads must agree on the file hash."""
    a = MacGyverDataset.load(XLSX, "all")
    b = MacGyverDataset.load(XLSX, "solvable")
    assert a.sha256 == b.sha256
