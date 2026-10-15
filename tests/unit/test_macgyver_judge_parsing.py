"""Parser tests for the MacGyver judge (no network)."""

from __future__ import annotations

from bridge_steering.stages.macgyver_judge import parse_macgyver_score


def test_score_colon_format():
    assert parse_macgyver_score("Long reasoning…\nScore: 3") == 3


def test_fraction_format():
    assert parse_macgyver_score("Final assessment: 4/5") == 4


def test_lowercase_score_label():
    assert parse_macgyver_score("score: 5") == 5


def test_garbage_returns_none():
    assert parse_macgyver_score("not a score anywhere") is None


def test_empty_returns_none():
    assert parse_macgyver_score("") is None


def test_out_of_range_in_colon_returns_none():
    # Colon regex captures "7", which fails the 0-5 check → falls through to
    # fraction regex (no match) → returns None.
    assert parse_macgyver_score("Score: 7") is None
