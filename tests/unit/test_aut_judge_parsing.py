"""Parser tests for the AUT judge (no network)."""

from __future__ import annotations

from bridge_steering.stages.aut_judge import parse_judge_response


def test_strict_json_parses():
    raw = '{"analysis": {"x": "y"}, "scores": {"novelty": 7, "usability": 6}}'
    assert parse_judge_response(raw) == (7, 6)


def test_json_with_wrapper_text_is_extracted():
    raw = 'Sure! Here is the JSON:\n{"scores": {"novelty": 4, "usability": 8}}\nLet me know.'
    assert parse_judge_response(raw) == (4, 8)


def test_out_of_range_clamps():
    raw = '{"scores": {"novelty": 15, "usability": -3}}'
    assert parse_judge_response(raw) == (10, 1)


def test_regex_fallback_for_malformed_json():
    raw = '{"scores": novelty: 7,  usability : 5'   # missing braces
    assert parse_judge_response(raw) == (7, 5)


def test_garbage_returns_none():
    assert parse_judge_response("nothing here") == (None, None)


def test_empty_input_returns_none():
    assert parse_judge_response("") == (None, None)
