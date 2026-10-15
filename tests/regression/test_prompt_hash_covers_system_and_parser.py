"""Regression test: PromptSpec.sha256 must cover system + parser_contract.

Drift in chat structure — e.g. someone moves a sentence from system to
user, or changes the parser contract — is invisible to a user-only hash.
This test ensures the hash function in ``prompts/__init__.py`` covers all
three fields so the lockfile catches such drift.
"""

from __future__ import annotations

from bridge_steering.prompts import PromptSpec


def test_different_system_changes_hash():
    a = PromptSpec(id="a", system="X", user_template="u")
    b = PromptSpec(id="b", system="Y", user_template="u")
    assert a.sha256() != b.sha256()


def test_different_parser_contract_changes_hash():
    a = PromptSpec(id="a", user_template="u", parser_contract="P1")
    b = PromptSpec(id="b", user_template="u", parser_contract="P2")
    assert a.sha256() != b.sha256()


def test_different_user_template_changes_hash():
    a = PromptSpec(id="a", user_template="u1")
    b = PromptSpec(id="b", user_template="u2")
    assert a.sha256() != b.sha256()


def test_id_alone_does_not_change_hash():
    """The id is metadata for selection; only template content drives the hash."""
    a = PromptSpec(id="a", user_template="u")
    b = PromptSpec(id="b", user_template="u")
    assert a.sha256() == b.sha256()


def test_moving_text_between_system_and_user_changes_hash():
    a = PromptSpec(id="a", system="hello", user_template="world")
    b = PromptSpec(id="b", system="hello world", user_template="")
    assert a.sha256() != b.sha256(), (
        "Hash must catch text moving from user to system (or vice versa)."
    )
