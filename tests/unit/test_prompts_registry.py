"""Unit tests for the PromptSpec primitive."""

from __future__ import annotations

import pytest

from bridge_steering.prompts import PromptSpec, all_specs, resolve, load_lockfile


def test_render_returns_messages_list():
    spec = PromptSpec(id="t", system="sys", user_template="hi {name}")
    msgs = spec.render(name="ada")
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi ada"},
    ]


def test_render_skips_system_when_none():
    spec = PromptSpec(id="t", user_template="x")
    msgs = spec.render()
    assert msgs == [{"role": "user", "content": "x"}]
    assert "system" not in {m["role"] for m in msgs}


def test_sha256_deterministic_and_sensitive():
    a = PromptSpec(id="a", system="s", user_template="u", parser_contract="p")
    b = PromptSpec(id="b", system="s", user_template="u", parser_contract="p")
    assert a.sha256() == b.sha256()  # id is not hashed
    c = PromptSpec(id="c", system="s", user_template="u DIFFERENT", parser_contract="p")
    assert a.sha256() != c.sha256()


def test_manifest_entry_schema():
    spec = PromptSpec(id="x", user_template="hello")
    e = spec.manifest_entry()
    assert set(e) == {"id", "sha256", "archived", "text_preview"}
    assert e["sha256"].startswith("sha256:")
    assert e["archived"] is False


def test_all_specs_resolves_archive_and_live():
    specs = all_specs()
    assert "aut_user_object_only_v1" in specs
    assert "aut_judge_novelty_usability_v0_5pt" in specs  # archived
    assert specs["aut_judge_novelty_usability_v0_5pt"].archived is True


def test_resolve_blocks_archived_by_default():
    with pytest.raises(ValueError, match="archived"):
        resolve("aut_judge_novelty_usability_v0_5pt")
    spec = resolve("aut_judge_novelty_usability_v0_5pt", allow_archived=True)
    assert spec.archived


def test_resolve_unknown_id_raises():
    with pytest.raises(KeyError):
        resolve("does_not_exist")


def test_lockfile_is_present():
    lock = load_lockfile()
    assert "aut_user_object_only_v1" in lock
    assert lock["aut_user_object_only_v1"].startswith("sha256:")
