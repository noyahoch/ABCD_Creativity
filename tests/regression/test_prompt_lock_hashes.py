"""Regression test: prompt_lock.json must match the live PromptSpec hashes.

If this test fails, either:

1. You edited a prompt's text and need to revert it, OR
2. You intentionally bumped the prompt — in that case give it a NEW id
   (e.g. ``_v3`` -> ``_v4``), move the old spec to
   ``bridge_steering/prompts/_archive.py`` with ``archived=True``, and add a new
   line to ``bridge_steering/prompts/prompt_lock.json``.

Editing the lockfile in place without those steps defeats the safeguard.
"""

from __future__ import annotations

from bridge_steering.prompts import all_specs, build_lockfile_dict, load_lockfile


def test_lockfile_present_and_complete():
    lock = load_lockfile()
    specs = all_specs()
    missing = set(specs) - set(lock)
    extra = set(lock) - set(specs)
    assert not missing, (
        f"prompt_lock.json is missing these PromptSpec ids: {sorted(missing)}\n"
        f"Add them with their current sha256 (run bridge_steering.prompts.build_lockfile_dict())."
    )
    assert not extra, (
        f"prompt_lock.json has ids that no longer exist in prompts/: {sorted(extra)}\n"
        f"Either restore the spec (move from _archive.py if needed) or remove the lockfile line."
    )


def test_each_spec_hash_matches_lockfile():
    lock = load_lockfile()
    expected = build_lockfile_dict()
    drift: list[str] = []
    for pid, want in expected.items():
        got = lock.get(pid)
        if got != want:
            drift.append(f"  {pid}: lock={got!r} live={want!r}")
    assert not drift, (
        "Prompt text drift detected — these PromptSpec sha256s differ from prompt_lock.json:\n"
        + "\n".join(drift)
        + "\nEither revert the edit OR bump the prompt id and archive the old version."
    )
