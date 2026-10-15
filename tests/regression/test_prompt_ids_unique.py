"""Regression test: no two PromptSpecs share an id.

Catches the original bug #1 — ``build_per_use_prompt`` was defined twice
in the same file, second silently shadowed the first.
"""

from __future__ import annotations

import ast
from pathlib import Path

from bridge_steering.prompts import PromptSpec, all_specs


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "bridge_steering" / "prompts"


def test_no_duplicate_prompt_ids_at_runtime():
    """all_specs() raises on duplicates, but make the failure mode explicit."""
    specs = all_specs()
    ids = [s.id for s in specs.values()]
    assert len(ids) == len(set(ids)), f"Duplicate ids: {ids}"


def _ast_find_promptspec_calls(file: Path) -> list[tuple[str, int]]:
    """Return ``(prompt_id, lineno)`` for every PromptSpec(...) at module level."""
    tree = ast.parse(file.read_text(encoding="utf-8"))
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        # Look for Call(PromptSpec, ...) with an explicit id= keyword
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name != "PromptSpec":
                continue
            for kw in node.keywords:
                if kw.arg == "id" and isinstance(kw.value, ast.Constant):
                    out.append((kw.value.value, node.lineno))
    return out


def test_ast_walk_finds_no_duplicate_ids():
    """Belt-and-suspenders: scan the source files, not just the registry."""
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(PROMPTS_DIR.glob("*.py")):
        for pid, line in _ast_find_promptspec_calls(f):
            assert pid not in seen, (
                f"PromptSpec id {pid!r} duplicated: "
                f"{seen[pid][0].name}:{seen[pid][1]} and {f.name}:{line}"
            )
            seen[pid] = (f, line)
