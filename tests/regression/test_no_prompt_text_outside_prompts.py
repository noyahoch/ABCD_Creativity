"""Regression test: prompt text and PromptSpec construction live in prompts/ only.

Catches the original bug #2 — ``run_vllm_judge.py`` redefined the judge
prompt locally instead of importing the shared one.

Two checks:

(a) Hard-fail on any ``PromptSpec(...)`` constructed outside
    ``bridge_steering/prompts/``.
(b) Hard-fail on string literals containing strong prompt markers in
    ``bridge_steering/`` (excluding ``prompts/`` and ``tests/``). The marker
    set is intentionally small to avoid false positives; an ``allowlist.txt``
    lets us whitelist known exceptions.

Field-name tokens like ``"novelty"`` and ``"usability"`` are NOT in the
marker set — they appear in column names, plots, and summaries.
"""

from __future__ import annotations

import ast
from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parents[2] / "bridge_steering"
PROMPTS_DIR = PKG_ROOT / "prompts"
TESTS_DIR = PKG_ROOT / "tests"
ALLOWLIST = Path(__file__).with_name("allowlist.txt")


_PROMPT_MARKERS = (
    "Output ONLY",
    "Return JSON",
    "USE:",
    "You are an expert",
)


def _scan_files() -> list[Path]:
    out: list[Path] = []
    for f in PKG_ROOT.rglob("*.py"):
        if PROMPTS_DIR in f.parents or f == PROMPTS_DIR:
            continue
        if TESTS_DIR in f.parents or f == TESTS_DIR:
            continue
        out.append(f)
    return out


def _load_allowlist() -> set[str]:
    if not ALLOWLIST.is_file():
        return set()
    return {
        line.strip()
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def test_no_promptspec_constructor_outside_prompts():
    offenders: list[str] = []
    for f in _scan_files():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name == "PromptSpec":
                    rel = f.relative_to(PKG_ROOT)
                    offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "PromptSpec(...) must be constructed only in bridge_steering/prompts/. Offenders:\n  "
        + "\n  ".join(offenders)
    )


def test_no_prompt_markers_in_string_literals():
    allow = _load_allowlist()
    offenders: list[str] = []
    for f in _scan_files():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for mk in _PROMPT_MARKERS:
                    if mk in node.value:
                        rel = f.relative_to(PKG_ROOT)
                        key = f"{rel}:{node.lineno}:{mk}"
                        if key in allow:
                            continue
                        offenders.append(key)
    assert not offenders, (
        "Prompt-marker string literals outside bridge_steering/prompts/. "
        "Either move the prompt to prompts/ or add an entry to "
        f"{ALLOWLIST.name} (one `relative/path.py:line:marker` per line):\n  "
        + "\n  ".join(offenders)
    )
