"""Translate sample content to Hebrew using Claude.

Drop-in replacement for translate_sample.py (Google Translate) — same JSON
input/output, better Hebrew, preserves markdown formatting.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

    # Sanity check: print 5 known-tricky fields WITHOUT writing to JSON
    python bridge_steering/human_eval/translate_with_claude.py --sanity

    # Full run, write back to JSON
    python bridge_steering/human_eval/translate_with_claude.py --force

    # Switch model
    python bridge_steering/human_eval/translate_with_claude.py --model claude-sonnet-4-5

Quality safeguards:
- Strict "TRANSLATE only, never extend / answer / complete" system prompt.
- Post-check: if a translation is more than 1.6× the source length, it's
  rejected and the script falls back to leaving the field unchanged.

Cost rough estimate:
    Haiku-4.5: $0.20 for ~150 fields.
    Sonnet-4.5: $0.50 for ~150 fields.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.stderr.write("install anthropic: pip install anthropic\n")
    raise

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MODEL = "claude-sonnet-4-5"
MAX_LENGTH_RATIO = 1.6  # reject if Hebrew is > 1.6x English chars

SYSTEM_PROMPT = (
    "You are translating short English text into Modern Hebrew for a "
    "human-rating study. This is a TRANSLATION TASK ONLY.\n\n"
    "**CRITICAL RULES**\n"
    "1. Translate ONLY what is given. Do NOT extend, expand, answer, "
    "complete, or add any content beyond what is in the input. If the "
    "input is a question, the output is the same question in Hebrew. If "
    "the input is one sentence, the output is one sentence.\n"
    "2. Output ONLY the Hebrew translation. No preamble, no commentary, "
    "no quotation marks wrapping the output, no English transliteration "
    "annotations in parentheses.\n"
    "3. Preserve markdown EXACTLY: keep **bold**, *italic*, bullets "
    "(-, •), numbered list prefixes (1., 2., 3.), and step/section "
    "headers ('Step N:', '## Section', etc.) as they appear in the input.\n"
    "4. Preserve line/paragraph structure: single newline stays single, "
    "blank-line paragraph break stays a blank-line paragraph break.\n\n"
    "**LEXICAL RULES**\n"
    "5. Translate common English nouns to their Hebrew equivalents, even "
    "when inside bolded titles. Examples:\n"
    "   - 'press' → 'מכבש' (NOT 'פרס' which means 'prize')\n"
    "   - 'medium' (as in growth substrate) → 'מצע'\n"
    "   - 'box' → 'קופסה'\n"
    "   - 'cutter' → 'חותך'\n"
    "6. Only transliterate when the source word is a proper noun, a "
    "brand name, or a genuinely made-up coinage with no Hebrew equivalent "
    "(e.g. 'Mimosa Press' → 'מכבש מימוזה' — translate 'Press', "
    "transliterate the made-up modifier 'Mimosa').\n"
    "7. Use natural, idiomatic modern Hebrew. Avoid word-for-word "
    "transliteration of English idioms. Use 2nd-person plural imperative "
    "for instructional steps where natural ('מרחו', 'הניחו', 'השתמשו').\n"
)


def translate_one(
    client: "anthropic.Anthropic", text: str, model: str
) -> tuple[str | None, str]:
    """Translate `text`. Returns (translation_or_None, status).
    If the output is empty or fails the length sanity check, returns (None, reason)."""
    text = text.strip()
    if not text:
        return ("", "empty")
    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    out = "".join(b.text for b in msg.content if getattr(b, "text", None)).strip()
    if not out:
        return (None, "empty_response")
    ratio = len(out) / max(len(text), 1)
    if ratio > MAX_LENGTH_RATIO:
        return (None, f"length_ratio={ratio:.2f} (>{MAX_LENGTH_RATIO}) likely hallucination")
    return (out, "ok")


def translate_file(
    client: "anthropic.Anthropic",
    path: Path,
    fields: list[tuple[str, str]],
    force: bool,
    model: str,
) -> int:
    items = json.loads(path.read_text())
    changed = 0
    rejected = 0
    for i, item in enumerate(items, 1):
        label = item.get("item_id", f"#{i}")
        for src, dst in fields:
            value = item.get(src)
            if not isinstance(value, str) or not value.strip():
                continue
            if item.get(dst) and not force:
                continue
            try:
                out, status = translate_one(client, value, model)
            except Exception as e:
                print(f"  ! {label}.{src}: {e}", file=sys.stderr)
                continue
            if out is None:
                print(f"  ! {label}.{src} rejected: {status}", file=sys.stderr)
                rejected += 1
            else:
                item[dst] = out
                changed += 1
            time.sleep(0.1)
        if i % 5 == 0:
            print(f"  {path.name}: processed {i}/{len(items)}")
    if changed:
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    print(
        f"  {path.name}: wrote {changed} translations, {rejected} rejected by safety check"
    )
    return changed


SANITY_TARGETS = [
    # (json_basename, item_id, src_field)
    ("aut_items.json",      "AUT_014", "content"),       # Mimosa Press
    ("aut_items.json",      "AUT_002", "content"),       # Crystals Growing Medium
    ("aut_items.json",      "AUT_000", "content"),       # Wi-Fi Signal Box (long, weird)
    ("macgyver_items.json", "MG_003",  "problem_text"),  # plant propagation (the hallucination case)
    ("macgyver_items.json", "MG_001",  "content"),       # multi-step solution (preserve structure)
]


def sanity_check(client: "anthropic.Anthropic", model: str) -> None:
    print(f"\n=== SANITY CHECK on model={model} (no JSON writes) ===\n")
    for fname, item_id, src in SANITY_TARGETS:
        items = json.loads((DATA_DIR / fname).read_text())
        item = next(it for it in items if it["item_id"] == item_id)
        en = item[src]
        prev_he = item.get(f"{src}_he", "(no prior translation)")
        print(f"--- {item_id} :: {src} ---")
        print(f"EN:\n{en}\n")
        print(f"Haiku-current HE:\n{prev_he}\n")
        out, status = translate_one(client, en, model)
        print(f"Sonnet HE [{status}]:\n{out}\n")
        print("-" * 70)
        time.sleep(0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Anthropic model (default: {DEFAULT_MODEL})")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing _he fields (default: only fill blanks)",
    )
    ap.add_argument(
        "--sanity",
        action="store_true",
        help="run a 5-item sanity check on known-tricky fields; print results, do NOT write JSON",
    )
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("set ANTHROPIC_API_KEY env var first")

    client = anthropic.Anthropic()
    print(f"Model: {args.model}")

    if args.sanity:
        sanity_check(client, args.model)
        return

    aut_path = args.data_dir / "aut_items.json"
    mg_path = args.data_dir / "macgyver_items.json"

    if aut_path.exists():
        print(f"\n→ {aut_path.name}")
        translate_file(
            client,
            aut_path,
            fields=[("object", "object_he"), ("content", "content_he")],
            force=args.force,
            model=args.model,
        )
    if mg_path.exists():
        print(f"\n→ {mg_path.name}")
        translate_file(
            client,
            mg_path,
            fields=[
                ("problem_text", "problem_text_he"),
                ("content", "content_he"),
            ],
            force=args.force,
            model=args.model,
        )


if __name__ == "__main__":
    main()
