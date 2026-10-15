"""Translate the LLM-generated content in the sample JSONs to Hebrew.

Idempotent: only translates fields that are missing or empty in the JSON.
Re-run safely after editing build_sample.py.

Usage:
    pip install deep-translator
    python bridge_steering/human_eval/translate_sample.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator

DATA_DIR = Path(__file__).parent / "data"
SOURCE_LANG = "en"
TARGET_LANG = "iw"  # Hebrew (ISO-639-1 'he' also works in deep-translator)
MAX_CHUNK = 4500   # Google's per-call limit is ~5000 chars; stay under it.


def _translate_text(translator: GoogleTranslator, text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= MAX_CHUNK:
        return translator.translate(text)
    # Split on paragraph boundaries to keep formatting reasonable.
    out_chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in text.split("\n"):
        line_len = len(line) + 1
        if buf_len + line_len > MAX_CHUNK and buf:
            out_chunks.append(translator.translate("\n".join(buf)))
            buf, buf_len = [], 0
        buf.append(line)
        buf_len += line_len
    if buf:
        out_chunks.append(translator.translate("\n".join(buf)))
    return "\n".join(out_chunks)


def _translate_field(translator, item: dict, src: str, dst: str, label: str) -> bool:
    if item.get(dst):  # already translated
        return False
    value = item.get(src)
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        item[dst] = _translate_text(translator, value)
    except Exception as e:
        print(f"  ! failed {label} {src!r}: {e}", file=sys.stderr)
        return False
    return True


def translate_file(path: Path, fields: list[tuple[str, str]]) -> None:
    items = json.loads(path.read_text())
    translator = GoogleTranslator(source=SOURCE_LANG, target=TARGET_LANG)
    changed = 0
    for i, item in enumerate(items, 1):
        item_label = item.get("item_id", f"#{i}")
        for src, dst in fields:
            if _translate_field(translator, item, src, dst, item_label):
                changed += 1
                # Light politeness sleep to avoid rate-limiting.
                time.sleep(0.1)
        if i % 10 == 0:
            print(f"  {path.name}: processed {i}/{len(items)}")
    if changed:
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2))
        print(f"  {path.name}: wrote {changed} new translations")
    else:
        print(f"  {path.name}: nothing to do (all translated)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    print(f"Target language: {TARGET_LANG} (Hebrew)")
    aut_path = args.data_dir / "aut_items.json"
    mg_path = args.data_dir / "macgyver_items.json"

    if aut_path.exists():
        print(f"\n→ {aut_path.name}")
        translate_file(
            aut_path,
            fields=[("object", "object_he"), ("content", "content_he")],
        )
    if mg_path.exists():
        print(f"\n→ {mg_path.name}")
        translate_file(
            mg_path,
            fields=[
                ("problem_text", "problem_text_he"),
                ("content", "content_he"),
            ],
        )


if __name__ == "__main__":
    main()
