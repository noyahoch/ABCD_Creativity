"""AUT-BridgeSet bridge audit: LLM screen of the intermediate-construct bridges.

For all 150 items, take the "Intermediate construct" family bridge (its ``rule`` text) and ask
Qwen2.5-32B-Instruct (the paper's AWQ judge build, served locally by vLLM) whether the bridge
names a concrete object / device / mechanism that is itself a usable answer (leaked answer)
rather than a reframing that opens a family of solutions (valid bridge). Rubric + the paper's
own two examples. Deterministic decoding (T=0, seed 42). Dataset is read only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys

from .registry import OUT_DIR, REPO

DATASET = REPO / "data" / "aut_bridgeset.json"
FAMILY = "Intermediate construct"
MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"

RUBRIC = """You audit "bridges" from a creativity dataset (Alternative Uses Task). A bridge is an intermediate re-representation of an object that opens a FAMILY of unconventional solutions; it must not itself be a specific usable answer.

Criterion: a bridge OPENS a family of solutions; an answer INSTANTIATES one.
Flag the bridge as a leaked answer if it names a concrete object, device, assembly or specific mechanism that is itself directly usable as an answer to "give unconventional uses for <object>" (e.g. it already says what to build or do with which extra parts). Keep it as a valid bridge if it reframes the object in terms of functional primitives, variables, roles or abstract mechanisms that many different answers could instantiate.

Examples from the paper:
- Object: brick. Bridge: "functional primitives: weight, thermal mass, rough surface" -> valid_bridge (it reframes; many uses follow).
- Object: bottle. Bridge: "wick-drip mechanism using a bottle and cloth strip" -> leaked_answer (it is a specific device, i.e. one answer).

Answer with ONE JSON object only: {"flag": "valid_bridge" | "leaked_answer", "reason": "<= 25 words"}"""


def build_messages(obj: str, instruction: str, bridge: str) -> list[dict]:
    user = (f"Object: {obj}\nTask prompt: {instruction}\nBridge (intermediate-construct family): \"{bridge}\"\n\n"
            "Does this bridge name a concrete object/device/mechanism that is itself a usable answer "
            "(leaked_answer), or does it reframe the object to open a family of solutions (valid_bridge)?")
    return [{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}]


def parse(text: str) -> tuple[str, str]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return "unparsed", text.strip()[:200]
    try:
        d = json.loads(m.group(0))
        flag = str(d.get("flag", "")).strip()
        return (flag if flag in ("valid_bridge", "leaked_answer") else "unparsed"), str(d.get("reason", "")).strip()
    except Exception:
        return "unparsed", text.strip()[:200]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8002/v1")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args(argv)
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key="EMPTY")
    items = json.load(open(DATASET))
    rows = []
    for it in items:
        b = [b for b in it["B_list"] if b["type"] == FAMILY]
        assert len(b) == 1, it["id"]
        bridge = b[0]["rule"]
        resp = client.chat.completions.create(model=a.model, messages=build_messages(it["A_OBJECT"], it["A"], bridge),
                                              temperature=0.0, max_tokens=200, seed=42)
        text = resp.choices[0].message.content or ""
        flag, reason = parse(text)
        rows.append(dict(item=it["id"], object=it["A_OBJECT"], bridge=bridge, unlocks=b[0]["unlocks"],
                         flag=flag, reason=reason))
        print(f"{it['id']:12s} {it['A_OBJECT']:22s} {flag:14s} {reason[:80]}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "bridge_audit.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    counts = {k: sum(r["flag"] == k for r in rows) for k in ("valid_bridge", "leaked_answer", "unparsed")}
    summary = (f"# AUT-BridgeSet bridge audit (intermediate-construct family, {len(rows)} items)\n\n"
               f"Judge: {a.model} via local vLLM, T=0, seed 42. Criterion: a bridge opens a family of solutions; "
               f"an answer instantiates one. Judged text = the bridge's `rule` field.\n\n"
               f"| flag | count |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in counts.items())
               + f"\nFlagged bridges:\n\n" + "".join(f"- {r['item']} ({r['object']}): \"{r['bridge']}\" — {r['reason']}\n"
                                                    for r in rows if r["flag"] == "leaked_answer"))
    (OUT_DIR / "bridge_audit_summary.md").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
