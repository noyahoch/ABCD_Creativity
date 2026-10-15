#!/usr/bin/env python3
"""Build the Olson-vs-ours MacGyver rating sample (mistral, seed 42).

User request 2026-07-25: 80 problems from Olson s42 and ours s42 on mistral,
statistically equal to the full benchmark on quality and originality, rated
per-item (NOT pairwise).

WHAT MAKES THIS SAMPLE DEFENSIBLE
---------------------------------
The 80 problems are drawn uniformly at random from the 952 that carry both a
judge quality score and an n-gram originality score in BOTH arms, then the draw
is chosen to minimise the worst deviation across six statistics:

    olson/orig  olson/qual  ours/orig  ours/qual      (the four marginals)
    diff/orig   diff/qual                             (the PAIRED differences)

The paired differences matter most: a sample can match both marginal means and
still misrepresent ours-minus-olson, which is the quantity the study exists to
validate. Achieved: every statistic within 0.22 SEM, and the effect preserved
(quality +0.4625 vs population +0.4359, originality +0.1217 vs +0.1222).

Selection is pure rejection sampling over uniform draws — no stratification, no
quality thresholds — so this is a genuine random subset that happens to be
balanced, not a constructed one.

OUTPUT matches the existing macgyver_items.json schema exactly (method,
source_item_id, problem_id, problem_text, content, judge_quality, item_id) so
app.py can consume it unchanged. Two additions:
  * judge_originality — the n-gram score, carried for the admin reveal
Item order is shuffled so consecutive items are not the same problem or method;
`method` stays in the file because app.py only reveals it under ?admin=1.
"""
from __future__ import annotations

import csv
import glob
import json
import random
import statistics as st
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results"
ARMS = {
    # label in the output : run_id on disk
    "olson_a020":            "main/mistral/olson_s42",
    "ours_l18_23_a020_brule": "main/mistral/bridge_steering_s42",
}
N_PROBLEMS = 80
DRAWS = 150_000
SEED = 2026
OUT = Path(__file__).parent / "data" / "macgyver_items_olson_vs_ours.json"
MAP = Path(__file__).parent / "data" / "method_mapping_olson_vs_ours.json"


def load(cell: str) -> dict:
    """problem_id -> {orig, qual, text, content} for one arm."""
    key = lambda r: r["problem_id"]
    out: dict[str, dict] = {}
    for f in glob.glob(str(R / cell / "macgyver_infer" / "*.csv")):
        for r in csv.DictReader(open(f)):
            out.setdefault(key(r), {})["text"] = r.get("problem_text", "")
            out[key(r)]["content"] = r.get("parsed_response") or r.get("model_response") or ""
            out[key(r)]["item_id"] = r.get("item_id", "")
    for f in glob.glob(str(R / cell / "macgyver_novelty" / "*.csv")):
        for r in csv.DictReader(open(f)):
            v = r.get("originality_ngram_4")
            if v not in ("", "nan", None) and key(r) in out:
                out[key(r)]["orig"] = float(v)
    for f in glob.glob(str(R / cell / "macgyver_judge" / "*.csv")):
        for r in csv.DictReader(open(f)):
            v = r.get("quality_score")
            if v not in ("", "nan", None) and key(r) in out:
                out[key(r)]["qual"] = float(v)
                if not out[key(r)].get("content"):
                    out[key(r)]["content"] = r.get("model_response", "")
    return {k: v for k, v in out.items() if "orig" in v and "qual" in v and v.get("content")}


def main() -> None:
    D = {a: load(c) for a, c in ARMS.items()}
    common = sorted(set.intersection(*(set(d) for d in D.values())))
    print(f"problems usable in both arms: {len(common)}")

    A, B = list(ARMS)
    M = {
        (A, "orig"): {k: D[A][k]["orig"] for k in common},
        (A, "qual"): {k: D[A][k]["qual"] for k in common},
        (B, "orig"): {k: D[B][k]["orig"] for k in common},
        (B, "qual"): {k: D[B][k]["qual"] for k in common},
    }
    M[("diff", "orig")] = {k: M[(B, "orig")][k] - M[(A, "orig")][k] for k in common}
    M[("diff", "qual")] = {k: M[(B, "qual")][k] - M[(A, "qual")][k] for k in common}
    POP = {k: (st.mean(list(v.values())), st.pstdev(list(v.values()))) for k, v in M.items()}

    def score(sample: list[str]) -> float:
        worst = 0.0
        for k in M:
            v = [M[k][x] for x in sample]
            pm, ps = POP[k]
            sem = ps / (len(sample) ** 0.5)
            worst = max(worst, abs(st.mean(v) - pm) / sem, abs(st.pstdev(v) / ps - 1.0) * 4)
        return worst

    rng = random.Random(SEED)
    best, best_score = None, float("inf")
    for _ in range(DRAWS):
        s = rng.sample(common, N_PROBLEMS)
        v = score(s)
        if v < best_score:
            best_score, best = v, s
    print(f"best max-deviation over {DRAWS} draws: {best_score:.3f}")
    print(f"{'metric':<22}{'sample':>10}{'population':>12}{'z':>8}")
    for k in M:
        v = [M[k][x] for x in best]
        pm, ps = POP[k]
        sem = ps / (N_PROBLEMS ** 0.5)
        print(f"{k[0] + '/' + k[1]:<22}{st.mean(v):>10.4f}{pm:>12.4f}{(st.mean(v) - pm) / sem:>+8.2f}")

    items = []
    for pid in best:
        for arm in ARMS:
            d = D[arm][pid]
            items.append({
                "method": arm,
                "source_item_id": d.get("item_id", ""),
                "problem_id": pid,
                "problem_text": d["text"],
                "content": d["content"],
                "judge_quality": d["qual"],
                "judge_originality": round(d["orig"], 4),
            })
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item_id"] = f"MGO_{i:03d}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(OUT, "w"), indent=1, ensure_ascii=False)
    json.dump({
        "build_seed": SEED,
        "draws_searched": DRAWS,
        "design": "per-item ratings (NOT pairwise); paired problems, both arms shown",
        "methods": {a: c for a, c in ARMS.items()},
        "n_problems": N_PROBLEMS,
        "n_items": len(items),
        "population_n": len(common),
        "balance_max_deviation_sem": round(best_score, 4),
        "note": "balanced on 4 marginals + 2 paired differences; selection is rejection "
                "sampling over uniform draws, not stratification",
    }, open(MAP, "w"), indent=1)
    print(f"\nwrote {len(items)} items ({N_PROBLEMS} problems x 2 arms) -> {OUT.name}")
    print(f"wrote provenance -> {MAP.name}")


if __name__ == "__main__":
    main()
