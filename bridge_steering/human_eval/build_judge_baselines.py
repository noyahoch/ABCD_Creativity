"""Pre-bake judge scores per display item into data/judge_baselines.json.

The deployed app can't reach into `results/...`
(that path is gitignored and not part of the cloud clone). Instead, we
pre-compute the judge novelty/usability/quality + ngram-4 per (method,
item_id-in-sample) and ship it alongside the items JSONs.

Run this once locally (or whenever the sample is rebuilt):
    python bridge_steering/human_eval/build_judge_baselines.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
REPO = HERE.parents[1]
REB_DIR = REPO / "results"

AUT_RUN = {
    "baseline":         ("main/mistral/baseline_s42",
                         "aut_judge/baseline_judge_scores.csv"),
    "baseline_t15":     ("main/mistral/baseline_high_temp_s42",
                         "aut_judge/baseline_judge_scores.csv"),
    "ours_l18_23_a017": ("judge_robustness/mistral/structured_bridges_a017_s42",
                         "aut_judge/steered_L18_L19_L20_L21_L22_L23_a0.17_judge_scores.csv"),
}
MG_JUDGE_RUN = {
    "baseline":              ("main/mistral/baseline_s42",
                              "macgyver_judge/baseline_judge_scores.csv"),
    "creative_minp005_t10":  ("main/mistral/creative_decoding_1_s42",
                              "macgyver_judge/baseline_judge_scores.csv"),
    "ours_l18_23_a017":      ("judge_robustness/mistral/structured_bridges_a017_s42",
                              "macgyver_judge/steered_L18_L19_L20_L21_L22_L23_a0.17_judge_scores.csv"),
}
MG_NGRAM_RUN = {
    "baseline":              ("main/mistral/baseline_s42",
                              "macgyver_novelty/baseline_novelty.csv"),
    "creative_minp005_t10":  ("main/mistral/creative_decoding_1_s42",
                              "macgyver_novelty/baseline_novelty.csv"),
    "ours_l18_23_a017":      ("judge_robustness/mistral/structured_bridges_a017_s42",
                              "macgyver_novelty/steered_L18_L19_L20_L21_L22_L23_a0.17_novelty.csv"),
}


def _aut_judge_by_use_text() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for method, (run, suffix) in AUT_RUN.items():
        path = REB_DIR / run / suffix
        if not path.exists():
            print(f"  ! missing {path}")
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                key = (method, str(r.get("use_text", "")).strip())
                try:
                    out[key] = {
                        "nov": int(r["novelty"]),
                        "use": int(r["usability"]),
                    }
                except (TypeError, ValueError, KeyError):
                    pass
    return out


def _mg_lookup(run_map: dict, value_keys: dict[str, str]) -> dict[tuple[str, str], dict]:
    """Generic loader for MG side data — keyed by (method, item_id)."""
    out: dict[tuple[str, str], dict] = {}
    for method, (run, suffix) in run_map.items():
        path = REB_DIR / run / suffix
        if not path.exists():
            print(f"  ! missing {path}")
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                key = (method, str(r.get("item_id", "")).strip())
                vals: dict = {}
                for out_key, in_key in value_keys.items():
                    try:
                        vals[out_key] = float(r[in_key])
                    except (TypeError, ValueError, KeyError):
                        pass
                if vals:
                    out[key] = vals
    return out


def main() -> None:
    print("AUT — joining judge scores by (method, use_text) ...")
    aut_judge = _aut_judge_by_use_text()
    aut_items = json.loads((DATA_DIR / "aut_items.json").read_text())
    aut_baselines: dict[str, dict] = {}
    misses = 0
    for it in aut_items:
        key = (it["method"], it["content"].strip())
        jv = aut_judge.get(key)
        if jv is None:
            misses += 1
            continue
        aut_baselines[it["item_id"]] = {
            "method": it["method"],
            "judge_nov": jv["nov"],
            "judge_use": jv["use"],
        }
    print(f"  matched {len(aut_baselines)}/{len(aut_items)} (misses: {misses})")

    print("MG — joining judge quality + ngram-4 by (method, source_item_id) ...")
    mg_judge = _mg_lookup(MG_JUDGE_RUN, {"q": "quality_score"})
    mg_ngram = _mg_lookup(MG_NGRAM_RUN, {"ngram4": "originality_ngram_4"})
    mg_items = json.loads((DATA_DIR / "macgyver_items.json").read_text())
    mg_baselines: dict[str, dict] = {}
    misses = 0
    for it in mg_items:
        key = (it["method"], str(it["source_item_id"]))
        jq = mg_judge.get(key, {}).get("q")
        ng = mg_ngram.get(key, {}).get("ngram4")
        if jq is None or ng is None:
            misses += 1
            continue
        mg_baselines[it["item_id"]] = {
            "method": it["method"],
            "judge_quality": jq,
            "ngram_4": ng,
        }
    print(f"  matched {len(mg_baselines)}/{len(mg_items)} (misses: {misses})")

    out = {"aut": aut_baselines, "macgyver": mg_baselines}
    path = DATA_DIR / "judge_baselines.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
