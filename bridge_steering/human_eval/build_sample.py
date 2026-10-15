"""Build a fixed sample of items for the human rating study.

Paired design (same source instances across methods):

  AUT
    1. Pick 15 objects from the AUT eval holdout (the 40 objects in the
       inference CSVs, all 3 methods cover the same 40 because they share
       eval_seed=7, eval_holdout=40).
    2. For each (object, method), randomly pick ONE of the 8 generated uses.
    3. Result: 15 objects × 3 methods = 45 items.

  MacGyver
    1. Pick 15 problem_ids from the solvable_unconventional eval set (all
       three methods cover the same 956 problems).
    2. For each (problem, method), take the parsed_response (one solution
       per row in this configuration).
    3. Result: 15 problems × 3 methods = 45 items.

Each sampled item is annotated with the LLM judge scores from the same run.

Outputs:
  data/aut_items.json
  data/macgyver_items.json
  data/method_mapping.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
AUT_SOURCE_DATASET = ROOT / "data" / "aut_bridgeset.json"


def _run_dir(name: str) -> Path:
    return RESULTS / name


AUT_SOURCES = {
    "baseline": {
        "infer": _run_dir("main/mistral/baseline_s42")
        / "aut_infer/baseline_results.csv",
        "judge": _run_dir("main/mistral/baseline_s42")
        / "aut_judge/baseline_judge_scores.csv",
    },
    "baseline_t15": {
        "infer": _run_dir("main/mistral/baseline_high_temp_s42")
        / "aut_infer/baseline_results.csv",
        "judge": _run_dir("main/mistral/baseline_high_temp_s42")
        / "aut_judge/baseline_judge_scores.csv",
    },
    "ours_l18_23_a017": {
        "infer": _run_dir("judge_robustness/mistral/structured_bridges_a017_s42")
        / "aut_infer/steered_L18_L19_L20_L21_L22_L23_a0.17_results.csv",
        "judge": _run_dir("judge_robustness/mistral/structured_bridges_a017_s42")
        / "aut_judge/steered_L18_L19_L20_L21_L22_L23_a0.17_judge_scores.csv",
    },
}

MACGYVER_SOURCES = {
    "baseline": {
        "infer": _run_dir("main/mistral/baseline_s42")
        / "macgyver_infer/baseline_results.csv",
        "judge": _run_dir("main/mistral/baseline_s42")
        / "macgyver_judge/baseline_judge_scores.csv",
    },
    "creative_minp005_t10": {
        "infer": _run_dir("main/mistral/creative_decoding_1_s42")
        / "macgyver_infer/baseline_results.csv",
        "judge": _run_dir("main/mistral/creative_decoding_1_s42")
        / "macgyver_judge/baseline_judge_scores.csv",
    },
    "ours_l18_23_a017": {
        "infer": _run_dir("judge_robustness/mistral/structured_bridges_a017_s42")
        / "macgyver_infer/steered_L18_L19_L20_L21_L22_L23_a0.17_results.csv",
        "judge": _run_dir("judge_robustness/mistral/structured_bridges_a017_s42")
        / "macgyver_judge/steered_L18_L19_L20_L21_L22_L23_a0.17_judge_scores.csv",
    },
}

N_OBJECTS = 15   # AUT: 15 objects × 3 methods = 45 items
N_PROBLEMS = 10  # MacGyver: 10 problems × 3 methods = 30 items
MG_MAX_PROBLEM_LEN = 350  # only sample problems whose problem_text is ≤ this many chars


def _aut_judge_index(judge_csv: Path) -> dict[str, dict]:
    judge = pd.read_csv(judge_csv)
    return {
        str(r["use_text"]).strip(): {
            "judge_novelty": int(r["novelty"]) if pd.notna(r["novelty"]) else None,
            "judge_usability": int(r["usability"]) if pd.notna(r["usability"]) else None,
            "judge_overall": float(r["overall"]) if pd.notna(r["overall"]) else None,
        }
        for _, r in judge.iterrows()
    }


def _mg_judge_index(judge_csv: Path) -> dict:
    judge = pd.read_csv(judge_csv)
    return {
        r["item_id"]: float(r["quality_score"])
        for _, r in judge.iterrows()
        if pd.notna(r["quality_score"])
    }


def _sample_aut(seed: int) -> list[dict]:
    rng = random.Random(seed)

    # Load all 3 inference CSVs once.
    infer_by_method: dict[str, pd.DataFrame] = {
        m: pd.read_csv(p["infer"]) for m, p in AUT_SOURCES.items()
    }
    judge_by_method: dict[str, dict] = {
        m: _aut_judge_index(p["judge"]) for m, p in AUT_SOURCES.items()
    }
    # Common object set (should be identical 40 across all methods, but
    # take intersection defensively).
    common_objects: list[str] = sorted(
        set.intersection(*[set(df["object"].unique()) for df in infer_by_method.values()])
    )

    # Sanity: every common object should also appear in the source dataset.
    src = json.loads(AUT_SOURCE_DATASET.read_text())
    src_objects = {e.get("A_OBJECT") for e in src}
    missing_from_src = [o for o in common_objects if o not in src_objects]
    if missing_from_src:
        raise RuntimeError(
            f"Eval holdout objects missing from source AUT dataset: {missing_from_src}"
        )

    if len(common_objects) < N_OBJECTS:
        raise RuntimeError(
            f"Only {len(common_objects)} common AUT objects, need {N_OBJECTS}"
        )

    chosen_objects = rng.sample(common_objects, N_OBJECTS)
    chosen_objects.sort()  # deterministic order before the per-method loop
    print(f"AUT chose {N_OBJECTS} objects: {chosen_objects}")

    items: list[dict] = []
    for method in AUT_SOURCES:
        df = infer_by_method[method]
        judge_idx = judge_by_method[method]
        df_by_obj = {r["object"]: r for _, r in df.iterrows()}
        for obj in chosen_objects:
            row = df_by_obj.get(obj)
            if row is None:
                raise RuntimeError(f"AUT: object {obj!r} missing in method {method}")
            try:
                uses_raw = json.loads(row["parsed_response"])
            except (TypeError, ValueError) as e:
                raise RuntimeError(f"bad parsed_response for {obj}/{method}: {e}")
            uses = [u.strip() for u in uses_raw if isinstance(u, str) and u.strip()]
            if not uses:
                raise RuntimeError(f"no parsed uses for {obj}/{method}")
            use_idx = rng.randrange(len(uses))
            use_text = uses[use_idx]
            j = judge_idx.get(use_text, {})
            items.append(
                {
                    "method": method,
                    "source_item_id": row["item_id"],
                    "object": obj,
                    "use_idx": use_idx,
                    "content": use_text,
                    "judge_novelty": j.get("judge_novelty"),
                    "judge_usability": j.get("judge_usability"),
                    "judge_overall": j.get("judge_overall"),
                }
            )
    return items


def _sample_macgyver(seed: int) -> list[dict]:
    rng = random.Random(seed)
    infer_by_method: dict[str, pd.DataFrame] = {
        m: pd.read_csv(p["infer"]) for m, p in MACGYVER_SOURCES.items()
    }
    judge_by_method: dict[str, dict] = {
        m: _mg_judge_index(p["judge"]) for m, p in MACGYVER_SOURCES.items()
    }
    # The inference subset is solvable_unconventional (per the configs); each
    # row carries solvable=True, unconventional=True. Take the intersection of
    # problem_ids across methods (sanity — should match all 3).
    common_pids: list = sorted(
        set.intersection(*[set(df["problem_id"].unique()) for df in infer_by_method.values()])
    )
    # Filter to "relatively short" problems so raters aren't overwhelmed.
    # Use the baseline run's problem_text (identical text across methods).
    baseline_df = infer_by_method["baseline"]
    pid_to_len = {
        r["problem_id"]: len(str(r["problem_text"])) for _, r in baseline_df.iterrows()
    }
    short_pids = [p for p in common_pids if pid_to_len.get(p, 10**9) <= MG_MAX_PROBLEM_LEN]
    print(
        f"MacGyver: {len(short_pids)}/{len(common_pids)} problems are ≤ "
        f"{MG_MAX_PROBLEM_LEN} chars; sampling {N_PROBLEMS} from those."
    )
    if len(short_pids) < N_PROBLEMS:
        raise RuntimeError(
            f"Only {len(short_pids)} short MacGyver problems (≤ {MG_MAX_PROBLEM_LEN} "
            f"chars), need {N_PROBLEMS}. Raise MG_MAX_PROBLEM_LEN."
        )
    common_pids = short_pids

    chosen_pids = rng.sample(common_pids, N_PROBLEMS)
    chosen_pids.sort()
    print(f"MacGyver chose {N_PROBLEMS} problems: {chosen_pids}")

    items: list[dict] = []
    for method in MACGYVER_SOURCES:
        df = infer_by_method[method]
        judge_idx = judge_by_method[method]
        # Verify all chosen pids are solvable & unconventional in this run.
        df_by_pid = {r["problem_id"]: r for _, r in df.iterrows()}
        for pid in chosen_pids:
            row = df_by_pid.get(pid)
            if row is None:
                raise RuntimeError(f"MG: problem {pid} missing in method {method}")
            if not (bool(row.get("solvable", True)) and bool(row.get("unconventional", True))):
                raise RuntimeError(
                    f"MG: problem {pid} not solvable_unconventional in {method}"
                )
            resp = row.get("parsed_response")
            if not isinstance(resp, str) or not resp.strip():
                raise RuntimeError(f"MG: empty parsed_response {pid}/{method}")
            items.append(
                {
                    "method": method,
                    "source_item_id": row["item_id"],
                    "problem_id": int(pid),
                    "problem_text": row["problem_text"],
                    "content": resp.strip(),
                    "judge_quality": judge_idx.get(row["item_id"]),
                }
            )
    return items


def build(seed: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_master = random.Random(seed)

    aut_items = _sample_aut(rng_master.randint(0, 2**31 - 1))
    macgyver_items = _sample_macgyver(rng_master.randint(0, 2**31 - 1))

    # Shuffle the combined 45-item pool so adjacent items aren't grouped
    # by method or object.
    rng_master.shuffle(aut_items)
    rng_master.shuffle(macgyver_items)
    for i, item in enumerate(aut_items):
        item["item_id"] = f"AUT_{i:03d}"
    for i, item in enumerate(macgyver_items):
        item["item_id"] = f"MG_{i:03d}"

    (out_dir / "aut_items.json").write_text(
        json.dumps(aut_items, ensure_ascii=False, indent=2)
    )
    (out_dir / "macgyver_items.json").write_text(
        json.dumps(macgyver_items, ensure_ascii=False, indent=2)
    )

    mapping = {
        "build_seed": seed,
        "design": "paired (same source instances across methods)",
        "aut_methods": list(AUT_SOURCES.keys()),
        "macgyver_methods": list(MACGYVER_SOURCES.keys()),
        "aut_n_objects": N_OBJECTS,
        "macgyver_n_problems": N_PROBLEMS,
        "aut_n_items": len(aut_items),
        "macgyver_n_items": len(macgyver_items),
    }
    (out_dir / "method_mapping.json").write_text(json.dumps(mapping, indent=2))

    print(
        f"\nWrote {len(aut_items)} AUT items and {len(macgyver_items)} MacGyver items "
        f"to {out_dir}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    args = ap.parse_args()
    build(args.seed, args.out_dir)


if __name__ == "__main__":
    main()
