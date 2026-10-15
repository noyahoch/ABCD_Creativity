"""Audit that the human-eval sample uses ONLY the 6 Mistral runs
with the correct decoding settings, and that the paired design is intact.

Run after every build_sample.py:
    python bridge_steering/human_eval/validate_provenance.py
Exit code 0 = pass, 1 = at least one check failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

# Import the configured per-task sample sizes from build_sample so both stay
# in lock-step instead of hardcoding constants here.
import importlib.util as _ilu
_bs_spec = _ilu.spec_from_file_location("build_sample", Path(__file__).parent / "build_sample.py")
_bs = _ilu.module_from_spec(_bs_spec)
_bs_spec.loader.exec_module(_bs)
N_OBJECTS = _bs.N_OBJECTS
N_PROBLEMS = _bs.N_PROBLEMS
MG_MAX_PROBLEM_LEN = _bs.MG_MAX_PROBLEM_LEN

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
DATA = Path(__file__).parent / "data"

EXPECTED_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"

EXPECTED = {
    "aut": {
        "baseline":         dict(run="main/mistral/baseline_s42",        file="baseline_results.csv",                                temp=0.7, min_p=None, top_p=None, ours=False),
        "baseline_t15":     dict(run="main/mistral/baseline_high_temp_s42",    file="baseline_results.csv",                                temp=1.5, min_p=None, top_p=None, ours=False),
        "ours_l18_23_a017": dict(run="judge_robustness/mistral/structured_bridges_a017_s42",     file="steered_L18_L19_L20_L21_L22_L23_a0.17_results.csv",  temp=0.7, min_p=None, top_p=None, ours=True),
    },
    "macgyver": {
        "baseline":              dict(run="main/mistral/baseline_s42",             file="baseline_results.csv",                                temp=0.7, min_p=None, top_p=None, ours=False),
        "creative_minp005_t10":  dict(run="main/mistral/creative_decoding_1_s42", file="baseline_results.csv",                                temp=1.0, min_p=0.05, top_p=None, ours=False),
        "ours_l18_23_a017":      dict(run="judge_robustness/mistral/structured_bridges_a017_s42",          file="steered_L18_L19_L20_L21_L22_L23_a0.17_results.csv",  temp=0.7, min_p=None, top_p=None, ours=True),
    },
}

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ✓ {msg}")
    else:
        fails.append(msg)
        print(f"  ✗ {msg}")


def audit_run(task: str, method: str, exp: dict) -> None:
    print(f"\n=== {task} / {method} ===")
    infer_subdir = "aut_infer" if task == "aut" else "macgyver_infer"
    run_dir = RESULTS / exp["run"]
    csv_path = run_dir / infer_subdir / exp["file"]
    cfg_path = CONFIGS / f"{exp['run']}.yaml"

    check(csv_path.exists(), f"infer CSV exists: {csv_path.relative_to(ROOT)}")
    check(cfg_path.exists(), f"config exists:    {cfg_path.relative_to(ROOT)}")
    if not csv_path.exists() or not cfg_path.exists():
        return

    df = pd.read_csv(csv_path)
    models = df["model_name"].dropna().unique()
    check(
        len(models) == 1 and models[0] == EXPECTED_MODEL,
        f"model_name = {EXPECTED_MODEL} (saw {list(models)})",
    )
    methods = df["method"].dropna().unique()
    if exp["ours"]:
        check(
            len(methods) == 1 and "steered" in methods[0].lower(),
            f"method column = steered_* (saw {list(methods)})",
        )
    else:
        check(
            len(methods) == 1 and methods[0] == "baseline",
            f"method column = baseline (saw {list(methods)})",
        )

    dec = json.loads(df.iloc[0]["decoding_json"])
    check(dec.get("temperature") == exp["temp"], f"temperature = {exp['temp']} (saw {dec.get('temperature')})")
    check(dec.get("min_p") == exp["min_p"],     f"min_p       = {exp['min_p']} (saw {dec.get('min_p')})")
    check(dec.get("top_p") == exp["top_p"],     f"top_p       = {exp['top_p']} (saw {dec.get('top_p')})")

    cfg = yaml.safe_load(cfg_path.read_text())
    check(cfg.get("model_name") == EXPECTED_MODEL, "yaml model_name matches")
    check(cfg.get("seed") == 42,                   f"yaml seed = 42 (saw {cfg.get('seed')})")
    dec_key = "decoding_aut" if task == "aut" else "decoding_macgyver"
    ydec = cfg.get(dec_key, {}) or {}
    check(ydec.get("temperature") == exp["temp"], f"yaml {dec_key}.temperature = {exp['temp']}")
    if exp["ours"]:
        steer = (cfg.get("aut_steering" if task == "aut" else "macgyver_steering") or {})
        check(steer.get("alphas") == [0.17],                            "yaml steering alphas = [0.17]")
        check(steer.get("steer_layers_group") == [18, 19, 20, 21, 22, 23], "yaml steering layers = [18..23]")


def audit_paired_design(task: str, items_file: str) -> None:
    print(f"\n=== {task}: paired design + provenance trace ===")
    items = json.loads((DATA / items_file).read_text())

    # 1) Same source instances across methods
    if task == "aut":
        key = "object"
        n_expected = N_OBJECTS
    else:
        key = "problem_id"
        n_expected = N_PROBLEMS
    total_expected = n_expected * len(EXPECTED[task])

    by_method: dict[str, set] = {}
    for it in items:
        by_method.setdefault(it["method"], set()).add(it[key])
    method_sets = list(by_method.values())
    common = set.intersection(*method_sets) if method_sets else set()
    union = set.union(*method_sets) if method_sets else set()
    check(common == union, f"all 3 methods cover the same set of {key}s (|∩|={len(common)}, |∪|={len(union)})")
    check(
        all(len(s) == n_expected for s in method_sets),
        f"each method has exactly {n_expected} {key}s (saw {[len(s) for s in method_sets]})",
    )
    check(len(items) == total_expected, f"total items = {total_expected} (saw {len(items)})")

    # MG: all sampled problems must fit the length cap
    if task == "macgyver":
        oversize = [it for it in items if len(it.get("problem_text", "")) > MG_MAX_PROBLEM_LEN]
        check(
            not oversize,
            f"every MG problem_text ≤ {MG_MAX_PROBLEM_LEN} chars "
            f"(violations: {len(oversize)}, max seen: {max((len(it['problem_text']) for it in items), default=0)})",
        )

    # 2) Provenance trace — every item's source_item_id lives in the right CSV.
    infer_subdir = "aut_infer" if task == "aut" else "macgyver_infer"
    method_groups: dict[str, list[dict]] = {}
    for it in items:
        method_groups.setdefault(it["method"], []).append(it)
    for method, group in method_groups.items():
        exp = EXPECTED[task][method]
        df = pd.read_csv(RESULTS / exp["run"] / infer_subdir / exp["file"])
        df_ids = set(df["item_id"].astype(str))
        sample_ids = {str(it["source_item_id"]) for it in group}
        unknown = sample_ids - df_ids
        check(
            not unknown,
            f"{method}: all {len(group)} sample items trace to {exp['run']}/{infer_subdir}/{exp['file']}",
        )
        df_by_id = {str(r["item_id"]): r for _, r in df.iterrows()}
        mismatches = 0
        for it in group:
            row = df_by_id.get(str(it["source_item_id"]))
            if row is None:
                mismatches += 1
                continue
            if task == "aut":
                try:
                    uses = [u.strip() for u in json.loads(row["parsed_response"])]
                except Exception:
                    mismatches += 1
                    continue
                if it["content"] not in uses:
                    mismatches += 1
            else:
                if str(row["parsed_response"]).strip() != it["content"]:
                    mismatches += 1
                if not (bool(row.get("solvable", True)) and bool(row.get("unconventional", True))):
                    mismatches += 1
        check(mismatches == 0, f"{method}: every item's text matches source parsed_response (mismatches={mismatches})")


def main() -> None:
    for task, methods in EXPECTED.items():
        for method, exp in methods.items():
            audit_run(task, method, exp)

    audit_paired_design("aut", "aut_items.json")
    audit_paired_design("macgyver", "macgyver_items.json")

    print("\n\n=== summary ===")
    if fails:
        print(f"FAIL — {len(fails)} check(s) failed:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS ✅")


if __name__ == "__main__":
    main()
