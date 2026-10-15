"""Pull current ratings from the Google Sheet and print per-method aggregates
for AUT, MacGyver, and Pairwise — without touching the deployed app.

Usage:
    pip install gspread google-auth pandas
    python bridge_steering/human_eval/view_live_results.py \\
        --key /path/to/service-account.json \\
        --sheet "https://docs.google.com/spreadsheets/d/.../edit"

The script reads the same three tabs (aut_ratings, macgyver_ratings,
cw_design_ratings) that the deployed app writes to, joins them with the
local data files (aut_items.json / macgyver_items.json / pairwise_trials.json)
to get method labels, and prints summary stats.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PAIRWISE_TRIALS_PATH = DATA_DIR / "pairwise_trials.json"

WS_AUT      = "aut_ratings"
WS_MACGYVER = "macgyver_ratings"
WS_PAIRWISE = "cw_design_ratings"


def _open_spreadsheet(key_path: Path, sheet_url: str):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        sys.exit("Install dependencies first: pip install gspread google-auth pandas")
    creds = Credentials.from_service_account_file(
        str(key_path),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
        ],
    )
    client = gspread.authorize(creds)
    return client.open_by_url(sheet_url)


def _ws_records(spreadsheet, name: str) -> list[dict]:
    try:
        ws = spreadsheet.worksheet(name)
    except Exception:
        print(f"  ! worksheet '{name}' not found")
        return []
    try:
        return ws.get_all_records()
    except Exception as e:
        print(f"  ! could not read '{name}': {e}")
        return []


# ---------- AUT ----------

def aut_report(records: list[dict]) -> None:
    try:
        items = json.loads((DATA_DIR / "aut_items.json").read_text())
    except Exception:
        items = []
    method_of = {it["item_id"]: it["method"] for it in items}

    per_method: dict[str, dict[str, list[int]]] = {}
    raters = Counter()
    for r in records:
        rid = str(r.get("rater_id", "")).strip()
        if rid:
            raters[rid] += 1
        iid = str(r.get("item_id", ""))
        m = method_of.get(iid, "?")
        per_method.setdefault(m, {"nov": [], "use": []})
        for col, key in (("novelty", "nov"), ("usability", "use")):
            try:
                v = int(float(r.get(col) or 0))
                if v > 0:
                    per_method[m][key].append(v)
            except (TypeError, ValueError):
                pass

    print(f"=== AUT  ({len(records)} answers · {len(raters)} unique raters) ===")
    if not per_method:
        print("  (no aut ratings yet)")
        return
    print(f"  {'method':<28s}  {'novelty (mean·n)':<20s}  {'usability (mean·n)'}")
    for m in sorted(per_method):
        s = per_method[m]
        mn = sum(s["nov"]) / len(s["nov"]) if s["nov"] else 0
        mu = sum(s["use"]) / len(s["use"]) if s["use"] else 0
        print(
            f"  {m:<28s}  "
            f"{mn:5.2f}  (n={len(s['nov']):3d})    "
            f"{mu:5.2f}  (n={len(s['use']):3d})"
        )
    print("  raters: " + ", ".join(f"{r}({c})" for r, c in raters.most_common()))


# ---------- MacGyver ----------

def macgyver_report(records: list[dict]) -> None:
    try:
        items = json.loads((DATA_DIR / "macgyver_items.json").read_text())
    except Exception:
        items = []
    method_of = {it["item_id"]: it["method"] for it in items}

    per_method: dict[str, list[int]] = {}
    raters = Counter()
    for r in records:
        rid = str(r.get("rater_id", "")).strip()
        if rid:
            raters[rid] += 1
        iid = str(r.get("item_id", ""))
        m = method_of.get(iid, "?")
        per_method.setdefault(m, [])
        try:
            v = int(float(r.get("overall") or 0))
            if v > 0:
                per_method[m].append(v)
        except (TypeError, ValueError):
            pass

    print(f"\n=== MacGyver  ({len(records)} answers · {len(raters)} unique raters) ===")
    if not per_method:
        print("  (no macgyver ratings yet)")
        return
    print(f"  {'method':<28s}  {'overall (mean·n)'}")
    for m in sorted(per_method):
        vals = per_method[m]
        mean = sum(vals) / len(vals) if vals else 0
        print(f"  {m:<28s}  {mean:5.2f}  (n={len(vals):3d})")
    print("  raters: " + ", ".join(f"{r}({c})" for r, c in raters.most_common()))


# ---------- Pairwise ----------

def pairwise_report(records: list[dict]) -> None:
    try:
        data = json.loads(PAIRWISE_TRIALS_PATH.read_text())
    except Exception:
        data = {"trials": []}
    trial_map = {
        t["trial_id"]: {
            "a_method": t.get("a_internal_method", "?"),
            "b_method": t.get("b_internal_method", "?"),
            "pair_id":  t.get("pair_id", "?"),
        }
        for t in data.get("trials", [])
    }

    raters = Counter()
    wins = Counter()
    strengths = Counter()
    per_pair_ours = Counter()  # pair_id -> [ours_won, total]
    total = 0
    for r in records:
        rid = str(r.get("rater_id", "")).strip()
        if rid:
            raters[rid] += 1
        tid = str(r.get("trial_id", ""))
        info = trial_map.get(tid, {})
        choice = str(r.get("rater_choice", ""))
        side = "A" if "A is" in choice else ("B" if "B is" in choice else None)
        if not side:
            continue
        winner = info.get(f"{side.lower()}_method", "?")
        strength = "clearly" if "clearly" in choice else "slightly"
        wins[winner] += 1
        strengths[(winner, strength)] += 1
        total += 1
        # Track pair-level ours-win rate
        ours_side = str(r.get("ours_position", "")).strip() or None
        pair_id = info.get("pair_id", "?")
        if ours_side in {"A", "B"}:
            per_pair_ours[(pair_id, "total")] += 1
            if side == ours_side:
                per_pair_ours[(pair_id, "ours_won")] += 1

    print(f"\n=== Pairwise  ({len(records)} answers · {len(raters)} unique raters) ===")
    if total == 0:
        print("  (no pairwise answers yet)")
        return
    print(f"  Wins per method (of {total} total):")
    for m, c in wins.most_common():
        pct = c / total * 100
        cl = strengths.get((m, "clearly"), 0)
        sl = strengths.get((m, "slightly"), 0)
        print(f"    {m:<32s}  {c:3d}  ({pct:5.1f}%)   clearly={cl}, slightly={sl}")
    print("  Ours-win rate by pair type:")
    pair_ids = sorted({k[0] for k in per_pair_ours.keys()})
    for pid in pair_ids:
        won  = per_pair_ours.get((pid, "ours_won"), 0)
        tot  = per_pair_ours.get((pid, "total"), 0)
        pct  = (won / tot * 100) if tot else 0
        print(f"    {pid:<32s}  ours: {won:3d} / {tot:3d}  ({pct:5.1f}%)")
    print("  raters: " + ", ".join(f"{r}({c})" for r, c in raters.most_common()))


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, required=True,
                    help="path to the service-account JSON key")
    ap.add_argument("--sheet", required=True,
                    help="spreadsheet URL")
    args = ap.parse_args()

    if not args.key.exists():
        sys.exit(f"key not found: {args.key}")

    spreadsheet = _open_spreadsheet(args.key, args.sheet)
    aut_report(_ws_records(spreadsheet, WS_AUT))
    macgyver_report(_ws_records(spreadsheet, WS_MACGYVER))
    pairwise_report(_ws_records(spreadsheet, WS_PAIRWISE))


if __name__ == "__main__":
    main()
