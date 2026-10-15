"""Quick read-out of the human ratings DB.

Usage:
    # summary across all raters, both tasks
    python bridge_steering/human_eval/view_results.py

    # one rater
    python bridge_steering/human_eval/view_results.py --rater 666

    # one task
    python bridge_steering/human_eval/view_results.py --task aut

    # write item-level CSV (one row per rating, with method + judge columns)
    python bridge_steering/human_eval/view_results.py --export ratings_export.csv

The CSV is the most useful artefact for stats — pass it to whatever
notebook / analysis you want.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DB_PATH = HERE / "responses" / "ratings.db"
DATA_DIR = HERE / "data"


def _load_items(task: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads((DATA_DIR / f"{task}_items.json").read_text()))


def _load_ratings(rater: str | None, task: str | None) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    q = "SELECT rater_id, task, item_id, novelty, usability, overall, ts FROM ratings"
    where, params = [], []
    if rater:
        where.append("rater_id = ?")
        params.append(rater)
    if task:
        where.append("task = ?")
        params.append(task)
    if where:
        q += " WHERE " + " AND ".join(where)
    return pd.read_sql_query(q, con, params=params)


def _join_metadata(ratings: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for t in ratings["task"].unique():
        items = _load_items(t)
        cols = ["item_id", "method"]
        if t == "aut":
            cols += ["object", "use_idx", "judge_novelty", "judge_usability", "judge_overall"]
        else:
            cols += ["problem_id", "judge_quality"]
        part = ratings[ratings["task"] == t].merge(items[cols], on="item_id", how="left")
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else ratings


def _harmonic_mean(a, b):
    """Harmonic mean of two non-negative scores; 0 if either is 0 or NaN."""
    import math
    if pd.isna(a) or pd.isna(b):
        return math.nan
    a, b = float(a), float(b)
    if a + b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


def _attach_harmonic(df: pd.DataFrame) -> pd.DataFrame:
    """For AUT rows, add ×2-scaled human scores and harmonic-mean overall
    columns for both human and judge (both on the same 1-10 scale)."""
    if "novelty" not in df.columns:
        return df
    aut_mask = df["task"] == "aut"
    df.loc[aut_mask, "hum_novelty_x2"] = df.loc[aut_mask, "novelty"] * 2
    df.loc[aut_mask, "hum_usability_x2"] = df.loc[aut_mask, "usability"] * 2
    df.loc[aut_mask, "hum_overall_hm"] = df.loc[aut_mask].apply(
        lambda r: _harmonic_mean(r["hum_novelty_x2"], r["hum_usability_x2"]), axis=1
    )
    df.loc[aut_mask, "judge_overall_hm"] = df.loc[aut_mask].apply(
        lambda r: _harmonic_mean(r["judge_novelty"], r["judge_usability"]), axis=1
    )
    return df


def _summary(df: pd.DataFrame) -> None:
    df = _attach_harmonic(df)

    print("=== progress ===")
    prog = df.groupby(["rater_id", "task"]).size().reset_index(name="n")
    print(prog.to_string(index=False))

    for task in df["task"].unique():
        sub = df[df["task"] == task]
        print(f"\n=== {task.upper()}: per-method means ===")
        if task == "aut":
            agg = sub.groupby("method").agg(
                n=("item_id", "count"),
                hum_novelty_x2=("hum_novelty_x2", "mean"),
                hum_usability_x2=("hum_usability_x2", "mean"),
                hum_overall_hm=("hum_overall_hm", "mean"),
                judge_novelty=("judge_novelty", "mean"),
                judge_usability=("judge_usability", "mean"),
                judge_overall_hm=("judge_overall_hm", "mean"),
            ).round(2)
        else:
            agg = sub.groupby("method").agg(
                n=("item_id", "count"),
                hum_overall=("overall", "mean"),
                judge_quality=("judge_quality", "mean"),
            ).round(2)
        print(agg.to_string())

        print(f"\n--- {task.upper()}: human ↔ judge correlation (item-level, both on 1-10) ---")
        if task == "aut":
            pairs = [
                ("novelty",   "hum_novelty_x2",   "judge_novelty"),
                ("usability", "hum_usability_x2", "judge_usability"),
                ("overall (harmonic)", "hum_overall_hm", "judge_overall_hm"),
            ]
            for label, hcol, jcol in pairs:
                m = sub[[hcol, jcol]].dropna()
                if len(m) < 3:
                    continue
                r = m[hcol].corr(m[jcol])
                rho = m[hcol].corr(m[jcol], method="spearman")
                print(f"  {label:>20s}: Pearson r = {r:.3f}, Spearman ρ = {rho:.3f}  (n={len(m)})")
        else:
            m = sub[["overall", "judge_quality"]].dropna()
            if len(m) >= 3:
                r = m["overall"].corr(m["judge_quality"])
                rho = m["overall"].corr(m["judge_quality"], method="spearman")
                print(f"           overall: Pearson r = {r:.3f}, Spearman ρ = {rho:.3f}  (n={len(m)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", help="filter to one rater_id")
    ap.add_argument("--task", choices=("aut", "macgyver"), help="filter to one task")
    ap.add_argument("--export", type=Path, help="write joined item-level CSV here")
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"no ratings DB at {DB_PATH}")
    ratings = _load_ratings(args.rater, args.task)
    if ratings.empty:
        print("No ratings yet.")
        return
    df = _join_metadata(ratings)

    _summary(df)

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"\nWrote {len(df)} rows to {args.export}")


if __name__ == "__main__":
    main()
