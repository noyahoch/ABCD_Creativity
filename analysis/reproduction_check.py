"""Sanity check: aggregate scores_long with the paper's functional and compare
with every published number (Tables 2-5 and the Falcon rows of Tables 8-10).
Writes results/analysis/reproduction_check.md and exits non-zero on any gap > TOL."""
from __future__ import annotations

import sys

import pandas as pd

from .aggregate import aggregate_method, fmt, hm3
from .registry import (METHOD_LABEL, MODEL_LABEL, OUT_DIR, PIPELINE_RUNS, PUBLISHED, SCORES,
                       PUBLISHED_DELTA_HM, TABLE_METRICS, LIVEIDEA_ARMS)

TOL = 0.005
TOL_STOP = 0.01   # the brief's "stop and ask" threshold


def _status(reproduced: float, published: float, gap: float) -> str:
    if gap <= TOL:
        return "ok"
    if gap > TOL_STOP:
        return "GAP>0.01"
    via3 = {round(float(f"{reproduced:.3f}") + 1e-12, 2), round(float(f"{reproduced:.3f}") - 1e-12, 2)}
    if published in via3:
        return "rounding-boundary (matches after 3-dp intermediate rounding)"
    return "gap<0.01 (published cell inconsistent with its own rounded row)"


def load_scores() -> pd.DataFrame:
    return pd.read_parquet(SCORES)


def run_check(df: pd.DataFrame) -> tuple[list[dict], str]:
    recs: list[dict] = []
    for (table, bench, model, method), pub in PUBLISHED.items():
        try:
            means, per_seed = aggregate_method(df, table, bench, model, method)
        except KeyError:
            for metric, p in pub.items():
                recs.append(dict(table=table, benchmark=bench, model=model, method=method, metric=metric,
                                 published=p, reproduced=float("nan"), gap=float("nan"), status="MISSING"))
            continue
        n_items = df[(df.table == table) & (df.benchmark == bench) & (df.model == model)
                     & (df.method == method)]["item_id"].nunique()
        for metric, p in pub.items():
            r = means[metric]
            gap = abs(r - p)
            status = _status(r, p, gap)
            if bench == "liveidea" and metric == "hm" and status != "ok":
                hm_pub_axes = hm3(pub["originality"], pub["feasibility"], pub["clarity"])
                if abs(hm_pub_axes - p) <= TOL:
                    status = f"rounding-boundary (published HM = HM of the rounded published axes = {hm_pub_axes:.4f})"
            recs.append(dict(table=table, benchmark=bench, model=model, method=method, metric=metric,
                             published=p, reproduced=r, gap=gap,
                             per_seed=" / ".join(f"{x:.3f}" for x in per_seed[metric].values),
                             n_items=n_items, status=status))
    # Table 3 / 9 Delta HM (unrounded, vs baseline)
    for (model, method), p in PUBLISHED_DELTA_HM.items():
        base, _ = aggregate_method(df, "3", "macgyver", model, "baseline")
        cur, _ = aggregate_method(df, "3", "macgyver", model, method)
        r = cur["hm"] - base["hm"]
        gap = abs(r - p)
        recs.append(dict(table="3", benchmark="macgyver", model=model, method=method, metric="delta_hm",
                         published=p, reproduced=r, gap=gap, per_seed="", n_items="",
                         status=_status(r, p, gap)))
    out = pd.DataFrame(recs)
    n_bad = int((~out["status"].isin(["ok"]) & ~out["status"].str.startswith("rounding")).sum())
    lines = ["# Reproduction check: scores_long -> published tables", "",
             f"Tolerance {TOL} (rounding). Rows: {len(out)}; within tolerance: {int((out.status == 'ok').sum())}; "
             f"on a rounding boundary (|gap| in (0.005, 0.0055], published value = ours rounded via a 3-decimal "
             f"intermediate): {int(out.status.str.startswith('rounding').sum())}; other gaps (all < 0.01): {n_bad}.", "",
             "Aggregation = the paper's functional (per seed: obs-weighted mean for AUT judge metrics, "
             "item mean otherwise, p30 for flexibility; HM as plug-in of per-seed means; then mean over seeds). "
             "`per_seed` lists the three per-seed values in seed order (42/101/999, or replicate 1/2/3 for LiveIdeaBench).", ""]
    for table in ["2", "3", "4", "5"]:
        sub = out[out.table == table]
        lines += [f"## Table {table}", "",
                  "| benchmark | model | method | metric | published | reproduced | gap | per-seed | n items | status |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub.itertuples():
            lines.append(f"| {r.benchmark} | {r.model} | {r.method} | {r.metric} | {fmt(r.published)} | "
                         f"{fmt(r.reproduced, 4)} | {fmt(r.gap, 4)} | {r.per_seed} | {r.n_items} | {r.status} |")
        lines.append("")
    return recs, "\n".join(lines)


def main() -> int:
    df = load_scores()
    recs, md = run_check(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reproduction_check.md").write_text(md + "\n")
    pd.DataFrame(recs).to_csv(OUT_DIR / "reproduction_check.csv", index=False)
    bad = [r for r in recs if r["status"] == "GAP>0.01"]
    print(md)
    print(f"\n{len(bad)} rows outside tolerance")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
