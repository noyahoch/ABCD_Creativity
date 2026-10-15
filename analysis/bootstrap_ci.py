"""Paired item-level bootstrap confidence intervals.

Resampling unit = item (AUT object / MacGyver problem / LiveIdeaBench keyword). Each draw
samples items with replacement; a drawn item brings all of its seeds and both methods'
scores, so pairing and the seed structure are preserved inside every draw. Seeds and
individual generations are never resampled.

Statistic = A(method_a) - A(method_b) where A is exactly the paper's aggregation
(``aggregate.py``): per seed an obs-weighted mean (AUT judge metrics), item mean, or p30
(flexibility); derived metrics (HM) as plug-in functionals of the per-seed aggregates;
then the mean over seeds. The point estimate therefore equals the table delta.

10,000 resamples, 95% percentile interval, numpy RNG seed 0 (``np.random.default_rng``).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aggregate import FUNCTIONALS, aggregate_method
from .registry import (BASE_METRICS, COMPARISONS, DERIVED_METRICS, METHOD_LABEL, METRIC_LABEL,
                       MODEL_LABEL, OUT_DIR, SCORES, TABLE_BENCHMARKS, TABLE_METRICS, TABLE_MODELS)

N_RESAMPLES = 10_000
SEED = 0
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Dense per-method arrays
# ---------------------------------------------------------------------------

@dataclass
class MethodArrays:
    """value[metric] and n_obs[metric]: arrays of shape (n_seeds, n_items); NaN where n_obs == 0."""
    items: list[str]
    seeds: list[int]
    value: dict[str, np.ndarray]
    n_obs: dict[str, np.ndarray]


def method_arrays(df: pd.DataFrame, table: str, benchmark: str, model: str, method: str,
                  items: list[str] | None = None) -> MethodArrays:
    d = df[(df.table == table) & (df.benchmark == benchmark) & (df.model == model) & (df.method == method)]
    if d.empty:
        raise KeyError((table, benchmark, model, method))
    seeds = sorted(d["seed"].unique())
    if items is None:
        items = sorted(d["item_id"].unique())
    idx_i = {it: k for k, it in enumerate(items)}
    idx_s = {s: k for k, s in enumerate(seeds)}
    value, n_obs = {}, {}
    for metric, g in d.groupby("metric"):
        V = np.full((len(seeds), len(items)), np.nan)
        N = np.zeros((len(seeds), len(items)))
        ii = g["item_id"].map(idx_i).to_numpy()
        ss = g["seed"].map(idx_s).to_numpy()
        assert not np.isnan(ii.astype(float)).any(), "item outside the paired item set"
        V[ss, ii] = g["value"].to_numpy(dtype=float)
        N[ss, ii] = g["n_obs"].to_numpy(dtype=float)
        assert N.size == len(g), (metric, "incomplete item x seed grid")
        V[N == 0] = np.nan
        value[metric], n_obs[metric] = V, N
    return MethodArrays(list(items), list(seeds), value, n_obs)


# ---------------------------------------------------------------------------
# Statistic on a (weighted) multiset of items
# ---------------------------------------------------------------------------

def _per_seed_base_from_counts(V: np.ndarray, N: np.ndarray, W: np.ndarray, weighting: str) -> np.ndarray:
    """V, N: (seeds, items). W: (draws, items) non-negative counts. Returns (draws, seeds)."""
    if weighting in ("obs", "item"):
        ok = (N > 0) & ~np.isnan(V)
        w = N if weighting == "obs" else ok.astype(float)
        num = np.where(ok, V * w, 0.0)          # (seeds, items)
        den = np.where(ok, w, 0.0)
        return (W @ num.T) / (W @ den.T)
    raise ValueError(weighting)


def _per_seed_p30_from_index(V: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """idx: (draws, items) resampled item indices. Returns (draws, seeds)."""
    out = np.empty((idx.shape[0], V.shape[0]))
    for s in range(V.shape[0]):
        vals = V[s][idx]                          # (draws, items)
        out[:, s] = np.nanpercentile(vals, 30, axis=1)
    return out


def statistic(a: MethodArrays, b: MethodArrays, benchmark: str, metric: str,
              W: np.ndarray, idx: np.ndarray | None) -> np.ndarray:
    """A(a) - A(b) for every draw. W: counts (draws, items); idx: indices (draws, items), only
    needed for percentile-type metrics."""
    needs, fname = DERIVED_METRICS[benchmark].get(metric, ((metric,), "identity"))
    f = FUNCTIONALS[fname]

    def agg(m: MethodArrays) -> np.ndarray:            # (draws, seeds)
        parts = []
        for base in needs:
            weighting = BASE_METRICS[benchmark][base]
            if weighting == "p30":
                parts.append(_per_seed_p30_from_index(m.value[base], idx))
            else:
                parts.append(_per_seed_base_from_counts(m.value[base], m.n_obs[base], W, weighting))
        return f(*parts)

    return agg(a).mean(axis=1) - agg(b).mean(axis=1)


def paired_bootstrap(a: MethodArrays, b: MethodArrays, benchmark: str, metric: str,
                     n_resamples: int = N_RESAMPLES, seed: int = SEED, chunk: int = 2000) -> dict:
    assert a.items == b.items and a.seeds == b.seeds, "methods must share items and seeds"
    n = len(a.items)
    rng = np.random.default_rng(seed)
    # point estimate: every item once
    W1 = np.ones((1, n))
    idx1 = np.arange(n)[None, :]
    point = float(statistic(a, b, benchmark, metric, W1, idx1)[0])
    needs_idx = any(BASE_METRICS[benchmark][m] == "p30"
                    for m in DERIVED_METRICS[benchmark].get(metric, ((metric,), ""))[0])
    draws = np.empty(n_resamples)
    done = 0
    while done < n_resamples:
        k = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(k, n))
        W = np.zeros((k, n))
        np.add.at(W, (np.repeat(np.arange(k), n), idx.ravel()), 1.0)
        draws[done:done + k] = statistic(a, b, benchmark, metric, W, idx if needs_idx else None)
        done += k
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return dict(point=point, ci_lo=float(lo), ci_hi=float(hi),
                excludes_zero=bool(lo > 0 or hi < 0), n_items=n, n_seeds=len(a.seeds),
                n_resamples=n_resamples, rng_seed=seed, draws=draws)


# ---------------------------------------------------------------------------
# Driver: all comparisons of the paper tables
# ---------------------------------------------------------------------------

def run_all(df: pd.DataFrame, n_resamples: int = N_RESAMPLES, seed: int = SEED,
            tables: list[str] | None = None,
            comparisons: dict[str, list[tuple[str, str]]] | None = None,
            benchmarks: list[str] | None = None, models: list[str] | None = None) -> pd.DataFrame:
    comparisons = comparisons or COMPARISONS
    recs = []
    for table in (tables or ["2", "3", "4", "5"]):
        for benchmark in TABLE_BENCHMARKS[table]:
            if benchmarks and benchmark not in benchmarks:
                continue
            for model in TABLE_MODELS[table]:
                if models and model not in models:
                    continue
                cache = {}
                for ma, mb in comparisons[table]:
                    for m in (ma, mb):
                        if m not in cache:
                            cache[m] = method_arrays(df, table, benchmark, model, m)
                    a, b = cache[ma], cache[mb]
                    means_a, _ = aggregate_method(df, table, benchmark, model, ma)
                    means_b, _ = aggregate_method(df, table, benchmark, model, mb)
                    for metric in TABLE_METRICS[benchmark]:
                        needs = DERIVED_METRICS[benchmark].get(metric, ((metric,), ""))[0]
                        if any(not np.any(m.n_obs[bm] > 0) for m in (a, b) for bm in needs):
                            print(f"T{table} {benchmark} {model} {ma}-{mb} {metric}: skipped (no scores)", flush=True)
                            continue
                        r = paired_bootstrap(a, b, benchmark, metric, n_resamples, seed)
                        table_delta = means_a[metric] - means_b[metric]
                        assert abs(r["point"] - table_delta) < 1e-9, (table, model, metric, r["point"], table_delta)
                        recs.append(dict(table=table, benchmark=benchmark, model=model, metric=metric,
                                         method_a=ma, method_b=mb, mean_a=means_a[metric], mean_b=means_b[metric],
                                         delta=r["point"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                         excludes_zero=r["excludes_zero"], n_items=r["n_items"],
                                         n_seeds=r["n_seeds"], n_resamples=r["n_resamples"], rng_seed=r["rng_seed"]))
                        print(f"T{table} {benchmark:8s} {model:7s} {ma:>10s}-{mb:<10s} {metric:12s} "
                              f"d={r['point']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
                              f"{' *' if r['excludes_zero'] else ''}", flush=True)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# LaTeX snippets + summary
# ---------------------------------------------------------------------------

def _cell(r, nd=2) -> str:
    mark = r"$^{\dagger}$" if r.excludes_zero else ""
    return f"${r.delta:+.{nd}f}$ [{r.ci_lo:+.{nd}f}, {r.ci_hi:+.{nd}f}]{mark}"


def write_latex(res: pd.DataFrame, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for table in sorted(res.table.unique()):
        sub = res[res.table == table]
        L = [f"% Table {table}: paired item-level bootstrap, 95% percentile CI, {int(sub.n_resamples.iloc[0]):,} resamples,",
             "% unit = item (object / problem / keyword), seeds kept together inside each draw. dagger = CI excludes 0.",
             "% Column order follows the paper's table. Paste the rows into the existing tabular.", ""]
        for benchmark in TABLE_BENCHMARKS[table]:
            metrics = TABLE_METRICS[benchmark]
            L.append(f"% --- {benchmark} ---")
            L.append("\\multicolumn{%d}{l}{\\textit{$\\Delta$ [95\\%% CI]}} \\\\" % (len(metrics) + 1))
            L.append("Comparison & " + " & ".join(f"$\\Delta$ {METRIC_LABEL[m]}" for m in metrics) + r" \\")
            L.append(r"\midrule")
            for model in TABLE_MODELS[table]:
                sb = sub[(sub.benchmark == benchmark) & (sub.model == model)]
                if sb.empty:
                    continue
                if len(TABLE_MODELS[table]) > 1:
                    L.append(f"\\multicolumn{{{len(metrics) + 1}}}{{l}}{{\\textbf{{{MODEL_LABEL[model]}}}}} \\\\")
                for ma, mb in COMPARISONS[table]:
                    sc = sb[(sb.method_a == ma) & (sb.method_b == mb)].set_index("metric")
                    label = f"{METHOD_LABEL[ma]} $-$ {METHOD_LABEL[mb]}"
                    L.append(label + " & " + " & ".join(_cell(sc.loc[m]) for m in metrics) + r" \\")
            L.append("")
        (out_dir / f"table{table}_ci.tex").write_text("\n".join(L) + "\n")


def write_summary(res: pd.DataFrame, path) -> None:
    L = ["# Bootstrap CI summary", "",
         "Paired item-level bootstrap (unit = AUT object / MacGyver problem / LiveIdeaBench keyword; each drawn item keeps "
         "all 3 seeds and both methods), 10,000 resamples, 95% percentile interval, numpy seed 0. "
         "'significant' = interval excludes 0. Deltas are method_a - method_b on the paper's aggregation.", ""]
    for table in sorted(res.table.unique()):
        L.append(f"## Table {table}")
        sub = res[res.table == table]
        for (benchmark, model, ma, mb), g in sub.groupby(["benchmark", "model", "method_a", "method_b"], sort=False):
            sig = [f"{METRIC_LABEL[r.metric]} ({r.delta:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}])" for r in g.itertuples() if r.excludes_zero]
            ns = [f"{METRIC_LABEL[r.metric]} ({r.delta:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}])" for r in g.itertuples() if not r.excludes_zero]
            L.append(f"- **{MODEL_LABEL[model]}, {benchmark}: {METHOD_LABEL[ma]} - {METHOD_LABEL[mb]}** "
                     f"(n={int(g.n_items.iloc[0])} items)")
            L.append(f"  - CI excludes 0: {', '.join(sig) if sig else 'none'}")
            L.append(f"  - CI includes 0: {', '.join(ns) if ns else 'none'}")
        L.append("")
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=str(SCORES))
    p.add_argument("--out", default=str(OUT_DIR / "ci_results.csv"))
    p.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tables", nargs="*", default=None)
    p.add_argument("--no-latex", action="store_true")
    a = p.parse_args(argv)
    df = pd.read_parquet(a.scores)
    res = run_all(df, a.n_resamples, a.seed, a.tables)
    res.to_csv(a.out, index=False, float_format="%.6f")
    if not a.no_latex:
        write_latex(res, OUT_DIR / "latex")
        write_summary(res, OUT_DIR / "ci_summary.md")
    print(f"wrote {a.out} ({len(res)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
