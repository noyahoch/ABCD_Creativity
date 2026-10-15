"""The paper's aggregation functional, applied to the long score table.

Per seed:
  * ``obs``  weighting: sum(value * n_obs) / sum(n_obs) over items   (AUT judge novelty /
             usability = mean over all judged uses, as in bridge_steering/stages/aggregate.py)
  * ``item`` weighting: mean of value over items with a value          (OCSAI per reply,
             MacGyver per problem, LiveIdeaBench per keyword)
  * ``p30``: 30th percentile over items                                 (LiveIdea flexibility)
Derived metrics (harmonic means) are plug-in functionals of the per-seed aggregates,
exactly as the pipeline computes them (HM(mean n, mean u), not mean of per-item HMs).
Across seeds: unweighted mean of the per-seed values (build_final_tables.py / the
LiveIdea table scripts average the three per-seed arm values).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .registry import BASE_METRICS, DERIVED_METRICS, TABLE_METRICS


def hm2(a, b):
    return 2.0 * a * b / (a + b)


def hm_mac(o, q):
    return hm2(o, q / 5.0)


def hm3(a, b, c):
    return 3.0 / (1.0 / a + 1.0 / b + 1.0 / c)


FUNCTIONALS = {"hm2": hm2, "hm_mac": hm_mac, "hm3": hm3, "identity": lambda x: x}


def per_seed_base(sub: pd.DataFrame, weighting: str) -> pd.Series:
    """``sub`` holds one base metric of one (table, benchmark, model, method)."""
    out = {}
    for seed, g in sub.groupby("seed", sort=True):
        v = g["value"].to_numpy(dtype=float)
        n = g["n_obs"].to_numpy(dtype=float)
        ok = (n > 0) & ~np.isnan(v)
        if weighting == "obs":
            out[seed] = float(np.sum(v[ok] * n[ok]) / np.sum(n[ok]))
        elif weighting == "item":
            out[seed] = float(np.mean(v[ok]))
        elif weighting == "p30":
            out[seed] = float(np.percentile(v[ok], 30))
        else:
            raise ValueError(weighting)
    return pd.Series(out)


def aggregate_method(df: pd.DataFrame, table: str, benchmark: str, model: str, method: str
                     ) -> tuple[dict[str, float], dict[str, pd.Series]]:
    """Return (metric -> across-seed mean, metric -> per-seed Series) for every
    metric of the benchmark's published table (base + derived)."""
    d = df[(df["table"] == table) & (df["benchmark"] == benchmark)
           & (df["model"] == model) & (df["method"] == method)]
    if d.empty:
        raise KeyError((table, benchmark, model, method))
    per_seed: dict[str, pd.Series] = {}
    for metric, weighting in BASE_METRICS[benchmark].items():
        sub = d[d["metric"] == metric]
        if not sub.empty:
            per_seed[metric] = per_seed_base(sub, weighting)
    for metric, (needs, fname) in DERIVED_METRICS[benchmark].items():
        if all(n in per_seed for n in needs):
            f = FUNCTIONALS[fname]
            per_seed[metric] = pd.Series({s: f(*[per_seed[n][s] for n in needs])
                                          for s in per_seed[needs[0]].index})
    means = {m: float(per_seed[m].mean()) for m in TABLE_METRICS[benchmark] if m in per_seed}
    return means, per_seed


def fmt(x: float | None, nd: int = 2) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"
