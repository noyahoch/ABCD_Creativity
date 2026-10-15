import numpy as np
import pandas as pd
import pytest

from analysis.build_scores_long import COLUMNS


def make_long(rng, n_items=30, seeds=(42, 101, 999), methods=("a", "b"), benchmark="aut", table="T",
              model="m", true_diff=0.0, n_obs_range=(6, 10), metrics=None, identical=False, noise=0.2,
              return_truth=False):
    """Synthetic long table in the real schema.

    Noise-free per-item/seed means carry a shared item effect; method 'a' is shifted by
    ``true_diff`` on every metric. Independent N(0, noise) measurement noise is added per
    (method, item, seed, metric) row. With ``return_truth`` the noise-free table is returned
    too, so the *true* paired difference of any (derived) metric is known exactly.
    """
    metrics = metrics or {"aut": ["ocsai", "novelty", "usability", "semdiv"],
                          "macgyver": ["orig4", "quality"]}[benchmark]
    rows, truth = [], []
    item_eff = rng.normal(0, 1, n_items)
    for k, it in enumerate([f"item{i:03d}" for i in range(n_items)]):
        for s in seeds:
            for metric in metrics:
                base = 5 + item_eff[k] + rng.normal(0, 0.3)
                n = rng.integers(*n_obs_range) if metric in ("novelty", "usability") else 1
                for m in methods:
                    clean = base + (true_diff if m == "a" else 0.0)
                    v = clean + (0.0 if identical else rng.normal(0, noise))
                    common = dict(table=table, benchmark=benchmark, model=model, method=m, item_id=it,
                                  seed=s, metric=metric, n_obs=int(n), run_id=f"{m}_s{s}")
                    rows.append(dict(common, value=float(v)))
                    truth.append(dict(common, value=float(clean)))
    df = pd.DataFrame(rows, columns=COLUMNS)
    if identical:
        b = df[df.method == "a"].copy()
        b["method"] = "b"
        df = pd.concat([df[df.method == "a"], b], ignore_index=True)
    if return_truth:
        return df, pd.DataFrame(truth, columns=COLUMNS)
    return df


@pytest.fixture
def rng():
    return np.random.default_rng(123)
