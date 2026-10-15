import numpy as np
import pandas as pd
import pytest

from analysis import bootstrap_ci as bc
from analysis.aggregate import aggregate_method
from .conftest import make_long


@pytest.mark.parametrize("metric", ["ocsai", "novelty", "hm", "semdiv"])
def test_known_true_difference_is_covered_and_point_equals_direct(rng, metric):
    df, truth = make_long(rng, n_items=40, true_diff=0.4, noise=0.1, return_truth=True)
    a = bc.method_arrays(df, "T", "aut", "m", "a")
    b = bc.method_arrays(df, "T", "aut", "m", "b")
    r = bc.paired_bootstrap(a, b, "aut", metric, n_resamples=2000, seed=0)
    # the true paired difference = the paper functional on the noise-free table
    ta, _ = aggregate_method(truth, "T", "aut", "m", "a")
    tb, _ = aggregate_method(truth, "T", "aut", "m", "b")
    true = ta[metric] - tb[metric]
    assert r["ci_lo"] <= true <= r["ci_hi"], (metric, true, r)
    # point estimate == direct paired difference of the paper aggregation, exactly
    ma, _ = aggregate_method(df, "T", "aut", "m", "a")
    mb, _ = aggregate_method(df, "T", "aut", "m", "b")
    assert r["point"] == pytest.approx(ma[metric] - mb[metric], abs=1e-12)
    if metric == "ocsai":
        # for an item-weighted base metric the point estimate is the plain mean of the item x seed differences
        pa = df[(df.method == "a") & (df.metric == metric)].set_index(["item_id", "seed"])["value"]
        pb = df[(df.method == "b") & (df.metric == metric)].set_index(["item_id", "seed"])["value"]
        assert r["point"] == pytest.approx(float((pa - pb).mean()), abs=1e-12)


def test_coverage_over_repeated_synthetic_datasets():
    """Nominal 95% intervals: over 150 synthetic datasets the true paired difference of HM
    (a derived, non-linear metric) must be covered at least ~90% of the time."""
    covered, total = 0, 150
    for k in range(total):
        rng = np.random.default_rng(1000 + k)
        df, truth = make_long(rng, n_items=30, true_diff=0.3, noise=0.3, return_truth=True)
        a = bc.method_arrays(df, "T", "aut", "m", "a")
        b = bc.method_arrays(df, "T", "aut", "m", "b")
        r = bc.paired_bootstrap(a, b, "aut", "hm", n_resamples=400, seed=0)
        ta, _ = aggregate_method(truth, "T", "aut", "m", "a")
        tb, _ = aggregate_method(truth, "T", "aut", "m", "b")
        true = ta["hm"] - tb["hm"]
        covered += r["ci_lo"] <= true <= r["ci_hi"]
    assert covered / total >= 0.90, covered / total


def test_identical_methods_ci_contains_zero_and_is_symmetric(rng):
    df = make_long(rng, n_items=40, identical=True)
    a = bc.method_arrays(df, "T", "aut", "m", "a")
    b = bc.method_arrays(df, "T", "aut", "m", "b")
    for metric in ["ocsai", "novelty", "usability", "hm", "semdiv"]:
        r = bc.paired_bootstrap(a, b, "aut", metric, n_resamples=500, seed=0)
        assert r["point"] == 0.0
        assert r["ci_lo"] <= 0.0 <= r["ci_hi"]
        assert abs(r["ci_lo"]) == pytest.approx(abs(r["ci_hi"]), abs=1e-12)   # degenerate at 0
        assert not r["excludes_zero"]


def test_pairing_and_seed_structure_preserved_in_every_draw(rng):
    """Instrument the statistic: for every draw, each drawn item contributes the same number of
    rows for method_a and method_b, and every drawn item carries its complete seed set."""
    df = make_long(rng, n_items=25, true_diff=0.2)
    a = bc.method_arrays(df, "T", "aut", "m", "a")
    b = bc.method_arrays(df, "T", "aut", "m", "b")
    seen = []
    orig = bc.statistic

    def spy(a_, b_, benchmark, metric, W, idx):
        seen.append(W.copy())
        return orig(a_, b_, benchmark, metric, W, idx)

    bc.statistic = spy
    try:
        bc.paired_bootstrap(a, b, "aut", "novelty", n_resamples=300, seed=0, chunk=100)
    finally:
        bc.statistic = orig
    W = np.concatenate([w for w in seen if w.shape[0] > 1])      # skip the point-estimate call
    assert W.shape == (300, 25)
    assert np.all(W.sum(axis=1) == 25)                              # 25 items drawn per draw
    # rows contributed by each item: count x (#seeds); identical for a and b because both are
    # indexed by the same W and share the item/seed grid
    na = W[:, :, None] * a.n_obs["novelty"].T[None, :, :]           # (draws, items, seeds)
    nb = W[:, :, None] * b.n_obs["novelty"].T[None, :, :]
    assert (na > 0).sum() == (nb > 0).sum()
    rows_a = (W[:, :, None] * np.ones_like(a.value["novelty"].T)[None]).sum(axis=(1, 2))
    rows_b = (W[:, :, None] * np.ones_like(b.value["novelty"].T)[None]).sum(axis=(1, 2))
    assert np.array_equal(rows_a, rows_b)
    # seeds kept together: each drawn item contributes exactly len(seeds) rows per copy
    assert np.all(rows_a == W.sum(axis=1) * len(a.seeds))


def test_partial_seed_set_is_rejected(rng):
    df = make_long(rng, n_items=10)
    df = df[~((df.method == "b") & (df.item_id == "item003") & (df.seed == 101))]
    with pytest.raises(AssertionError):
        bc.method_arrays(df, "T", "aut", "m", "b")


def test_determinism_byte_identical_csv(rng, tmp_path):
    df = make_long(rng, n_items=20, true_diff=0.3)
    df["table"] = "5"; df["benchmark"] = "aut"; df["model"] = "llama"
    df["method"] = df["method"].map({"a": "bd", "b": "baseline"})
    p = tmp_path / "scores.parquet"
    df.to_parquet(p, index=False)
    outs = []
    for k in range(2):
        out = tmp_path / f"ci{k}.csv"
        res = bc.run_all(df, n_resamples=500, seed=0, tables=["5"], comparisons={"5": [("bd", "baseline")]},
                         benchmarks=["aut"], models=["llama"])
        res.to_csv(out, index=False, float_format="%.6f")
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]


def test_p30_metric_path(rng):
    """LiveIdea flexibility uses the percentile path (index-based resampling)."""
    df = make_long(rng, n_items=50, benchmark="liveidea", true_diff=0.5,
                   metrics=["originality", "feasibility", "clarity", "fluency", "flex_kwmean"])
    a = bc.method_arrays(df, "T", "liveidea", "m", "a")
    b = bc.method_arrays(df, "T", "liveidea", "m", "b")
    r = bc.paired_bootstrap(a, b, "liveidea", "flex", n_resamples=500, seed=0)
    ma, _ = aggregate_method(df, "T", "liveidea", "m", "a")
    mb, _ = aggregate_method(df, "T", "liveidea", "m", "b")
    assert r["point"] == pytest.approx(ma["flex"] - mb["flex"], abs=1e-12)
    assert r["ci_lo"] <= 0.5 <= r["ci_hi"]
