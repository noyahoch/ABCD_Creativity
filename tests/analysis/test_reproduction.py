"""Aggregating scores_long must reproduce every published number (tolerance 0.005).

Known exceptions (all < 0.01, see `python -m analysis.reproduction_check`):
  * cells whose reproduced value sits on a 2-decimal rounding boundary and matches the
    published value once rounded via a 3-decimal intermediate (how the table scripts print);
  * LiveIdeaBench HM (Tables 4/10) is printed as the harmonic mean of the already-rounded axis
    means (e.g. Mistral creative: HM3(7.16, 6.38, 8.22) = 7.176 -> 7.18, unrounded 7.1745);
  * Table 9, Falcon, Olson "Delta HM": the published +0.01 is inconsistent with the published
    HMs of the same row (0.87 - 0.85); the unrounded delta is +0.017.
"""
import pytest

from analysis.aggregate import aggregate_method, hm3
from analysis.registry import PUBLISHED, PUBLISHED_DELTA_HM, SCORES as _SCORES

SCORES = _SCORES
TOL, HARD_CAP = 0.005, 0.01
pytestmark = pytest.mark.skipif(not SCORES.exists(), reason="scores_long.parquet not built")


def rounds_via_3dp(x: float, published: float) -> bool:
    return round(float(f"{x:.3f}") + 1e-12, 2) == published or round(float(f"{x:.3f}") - 1e-12, 2) == published


@pytest.fixture(scope="module")
def df():
    import pandas as pd
    return pd.read_parquet(SCORES)


@pytest.mark.parametrize("key", sorted(PUBLISHED))
def test_published_means(df, key):
    table, bench, model, method = key
    means, _ = aggregate_method(df, table, bench, model, method)
    for metric, pub in PUBLISHED[key].items():
        gap = abs(means[metric] - pub)
        assert gap <= HARD_CAP, (key, metric, means[metric], pub)
        if bench == "liveidea" and metric == "hm":
            # Tables 4/10 print HM as the harmonic mean of the *rounded* published axis means
            p = PUBLISHED[key]
            assert abs(hm3(p["originality"], p["feasibility"], p["clarity"]) - pub) <= TOL
            assert abs(hm3(means["originality"], means["feasibility"], means["clarity"]) - means["hm"]) <= TOL
            continue
        assert gap <= TOL or rounds_via_3dp(means[metric], pub), (key, metric, means[metric], pub)


@pytest.mark.parametrize("key", sorted(PUBLISHED_DELTA_HM))
def test_published_delta_hm(df, key):
    model, method = key
    base, _ = aggregate_method(df, "3", "macgyver", model, "baseline")
    cur, _ = aggregate_method(df, "3", "macgyver", model, method)
    delta = cur["hm"] - base["hm"]
    gap = abs(delta - PUBLISHED_DELTA_HM[key])
    if key == ("falcon", "olson"):
        # published +0.01 is internally inconsistent (0.87 - 0.85); keep it visible, cap at 0.01
        assert gap <= HARD_CAP and delta == pytest.approx(0.0173, abs=5e-4)
    else:
        assert gap <= TOL, (key, delta, PUBLISHED_DELTA_HM[key])
