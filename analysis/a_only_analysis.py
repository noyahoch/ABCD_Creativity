"""Add the A-only runs to scores_long, bootstrap A-only vs B-D, write a_only_ablation.md.

Methods added (table "5", model llama): ``a_only_raw`` (alpha=0.17, raw vector) and
``a_only_renorm`` (alpha=0.17, per-layer renormalised to ||v_{B-D}||). OCSAI was not run
(paid scorer), so the OCSAI column is reported as pending.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import bootstrap_ci as bc
from .aggregate import aggregate_method, fmt
from .build_scores_long import COLUMNS, assert_paired, load_aut_run, load_mac_run
from .registry import METRIC_LABEL, OUT_DIR, RESULTS_ROOT, SCORES, SEEDS, TABLE_METRICS

A_ONLY_METHODS = ("a_only_raw", "a_only_renorm")
RUN_ROOT = RESULTS_ROOT / "ablations" / "llama"
COMPARISONS = [("bd", "a_only_raw"), ("bd", "a_only_renorm"),
               ("a_only_raw", "baseline"), ("a_only_renorm", "baseline"), ("a_only_renorm", "a_only_raw")]


def run_dir(method: str, seed: int) -> Path:
    return RUN_ROOT / f"{method}_s{seed}"


def build_rows() -> list[dict]:
    rows = []
    for method in A_ONLY_METHODS:
        for seed in SEEDS:
            rd = run_dir(method, seed)
            aut, _ = load_aut_run(rd, require_ocsai=False)
            mac, _ = load_mac_run(rd)
            for bench, per in (("aut", aut), ("macgyver", mac)):
                for metric, d in per.items():
                    for iid, (v, n) in d.items():
                        rows.append(dict(table="5", benchmark=bench, model="llama", method=method, item_id=iid,
                                         seed=seed, metric=metric, value=v, n_obs=n, run_id=rd.name))
    return rows


def main() -> int:
    base = pd.read_parquet(SCORES)
    base = base[~base["method"].isin(A_ONLY_METHODS)]
    new = pd.DataFrame(build_rows(), columns=COLUMNS)
    df = pd.concat([base, new], ignore_index=True)
    assert_paired(df)
    df.to_parquet(SCORES, index=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    res = bc.run_all(df, tables=["5"], comparisons={"5": COMPARISONS}, benchmarks=["aut", "macgyver"], models=["llama"])
    res.to_csv(OUT_DIR / "ci_results_a_only.csv", index=False, float_format="%.6f")

    # Table-5-style rows: mean ± std over seeds
    L = ["# A-only reference ablation", "",
         "Llama-3.1-8B-Instruct, layers 14-19, alpha=0.17, 3 seeds (42/101/999; sampler RNG seeded), same judges "
         "(Qwen2.5-32B-Instruct-AWQ, T=0) and MacGyver n-gram index as Table 5. Vector: mean over the B completion "
         "tokens minus mean over the anchor (user-prompt) tokens, both from the single A+B teacher-forced pass, "
         "averaged over the 110 construction objects (`analysis/a_only_vector.py`). "
         "`raw` = that vector at alpha=0.17; `renorm` = rescaled per layer to ||v_{B-D}|| (scales 0.68-0.72). "
         "OCSAI was not run (paid scorer; needs the author's OK), so that column is pending.", "",
         "## Table-5-style rows (mean ± SD over seeds)", "",
         "| Variant | OCSAI | Novelty | Usability | HM | Sem. div. | MG Orig. | MG Quality | MG HM |", "|---|---|---|---|---|---|---|---|---|"]
    for method, label in (("baseline", "Baseline"), ("bd", "B−D (full)"), ("a_only_raw", "B−A (raw, α=0.17)"),
                          ("a_only_renorm", "B−A (renorm to ‖B−D‖, α=0.17)")):
        cells = []
        for bench in ("aut", "macgyver"):
            means, per_seed = aggregate_method(df, "5", bench, "llama", method)
            for m in TABLE_METRICS[bench]:
                if m in means and not np.isnan(means[m]):
                    sd = float(per_seed[m].std(ddof=1))
                    cells.append(f"{means[m]:.2f} ± {sd:.2f}")
                else:
                    cells.append("pending")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "## Paired item-level bootstrap (10,000 resamples, 95% percentile CI, seed 0)", "",
          "| benchmark | comparison | metric | Δ | 95% CI | excludes 0 |", "|---|---|---|---|---|---|"]
    for r in res.itertuples():
        L.append(f"| {r.benchmark} | {r.method_a} − {r.method_b} | {METRIC_LABEL[r.metric]} | {r.delta:+.3f} | "
                 f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {'yes' if r.excludes_zero else 'no'} |")
    stats = json.load(open(RUN_ROOT / "vectors" / "_manifest.json"))
    L += ["", "## Vector statistics (per layer)", "", "| layer | ‖v_A-only‖ | ‖v_B−D‖ | cos(A-only, B−D) | renorm scale |", "|---|---|---|---|---|"]
    for s in stats["stats"]:
        L.append(f"| {s['layer']} | {s['norm_v_a_only']} | {s['norm_v_bd']} | {s['cos_a_only_vs_bd']} | {s['renorm_scale']} |")
    L += ["", f"Anchor spans used: {stats['anchor_kinds']} (mean {stats['anchor_tokens_mean']:.1f} anchor tokens, "
          f"{stats['b_tokens_mean']:.1f} B tokens per object). B-side centroid vs the paper's stored centroids_B: "
          f"max |diff| ≤ {max(s['max_abs_diff_mu_B_vs_paper'] for s in stats['stats']):.3f} (bf16 storage)."]
    (OUT_DIR / "a_only_ablation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
