"""Build analysis/scores_long.parquet (one row per table x model x method x item x seed x metric).

One row per (table, benchmark, model, method, item_id, seed, metric) with the per-item
value and the number of underlying observations ``n_obs`` (judged uses for the AUT
judge metrics; 1 for per-reply / per-problem / per-keyword metrics; 0 when the
pipeline dropped the observation, in which case ``value`` is NaN and the row is kept
so that every item carries its full seed set).

Sources (read-only):
  * AUT / MacGyver: per-run stage CSVs under results/<experiment>/<model>/<run>/
    (aut_ocsai, aut_judge, macgyver_novelty, macgyver_judge). Semantic diversity is
    recomputed per item from the judge CSV's use_text with all-MiniLM-L6-v2, replicating
    bridge_steering/stages/aggregate.py::_compute_within_item_diversity (the pipeline only
    stored the per-run mean). Every run is checked against its own
    aggregate/cross_stage_summary.csv.
  * LiveIdeaBench: liveideabench/data/ideabench.db (o3-mini critic rows),
    data/idea_hash_o3.json (fluency pair scores) and data/data.parquet (flexibility),
    replicating liveideabench/table_chosen_o3_allseeds.py.
"""
from __future__ import annotations

import glob
import hashlib
import itertools
import json
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import (BASE_METRICS, LIVEIDEA_ARMS, LIVEIDEA_CRITIC, LIVEIDEA_DB,
                       LIVEIDEA_FLUENCY_FILE, LIVEIDEA_KEYWORDS, LIVEIDEA_PARQUET,
                       LIVEIDEA_SEED_LABELS, OUT_DIR, PIPELINE_RUNS, SCORES, SEEDS, run_dir)

COLUMNS = ["table", "benchmark", "model", "method", "item_id", "seed", "metric", "value", "n_obs", "run_id"]


def _one(rd: Path, pattern: str) -> Path:
    hits = sorted(glob.glob(str(rd / pattern)))
    if len(hits) != 1:
        raise FileNotFoundError(f"{rd}/{pattern}: {len(hits)} matches")
    return Path(hits[0])


# ---------------------------------------------------------------------------
# Semantic diversity (replica of bridge_steering/stages/aggregate.py)
# ---------------------------------------------------------------------------
_ST_MODEL = None


def _st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


def semantic_diversity_per_item(uses_by_item: dict[str, list[str]]) -> dict[str, float]:
    item_ids = sorted(k for k, v in uses_by_item.items() if len(v) >= 2)
    if not item_ids:
        return {}
    flat: list[str] = []
    offsets: dict[str, tuple[int, int]] = {}
    for iid in item_ids:
        start = len(flat)
        flat.extend(uses_by_item[iid])
        offsets[iid] = (start, len(flat))
    embs = _st_model().encode(flat, convert_to_numpy=True, show_progress_bar=False)
    norms = np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8)
    normed = embs / norms
    out = {}
    for iid in item_ids:
        s, e = offsets[iid]
        block = normed[s:e]
        cos = block @ block.T
        total, count = 0.0, 0
        for i, j in combinations(range(e - s), 2):
            total += 1.0 - cos[i, j]
            count += 1
        out[iid] = float(total / count)
    return out


# ---------------------------------------------------------------------------
# AUT / MacGyver per run
# ---------------------------------------------------------------------------

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_aut_run(rd: Path, require_ocsai: bool = True) -> tuple[dict[str, dict[str, tuple[float, int]]], dict[str, float]]:
    """metric -> {item_id: (value, n_obs)}; plus the per-run aggregates for the self-check."""
    inf = pd.read_csv(_one(rd, "aut_infer/*_results.csv"))
    items = sorted(inf["item_id"].astype(str).unique())
    assert len(items) == len(inf), "expected one reply per item"
    if require_ocsai or glob.glob(str(rd / "aut_ocsai/*_ocsai_scores.csv")):
        O = pd.read_csv(_one(rd, "aut_ocsai/*_ocsai_scores.csv"))
    else:   # OCSAI not run: keep the column as missing
        O = pd.DataFrame({"item_id": [], "ocsai_originality": []})
    J = pd.read_csv(_one(rd, "aut_judge/*_judge_scores.csv"))
    out: dict[str, dict[str, tuple[float, int]]] = {m: {} for m in BASE_METRICS["aut"]}

    oc = _num(O["ocsai_originality"])
    ok = oc.between(1.0, 5.0)                      # _ocsai_safe_mean filter
    per = {}
    for iid, v, k in zip(O["item_id"].astype(str), oc, ok):
        assert iid not in per, "one OCSAI row per reply expected"
        per[iid] = (float(v) if k else np.nan, int(k))
    for iid in items:
        out["ocsai"][iid] = per.get(iid, (np.nan, 0))

    for metric in ("novelty", "usability"):
        v = _num(J[metric])
        g = pd.DataFrame({"item_id": J["item_id"].astype(str), "v": v}).dropna().groupby("item_id")["v"]
        mean, cnt = g.mean(), g.size()
        for iid in items:
            out[metric][iid] = (float(mean[iid]), int(cnt[iid])) if iid in mean.index else (np.nan, 0)

    uses_by_item: dict[str, list[str]] = {}
    for iid, txt in zip(J["item_id"].astype(str), J["use_text"]):
        if iid and isinstance(txt, str) and txt:
            uses_by_item.setdefault(iid, []).append(txt)
    sem = semantic_diversity_per_item(uses_by_item)
    for iid in items:
        out["semdiv"][iid] = (sem[iid], 1) if iid in sem else (np.nan, 0)

    agg = {
        "ocsai": float(oc[ok].mean()) if ok.any() else float("nan"),
        "novelty": float(_num(J["novelty"]).dropna().mean()),
        "usability": float(_num(J["usability"]).dropna().mean()),
        "semdiv": float(np.mean(list(sem.values()))),
    }
    agg["hm"] = 2 * agg["novelty"] * agg["usability"] / (agg["novelty"] + agg["usability"])
    return out, agg


def load_mac_run(rd: Path) -> tuple[dict[str, dict[str, tuple[float, int]]], dict[str, float]]:
    inf = pd.read_csv(_one(rd, "macgyver_infer/*_results.csv"))
    items = sorted(inf["item_id"].astype(str).unique())
    assert len(items) == len(inf)
    N = pd.read_csv(_one(rd, "macgyver_novelty/*_novelty.csv"))
    Q = pd.read_csv(_one(rd, "macgyver_judge/*_judge_scores.csv"))
    out: dict[str, dict[str, tuple[float, int]]] = {m: {} for m in BASE_METRICS["macgyver"]}
    o4 = dict(zip(N["item_id"].astype(str), _num(N["originality_ngram_4"])))
    q = dict(zip(Q["item_id"].astype(str), _num(Q["quality_score"])))
    assert len(o4) == len(N) and len(q) == len(Q), "duplicate MacGyver rows"
    for iid in items:
        v = o4.get(iid, np.nan)
        out["orig4"][iid] = (float(v), 0 if np.isnan(v) else 1)
        v = q.get(iid, np.nan)
        out["quality"][iid] = (float(v), 0 if np.isnan(v) else 1)
    agg = {"orig4": float(np.nanmean(list(o4.values()))), "quality": float(np.nanmean(list(q.values())))}
    agg["hm"] = 2 * agg["orig4"] * (agg["quality"] / 5) / (agg["orig4"] + agg["quality"] / 5)
    return out, agg


def _summary_row(rd: Path, stem_metric_cols: list[str]) -> dict[str, float]:
    s = pd.read_csv(rd / "aggregate" / "cross_stage_summary.csv")
    live = s[s[stem_metric_cols].notna().any(axis=1)]
    nb = live[live["method"] != "baseline"]
    row = (nb if len(nb) else live).iloc[0]
    return row


def check_against_summary(rd: Path, bench: str, agg: dict[str, float], log: list[str]) -> None:
    if bench == "aut":
        cols = {"ocsai": "aut_mean_ocsai_originality", "novelty": "aut_mean_novelty",
                "usability": "aut_mean_usability", "hm": "aut_hm_novelty_usability",
                "semdiv": "aut_semantic_diversity_mean_pairwise_cosine"}
    else:
        cols = {"orig4": "mac_mean_originality_ngram_4", "quality": "mac_mean_quality_score",
                "hm": "mac_hm_originality_quality"}
    row = _summary_row(rd, list(cols.values()))
    for m, c in cols.items():
        ref = float(row[c])
        tol = 1e-4 if m == "semdiv" else 1e-6
        diff = abs(agg[m] - ref)
        status = "ok" if diff <= tol else "MISMATCH"
        log.append(f"{rd.name:62s} {bench:8s} {m:9s} ours={agg[m]:.6f} pipeline={ref:.6f} diff={diff:.2e} {status}")
        if diff > tol:
            raise AssertionError(log[-1])


def build_pipeline_rows(log: list[str]) -> list[dict]:
    rows: list[dict] = []
    for (table, bench, model), methods in sorted(PIPELINE_RUNS.items()):
        for method, pattern in methods.items():
            for seed in SEEDS:
                rd = run_dir(model, pattern, seed)
                per, agg = (load_aut_run if bench == "aut" else load_mac_run)(rd)
                check_against_summary(rd, bench, agg, log)
                for metric, d in per.items():
                    for iid, (v, n) in d.items():
                        rows.append(dict(table=table, benchmark=bench, model=model, method=method,
                                         item_id=iid, seed=seed, metric=metric, value=v, n_obs=n,
                                         run_id=rd.name))
    return rows


# ---------------------------------------------------------------------------
# LiveIdeaBench (replica of liveideabench/table_chosen_o3_allseeds.py)
# ---------------------------------------------------------------------------

def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _db():
    # immutable=1: no locking / WAL handshake (the DB is static since 2026-08-02; sqlite locks
    # over the network filesystem stall both this reader and the repo's own table script).
    return sqlite3.connect(f"file:{LIVEIDEA_DB}?mode=ro&immutable=1", uri=True)


def liveidea_quality(arm: str, kws: set[str]) -> dict[str, dict[str, tuple[float, int]]]:
    """axis -> {kw: (mean over all (idea x o3-critic) rows, n rows)} (the script's qual())."""
    con = _db()
    per: dict[str, dict[str, list[float]]] = {a: {} for a in ("originality", "feasibility", "clarity")}
    for kw, ps, err in con.execute(
            "SELECT keywords, parsed_scores, error FROM results WHERE idea_model=? AND critic_model=?",
            (arm, LIVEIDEA_CRITIC)):
        if kw not in kws or err or not ps:
            continue
        try:
            s = json.loads(ps)
            vals = {a: float(s[a]) for a in per}
        except Exception:
            continue
        for a, v in vals.items():
            per[a].setdefault(kw, []).append(v)
    con.close()
    return {a: {k: (float(np.mean(v)), len(v)) for k, v in d.items()} for a, d in per.items()}


def liveidea_fluency(arm: str, kws: set[str], p2s: dict[frozenset, float]) -> tuple[dict[str, tuple[float, int]], int]:
    """kw -> (pair score, n scored pairs). The script takes the first scored pair it meets while
    iterating a Python set; with exactly two ideas per keyword that is the unique pair. We iterate
    in sorted-hash order for determinism and report how many keywords had more than one scored pair."""
    con = _db()
    ideas: dict[str, set[str]] = {}
    for kw, idea in con.execute("SELECT DISTINCT keywords, idea FROM results WHERE idea_model=?", (arm,)):
        if kw in kws:
            ideas.setdefault(kw, set()).add(idea)
    con.close()
    out, ambiguous = {}, 0
    for kw, u in ideas.items():
        hs = sorted(stable_hash(x) for x in u)
        scored = [p2s[frozenset(p)] for p in itertools.combinations(hs, 2) if frozenset(p) in p2s]
        if scored:
            out[kw] = (float(scored[0]), len(scored))
            ambiguous += len(scored) > 1
    return out, ambiguous


def liveidea_flex_frame(kws: set[str], h2s: dict[str, float]) -> pd.DataFrame:
    df = pd.read_parquet(LIVEIDEA_PARQUET, columns=["keywords", "idea_model", "critic_model", "idea",
                                                     "originality", "feasibility", "clarity"])
    df = df[(df["critic_model"] == LIVEIDEA_CRITIC) & (df["keywords"].isin(kws))]
    df = df[df["idea"].apply(lambda x: 100 < len(str(x)) < 2500)]
    df = df[df["idea"].apply(lambda x: len(str(x).split()) < 200)]
    df = df.copy()
    df["flue"] = df["idea"].apply(stable_hash).map(h2s)
    return df


def liveidea_flex_kwmean(df: pd.DataFrame, arm: str) -> dict[str, tuple[float, int]]:
    v = df[df["idea_model"] == arm]
    md = (v.groupby("keywords").agg({a: "mean" for a in ["feasibility", "originality", "flue", "clarity"]})
           .dropna())
    return {kw: (float(x), 1) for kw, x in zip(md.index, md.values.mean(axis=1))}


def build_liveidea_rows(log: list[str]) -> list[dict]:
    kws = {l.strip() for l in open(LIVEIDEA_KEYWORDS) if l.strip()}
    pairs = json.load(open(LIVEIDEA_FLUENCY_FILE))
    p2s = {frozenset((r["hash_A"], r["hash_B"])): r["score"] for r in pairs}
    h2s: dict[str, float] = {}
    for r in pairs:
        h2s[r["hash_A"]] = r["score"]
        h2s[r["hash_B"]] = r["score"]
    flexdf = liveidea_flex_frame(kws, h2s)
    rows: list[dict] = []
    models = sorted({m for m, _ in LIVEIDEA_ARMS})
    for model in models:
        per_arm = {}
        for (m, method), arms in LIVEIDEA_ARMS.items():
            if m != model:
                continue
            for seed, arm in zip(LIVEIDEA_SEED_LABELS, arms):
                qual = liveidea_quality(arm, kws)
                flu, amb = liveidea_fluency(arm, kws, p2s)
                flex = liveidea_flex_kwmean(flexdf, arm)
                per_arm[(method, seed, arm)] = {**qual, "fluency": flu, "flex_kwmean": flex}
                log.append(f"liveidea {model:8s} {method:9s} seed={seed} arm={arm:45s} "
                           f"kw_quality={len(qual['originality'])} kw_fluency={len(flu)} "
                           f"(>1 scored pair: {amb}) kw_flex={len(flex)}")
        # item set = keywords with quality scores in ANY arm of this model (the paper averages each
        # arm over its own scored keywords; the only hole is 5 keywords of llama baseline arm 1).
        # Missing cells are kept as value=NaN / n_obs=0 rows, exactly like unparseable judge rows.
        common = set.union(*[set(d["originality"]) for d in per_arm.values()])
        n_common = len(set.intersection(*[set(d["originality"]) for d in per_arm.values()]))
        log.append(f"liveidea {model:8s} keyword set = {len(common)} of {len(kws)} "
                   f"(scored in every arm: {n_common})")
        for (method, seed, arm), d in per_arm.items():
            for metric in BASE_METRICS["liveidea"]:
                dm = d[metric]
                for kw in sorted(common):
                    v, n = dm.get(kw, (np.nan, 0))
                    rows.append(dict(table="4", benchmark="liveidea", model=model, method=method,
                                     item_id=kw, seed=seed, metric=metric, value=v, n_obs=n, run_id=arm))
    return rows


# ---------------------------------------------------------------------------
# Assertions + output
# ---------------------------------------------------------------------------

def assert_paired(df: pd.DataFrame) -> None:
    for (t, b, m), g in df.groupby(["table", "benchmark", "model"]):
        ref = None
        for method, gm in g.groupby("method"):
            for metric, gmm in gm.groupby("metric"):
                key = (frozenset(gmm["item_id"]), frozenset(gmm["seed"]))
                assert not gmm.duplicated(["item_id", "seed"]).any(), (t, b, m, method, metric, "dup rows")
                assert len(gmm) == len(key[0]) * len(key[1]), (t, b, m, method, metric, "incomplete item x seed grid")
                if ref is None:
                    ref = key
                assert key == ref, (t, b, m, method, metric, "item/seed set differs from other methods")


def main(skip_liveidea: bool = False) -> pd.DataFrame:
    log: list[str] = []
    rows = build_pipeline_rows(log)
    if not skip_liveidea:
        rows += build_liveidea_rows(log)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["seed"] = df["seed"].astype(int)
    df["n_obs"] = df["n_obs"].astype(int)
    assert_paired(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SCORES, index=False)
    df.to_csv(OUT_DIR / "scores_long.csv", index=False)
    (OUT_DIR / "build_scores_long.log").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"wrote {len(df)} rows -> {SCORES}")
    return df


if __name__ == "__main__":
    main(skip_liveidea="--skip-liveidea" in sys.argv)
