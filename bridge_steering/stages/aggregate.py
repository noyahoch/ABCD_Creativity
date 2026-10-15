"""Stage: aggregate per-method summary across the AUT and MacGyver stages.

Joins inference + ocsai + judge CSVs on ``(item_id, sample_idx)``, then
writes a wide cross-stage summary table. Deterministic — given the same
input CSVs, the output is byte-identical.

Inline diversity columns are added here when ``cfg.diversity`` is set
(semantic_diversity_mean_pairwise_cosine, ngram_diversity_self_bleu_<N>).
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from itertools import combinations

from ..config import DiversityConfig, RunConfig
from ..repro import RunContext, StageResult, hashed_input, hashed_output


# ---------------------------------------------------------------------------
# Diversity helpers (deterministic; semantic uses a sentence-transformer
# loaded once per stage run).
# ---------------------------------------------------------------------------


def ngram_diversity(uses: list[str], n: int) -> float:
    """Distinct n-gram ratio across the use set. Higher = more diverse.

    Returns 0.0 when no n-grams can be formed.
    """
    all_ngrams: list[tuple[str, ...]] = []
    for use in uses:
        tokens = (use or "").lower().split()
        if len(tokens) < n:
            continue
        all_ngrams.extend(zip(*(tokens[i:] for i in range(n))))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def semantic_diversity_mean_pairwise_cosine(uses: list[str], div_cfg: DiversityConfig) -> float | None:
    """Mean pairwise cosine *distance* of sentence-transformer embeddings
    for a single response's uses (within-item).

    Returns 0.0 when fewer than 2 uses; ``None`` if the embedding model
    can't be loaded (sentence-transformers missing).
    """
    if len(uses) < 2:
        return 0.0
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    device = "cpu" if div_cfg.force_cpu else None
    model = SentenceTransformer(div_cfg.embedding_model, device=device)
    embs = model.encode(uses, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embs / norms
    cos = normed @ normed.T
    n = len(uses)
    total, count = 0.0, 0
    for i, j in combinations(range(n), 2):
        total += 1.0 - cos[i, j]
        count += 1
    return float(total / count) if count else 0.0


_NGRAM_DIVERSITY_N_VALUES = (1, 2, 3)


def diversity_columns(div_cfg: DiversityConfig | None) -> list[str]:
    """Diversity column(s) emitted in the AUT block.

    Only ``aut_semantic_diversity_mean_pairwise_cosine`` is in the default
    output to match the paper-style table the team uses (single "Sem. div."
    column). The within-item n-gram distinct ratios are still computed by
    :func:`_compute_within_item_diversity` for callers who want them, but
    they're not part of the summary CSV.
    """
    if div_cfg is None:
        return []
    return ["aut_semantic_diversity_mean_pairwise_cosine"]


def _compute_within_item_diversity(
    uses_by_item: dict[str, list[str]],
    div_cfg: DiversityConfig,
) -> dict[str, float | None]:
    """Diversity computed WITHIN each item, then averaged across items.

    Pooling all uses from all objects together (e.g. "brick" uses next to
    "rubber band" uses) inflates pairwise distance — the metric is supposed
    to capture *creative spread within one prompt*, not how different the
    objects are. The old ``scripts/compute_response_diversity.py`` worked
    per-response and averaged; we replicate that here.
    """
    per_item_sem: list[float] = []
    per_item_ng: dict[int, list[float]] = {n: [] for n in _NGRAM_DIVERSITY_N_VALUES}

    # Semantic side: encode all uses in one batch (much faster than per-item
    # model.encode calls), then slice into per-item embedding arrays.
    item_ids = sorted(k for k, v in uses_by_item.items() if len(v) >= 2)
    if item_ids:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            np = None  # type: ignore[assignment]
            SentenceTransformer = None  # type: ignore[assignment]

        if np is not None and SentenceTransformer is not None:
            device = "cpu" if div_cfg.force_cpu else None
            model = SentenceTransformer(div_cfg.embedding_model, device=device)
            flat: list[str] = []
            offsets: dict[str, tuple[int, int]] = {}
            for iid in item_ids:
                start = len(flat)
                flat.extend(uses_by_item[iid])
                offsets[iid] = (start, len(flat))
            embs = model.encode(flat, convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normed = embs / norms

            for iid in item_ids:
                s, e = offsets[iid]
                block = normed[s:e]
                cos = block @ block.T
                k = e - s
                total, count = 0.0, 0
                for i, j in combinations(range(k), 2):
                    total += 1.0 - cos[i, j]
                    count += 1
                if count:
                    per_item_sem.append(float(total / count))

    # N-gram side: pure-Python, per-item.
    for iid, uses in uses_by_item.items():
        if len(uses) < 2:
            continue
        for n in _NGRAM_DIVERSITY_N_VALUES:
            per_item_ng[n].append(ngram_diversity(uses, n))

    def _mean_or_none(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    out: dict[str, float | None] = {
        "aut_semantic_diversity_mean_pairwise_cosine": _mean_or_none(per_item_sem),
    }
    for n in _NGRAM_DIVERSITY_N_VALUES:
        out[f"aut_ngram_diversity_distinct_ratio_{n}"] = _mean_or_none(per_item_ng[n])
    return out


# ---------------------------------------------------------------------------
# Per-method summary row schema
# ---------------------------------------------------------------------------


# One row per method. Column order matches the paper-style comparison
# tables:
#   * AUT: counts → OCSAI, judge novelty/usability, HM(N,U), diversity.
#     AUT does NOT include n-gram originality — OCSAI is the originality
#     signal on the AUT side per the paper.
#   * MacGyver: counts → 4-gram originality, judge quality, HM.
#     MacGyver reports a single 4-gram originality column; the per-use CSVs
#     still carry n=5 and n=6 for inspection but they aren't summarised.
AUT_SUMMARY_COLUMNS = [
    "aut_n_replies", "aut_n_uses",
    "aut_mean_ocsai_originality",
    "aut_mean_novelty", "aut_mean_usability",
    "aut_hm_novelty_usability",          # harmonic mean of novelty + usability
]
MAC_SUMMARY_COLUMNS = [
    "mac_n_replies",
    "mac_mean_originality_ngram_4",      # paper-reported 4-gram originality
    "mac_mean_quality_score",
    "mac_hm_originality_quality",        # harmonic mean of ngram-4 originality + quality/5
]


def _summary_columns(cfg: RunConfig) -> list[str]:
    return (
        ["method"]
        + AUT_SUMMARY_COLUMNS
        + diversity_columns(cfg.diversity)
        + MAC_SUMMARY_COLUMNS
        + ["ngram_n_values"]
    )


def _safe_mean(xs: list[float | None]) -> float | None:
    """Mean over the non-None, non-NaN entries (NaN returned by e.g. OCSAI
    when its scorer can't grade a specific use — count those as missing,
    not as a value that drags the mean to NaN)."""
    xs = [x for x in xs
          if x is not None
          and not (isinstance(x, float) and math.isnan(x))]
    return mean(xs) if xs else None


def _ocsai_safe_mean(xs: list[float | None]) -> float | None:
    """Mean over OCSAI scores after filtering NaN and out-of-range outliers.

    OCSAI's documented scale is 1.0–5.0. The openscoring.du.edu service
    occasionally returns junk (we've seen 412.7) that destroys the mean.
    Filter to the documented range so we report something interpretable.
    """
    xs = [
        x for x in xs
        if x is not None
        and not (isinstance(x, float) and math.isnan(x))
        and 1.0 <= float(x) <= 5.0
    ]
    return mean(xs) if xs else None


def _harmonic_mean(a: float | None, b: float | None) -> float | None:
    """Standard harmonic mean ``2ab/(a+b)``. ``None`` if either input is
    missing or non-positive (HM is undefined for ≤ 0)."""
    if a is None or b is None:
        return None
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if math.isnan(a) or math.isnan(b) or a <= 0 or b <= 0:
        return None
    return 2.0 * a * b / (a + b)


def _load_csv(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _by_method(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r.get("method", "")].append(r)
    return out


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(ctx: RunContext) -> StageResult:
    cfg = ctx.cfg
    t0 = time.time()

    run_dir = ctx.run_dir

    # --- Collect available stage outputs --------------------------------
    inputs: dict[str, dict] = {}
    per_method: dict[str, dict] = {}  # method → partial summary row

    def _row(method: str) -> dict:
        return per_method.setdefault(method, {"method": method})

    # AUT path
    aut_infer_dir    = run_dir / "aut_infer"
    aut_select_dir   = run_dir / "aut_brainstorm_select"
    aut_ocsai_dir    = run_dir / "aut_ocsai"
    aut_novelty_dir  = run_dir / "aut_novelty"
    aut_judge_dir    = run_dir / "aut_judge"

    aut_replies_dir = aut_select_dir if aut_select_dir.is_dir() and any(aut_select_dir.glob("*_selected_results.csv")) else aut_infer_dir
    aut_glob = "*_selected_results.csv" if aut_replies_dir == aut_select_dir else "*_results.csv"

    for p in sorted(aut_replies_dir.glob(aut_glob)):
        method = p.stem.replace("_selected_results", "").replace("_results", "")
        inputs[f"aut_infer/{method}"] = hashed_input(p)

        replies = _load_csv(p)
        # The inference CSV has one row per (item, sample); use-level data
        # comes from ocsai / novelty / judge CSVs which are one row per use.
        ocsai_p   = aut_ocsai_dir   / f"{method}_ocsai_scores.csv"
        novelty_p = aut_novelty_dir / f"{method}_novelty.csv"
        judge_p   = aut_judge_dir   / f"{method}_judge_scores.csv"
        ocsai_rows   = _load_csv(ocsai_p)
        novelty_rows = _load_csv(novelty_p)
        judge_rows   = _load_csv(judge_p)
        if ocsai_p.is_file():
            inputs[f"aut_ocsai/{method}"] = hashed_input(ocsai_p)
        if novelty_p.is_file():
            inputs[f"aut_novelty/{method}"] = hashed_input(novelty_p)
        if judge_p.is_file():
            inputs[f"aut_judge/{method}"] = hashed_input(judge_p)

        # Per-use data lives in the judge / novelty CSVs (one row per use);
        # OCSAI rows are per-reply now (one row per (item, sample)), so they
        # don't carry parsed uses — skip them in the per-use fallback chain.
        per_use_rows = judge_rows or novelty_rows
        use_texts = [r.get("use_text", "") for r in per_use_rows if r.get("use_text")]

        row = _row(method)
        mean_novelty   = _safe_mean([_to_float(r.get("novelty")) for r in judge_rows])
        mean_usability = _safe_mean([_to_float(r.get("usability")) for r in judge_rows])
        row.update({
            "aut_n_replies": len(replies),
            "aut_n_uses": len(use_texts),
            "aut_mean_ocsai_originality":   _ocsai_safe_mean([_to_float(r.get("ocsai_originality")) for r in ocsai_rows]),
            "aut_mean_novelty":             mean_novelty,
            "aut_mean_usability":           mean_usability,
            "aut_hm_novelty_usability":     _harmonic_mean(mean_novelty, mean_usability),
            "aut_mean_originality_ngram_4": _safe_mean([_to_float(r.get("originality_ngram_4")) for r in novelty_rows]),
            "aut_mean_originality_ngram_5": _safe_mean([_to_float(r.get("originality_ngram_5")) for r in novelty_rows]),
            "aut_mean_originality_ngram_6": _safe_mean([_to_float(r.get("originality_ngram_6")) for r in novelty_rows]),
            "ngram_n_values":               json.dumps(cfg.originality.n_values) if cfg.originality else "",
        })
        # Within-item diversity (group uses by item_id, average per-item scores).
        if cfg.diversity is not None:
            uses_by_item: dict[str, list[str]] = defaultdict(list)
            for r in per_use_rows:
                iid = r.get("item_id") or ""
                txt = r.get("use_text") or ""
                if iid and txt:
                    uses_by_item[iid].append(txt)
            row.update(_compute_within_item_diversity(dict(uses_by_item), cfg.diversity))

    # MacGyver path
    mac_infer_dir   = run_dir / "macgyver_infer"
    mac_novelty_dir = run_dir / "macgyver_novelty"
    mac_judge_dir   = run_dir / "macgyver_judge"
    for p in sorted(mac_infer_dir.glob("*_results.csv")):
        method = p.stem.replace("_results", "")
        inputs[f"macgyver_infer/{method}"] = hashed_input(p)
        replies = _load_csv(p)
        novelty_p = mac_novelty_dir / f"{method}_novelty.csv"
        judge_p   = mac_judge_dir   / f"{method}_judge_scores.csv"
        if novelty_p.is_file():
            inputs[f"macgyver_novelty/{method}"] = hashed_input(novelty_p)
        if judge_p.is_file():
            inputs[f"macgyver_judge/{method}"] = hashed_input(judge_p)
        novelty_rows = _load_csv(novelty_p)
        judge_rows   = _load_csv(judge_p)

        row = _row(method)
        mac_orig_4    = _safe_mean([_to_float(r.get("originality_ngram_4")) for r in novelty_rows])
        mac_quality   = _safe_mean([_to_float(r.get("quality_score")) for r in judge_rows])
        # HM combines 4-gram originality (already in [0,1]) with quality on
        # its native 0–5 scale normalised to [0,1] (rubric max = 5), so the
        # two metrics live on the same axis when the harmonic mean is taken.
        mac_quality_unit = mac_quality / 5.0 if mac_quality is not None else None
        row.update({
            "mac_n_replies": len(replies),
            "mac_mean_originality_ngram_4": mac_orig_4,
            "mac_mean_originality_ngram_5": _safe_mean([_to_float(r.get("originality_ngram_5")) for r in novelty_rows]),
            "mac_mean_originality_ngram_6": _safe_mean([_to_float(r.get("originality_ngram_6")) for r in novelty_rows]),
            "mac_mean_quality_score":       mac_quality,
            "mac_hm_originality_quality":   _harmonic_mean(mac_orig_4, mac_quality_unit),
        })
        if cfg.originality is not None and "ngram_n_values" not in row:
            row["ngram_n_values"] = json.dumps(cfg.originality.n_values)

    summary_rows = [per_method[m] for m in sorted(per_method)]

    # --- Write summary ---------------------------------------------------
    stage_dir = ctx.stage_dir("aggregate")
    stage_dir.mkdir(parents=True, exist_ok=True)
    out = stage_dir / "cross_stage_summary.csv"
    columns = _summary_columns(cfg)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)

    return StageResult(
        name="aggregate",
        inputs=inputs,
        outputs=[hashed_output(out, rows=len(summary_rows))],
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "diversity_enabled": cfg.diversity is not None,
                "originality_enabled": cfg.originality is not None,
            },
            "methods_summarized": [r["method"] for r in summary_rows],
        },
    )
