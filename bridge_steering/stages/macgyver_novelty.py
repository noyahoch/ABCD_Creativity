"""Stage: per-solution n-gram originality scoring for MacGyver replies.

Reads ``macgyver_infer/<method>_results.csv`` (one row per solution), looks
up each solution's n-grams in a pre-built reference corpus, and writes
``originality_ngram_<N>`` columns to
``<run_dir>/macgyver_novelty/<method>_novelty.csv``.

Mirrors :mod:`aut_novelty` but operates per-row (one solution per reply),
not per-use — MacGyver replies are single solutions, not lists.

Implements the originality component of:
  Padmakumar et al., "Measuring LLM Novelty As The Frontier Of Original
  And High-Quality Output", ICLR 2026.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from ..config import OriginalityConfig, RunConfig
from ..repro import RunContext, StageResult, hashed_input, hashed_output


# ---------------------------------------------------------------------------
# CSV columns — joinable to macgyver_judge on (item_id, sample_idx).
# ---------------------------------------------------------------------------


NOVELTY_BASE_COLUMNS = [
    "item_id", "task_type", "domain", "split", "source_id", "sample_idx", "seed",
    "problem_id", "method", "num_tokens",
]


def _columns(orig_cfg: OriginalityConfig) -> list[str]:
    return NOVELTY_BASE_COLUMNS + [f"originality_ngram_{n}" for n in orig_cfg.n_values]


def _row(reply_row: dict, num_tokens: int, scores: dict[int, float]) -> dict:
    base = {
        "item_id": reply_row["item_id"],
        "task_type": reply_row.get("task_type", "macgyver"),
        "domain": reply_row.get("domain", ""),
        "split": reply_row.get("split", "eval"),
        "source_id": reply_row.get("source_id", reply_row["item_id"]),
        "sample_idx": reply_row.get("sample_idx", 0),
        "seed": reply_row.get("seed", ""),
        "problem_id": reply_row.get("problem_id", reply_row.get("item_id", "")),
        "method": reply_row.get("method", ""),
        "num_tokens": num_tokens,
    }
    for n, val in scores.items():
        base[f"originality_ngram_{n}"] = val
    return base


def _write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(ctx: RunContext) -> StageResult:
    from bridge_steering.novelty import NgramCorpusIndex, compute_originality, tokenize_text

    cfg = ctx.cfg
    t0 = time.time()
    if cfg.originality is None or cfg.originality.ngram_index is None:
        raise ValueError(
            "macgyver_novelty requires cfg.originality with ngram_index set."
        )
    orig_cfg = cfg.originality

    index = NgramCorpusIndex.load(orig_cfg.ngram_index)
    infer_dir = ctx.stage_dir("macgyver_infer")
    stage_dir = ctx.stage_dir("macgyver_novelty")
    stage_dir.mkdir(parents=True, exist_ok=True)
    columns = _columns(orig_cfg)

    outputs: list[dict] = []
    inputs: dict[str, dict] = {
        "ngram_index": hashed_input(orig_cfg.ngram_index),
    }
    n_rows_total = 0

    for path in sorted(infer_dir.glob("*_results.csv")):
        method = path.stem[: -len("_results")]
        with path.open("r", encoding="utf-8") as f:
            reply_rows = list(csv.DictReader(f))
        n_rows_total += len(reply_rows)

        scored_rows: list[dict] = []
        for r in reply_rows:
            text = r.get("parsed_response") or r.get("raw_response", "")
            tokens = tokenize_text(text)
            scores = compute_originality(text, index, orig_cfg.n_values)
            scored_rows.append(_row(r, len(tokens), scores))

        out_path = stage_dir / f"{method}_novelty.csv"
        _write_csv(scored_rows, out_path, columns)
        outputs.append(hashed_output(out_path, rows=len(scored_rows)))
        inputs[method] = hashed_input(path)

    return StageResult(
        name="macgyver_novelty",
        inputs=inputs,
        outputs=outputs,
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "reads_from": "macgyver_infer",
                "n_values": list(orig_cfg.n_values),
            },
            "n_scored_rows": n_rows_total,
        },
    )
