"""Stage: per-use n-gram originality scoring for AUT replies.

Reads the canonical AUT replies CSV (``aut_brainstorm_select`` if configured,
else ``aut_infer``), explodes parsed uses into rows, looks up each use's
n-grams in a pre-built reference corpus, and writes ``originality_ngram_<N>``
columns to ``<run_dir>/aut_novelty/<method>_novelty.csv``.

Decoupled from ``aut_ocsai`` by design — the OCSAI stage is an external HTTP
call, this one is pure local computation. They can run in either order, both
can be skipped independently. Aggregate joins them by ``(method, item_id,
sample_idx, use_text)``.

Implements the originality component of:
  Padmakumar et al., "Measuring LLM Novelty As The Frontier Of Original
  And High-Quality Output", ICLR 2026.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from ..config import OriginalityConfig, RunConfig
from ..repro import RunContext, StageResult, hashed_input, hashed_output


# ---------------------------------------------------------------------------
# Path resolution: brainstorm-select output if present, else aut_infer
# ---------------------------------------------------------------------------


def aut_replies_dir(ctx: RunContext) -> Path:
    """The directory ``aut_novelty`` consumes (same rule as ``aut_ocsai``)."""
    if ctx.cfg.brainstorm_select is not None:
        return ctx.stage_dir("aut_brainstorm_select")
    return ctx.stage_dir("aut_infer")


def _replies_glob(ctx: RunContext) -> list[Path]:
    if ctx.cfg.brainstorm_select is not None:
        return sorted(aut_replies_dir(ctx).glob("*_selected_results.csv"))
    return sorted(aut_replies_dir(ctx).glob("*_results.csv"))


def _method_slug_from_path(p: Path) -> str:
    stem = p.stem
    for suffix in ("_selected_results", "_results"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# Row materialization — same shape as aut_ocsai._explode_uses to preserve
# joinability on (method, item_id, sample_idx, use_text).
# ---------------------------------------------------------------------------


def _explode_uses(replies_csv: Path) -> list[dict]:
    out: list[dict] = []
    with replies_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "use_text" in r and r.get("use_text"):
                out.append(r)
                continue
            try:
                uses = json.loads(r.get("parsed_response", "[]"))
            except json.JSONDecodeError:
                uses = []
            for use in uses:
                out.append({**r, "use_text": use})
    return out


# ---------------------------------------------------------------------------
# CSV columns — joinable to OCSAI/judge on (item_id, sample_idx, use_text).
# ---------------------------------------------------------------------------


NOVELTY_BASE_COLUMNS = [
    "item_id", "task_type", "domain", "split", "source_id", "sample_idx", "seed",
    "object", "method", "use_text", "num_tokens",
]


def _columns(orig_cfg: OriginalityConfig) -> list[str]:
    return NOVELTY_BASE_COLUMNS + [f"originality_ngram_{n}" for n in orig_cfg.n_values]


def _row(item_row: dict, num_tokens: int, scores: dict[int, float]) -> dict:
    base = {
        "item_id": item_row["item_id"],
        "task_type": item_row.get("task_type", "aut"),
        "domain": item_row.get("domain", ""),
        "split": item_row.get("split", "eval"),
        "source_id": item_row.get("source_id", item_row["item_id"]),
        "sample_idx": item_row.get("sample_idx", 0),
        "seed": item_row.get("seed", ""),
        "object": item_row.get("object", ""),
        "method": item_row.get("method", ""),
        "use_text": item_row.get("use_text", ""),
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
            "aut_novelty requires cfg.originality with ngram_index set."
        )
    orig_cfg = cfg.originality

    index = NgramCorpusIndex.load(orig_cfg.ngram_index)
    replies = _replies_glob(ctx)
    stage_dir = ctx.stage_dir("aut_novelty")
    stage_dir.mkdir(parents=True, exist_ok=True)
    columns = _columns(orig_cfg)

    outputs: list[dict] = []
    inputs: dict[str, dict] = {
        "ngram_index": hashed_input(orig_cfg.ngram_index),
    }
    n_rows_total = 0

    for path in replies:
        method = _method_slug_from_path(path)
        item_rows = _explode_uses(path)
        n_rows_total += len(item_rows)

        scored_rows: list[dict] = []
        for r in item_rows:
            text = r.get("use_text", "")
            tokens = tokenize_text(text)
            scores = compute_originality(text, index, orig_cfg.n_values)
            scored_rows.append(_row(r, len(tokens), scores))

        out_path = stage_dir / f"{method}_novelty.csv"
        _write_csv(scored_rows, out_path, columns)
        outputs.append(hashed_output(out_path, rows=len(scored_rows)))
        inputs[method] = hashed_input(path)

    return StageResult(
        name="aut_novelty",
        inputs=inputs,
        outputs=outputs,
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "reads_from": str(aut_replies_dir(ctx).name),
                "n_values": list(orig_cfg.n_values),
            },
            "n_scored_rows": n_rows_total,
        },
    )
