"""Stage: brainstorm-then-select baseline (Summers-Stay et al. 2022).

Phase 1 (brainstorm) is ``aut_infer`` with sampling (``temperature > 0``,
``num_inferences > 1``). This stage is Phase 2 (select): for each use the
SAME model produces a free-form *advantages* + *drawbacks* completion and
then yes/no log-prob probes for *utility* and *originality* (paper Tables
2 & 4). A strategy then decides which uses to keep.

Four strategies (see ``config.py``):

* ``ThresholdStrategy`` — keep if utility/originality confidence exceeds
  an absolute threshold; ``combine`` decides AND vs OR when both axes set.
* ``TopFractionStrategy`` — rank by one axis with optional tiebreaker,
  keep top K% across all rows.
* ``TopFractionIntersectionStrategy`` — top K% on each axis separately,
  keep the intersection (final count can be smaller than K).
* ``TopKPerItemStrategy`` — per-``item_id`` Top-K by combined score, so
  every object emits exactly ``k_per_item`` uses (modulo Phase-1 supply).
  This is the default for fair comparison with other 8-uses-per-object
  cells in the pipeline.

The output CSV mirrors ``aut_infer`` plus the score and decision columns.
Downstream ``aut_ocsai`` / ``aut_judge`` / ``aut_novelty`` read this CSV
when brainstorm-select is configured.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import (
    BrainstormSelectConfig, RunConfig,
    SelectStrategy, ThresholdStrategy,
    TopFractionStrategy, TopFractionIntersectionStrategy,
    TopKPerItemStrategy,
)
from ..prompts.aut import (
    AUT_BS_ADVANTAGES_V1, AUT_BS_DRAWBACKS_V1,
    AUT_BS_UTIL_EVAL_V1, AUT_BS_ORIG_EVAL_V1,
)
from ..repro import RunContext, StageResult, hashed_input, hashed_output


# ---------------------------------------------------------------------------
# Pure-Python strategy selectors (no model needed — unit-tested)
# ---------------------------------------------------------------------------


def _select_threshold(rows: list[dict], s: ThresholdStrategy) -> list[dict]:
    """Keep rows whose utility/originality confidence beats per-axis thresholds.

    Sets ``kept`` and ``keep_reason`` on every row.
    """
    u_thr = s.utility_threshold
    o_thr = s.originality_threshold
    parts = []
    if u_thr is not None:
        parts.append(f"u>={u_thr}")
    if o_thr is not None:
        parts.append(f"o>={o_thr}")
    sep = f" {s.combine} "
    reason_template = sep.join(parts) if parts else "no-op"

    out = []
    for r in rows:
        r = dict(r)  # don't mutate caller's row
        u_pass = (u_thr is None) or (r["utility_conf"] >= u_thr)
        o_pass = (o_thr is None) or (r["originality_conf"] >= o_thr)
        if u_thr is not None and o_thr is not None:
            kept = (u_pass and o_pass) if s.combine == "and" else (u_pass or o_pass)
        else:
            kept = u_pass and o_pass
        r["kept"] = kept
        r["keep_reason"] = reason_template if kept else "below_threshold"
        out.append(r)
    return out


def _select_top_fraction(rows: list[dict], s: TopFractionStrategy) -> list[dict]:
    """Rank by ``rank_by`` (with optional tiebreaker), keep top K%."""
    if not rows:
        return []
    n = len(rows)
    k = max(1, min(n, round(n * s.keep_fraction)))

    if s.tiebreaker == "none":
        key = lambda r: -r[f"{s.rank_by}_conf"]
    else:
        key = lambda r: (-r[f"{s.rank_by}_conf"], -r[f"{s.tiebreaker}_conf"])
    order = sorted(range(n), key=lambda i: key(rows[i]))
    kept_set = set(order[:k])
    reason = (
        f"top{int(s.keep_fraction*100)}_rank_{s.rank_by}"
        + (f"_tb_{s.tiebreaker}" if s.tiebreaker != "none" else "")
    )

    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        r["kept"] = i in kept_set
        r["keep_reason"] = reason if r["kept"] else "ranked_out"
        out.append(r)
    return out


def _select_intersection(rows: list[dict], s: TopFractionIntersectionStrategy) -> list[dict]:
    """Top-K% on each axis separately, kept ⇔ in both sets."""
    if not rows:
        return []
    n = len(rows)
    k = max(1, min(n, round(n * s.keep_fraction)))
    by_u = sorted(range(n), key=lambda i: -rows[i]["utility_conf"])[:k]
    by_o = sorted(range(n), key=lambda i: -rows[i]["originality_conf"])[:k]
    kept_set = set(by_u) & set(by_o)
    reason = f"in_both_top{int(s.keep_fraction*100)}_intersections"

    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        r["kept"] = i in kept_set
        r["keep_reason"] = reason if r["kept"] else "outside_intersection"
        out.append(r)
    return out


def _rank_score(row: dict, rank_by: str) -> float:
    """Resolve ``rank_by ∈ {'utility','originality','sum'}`` to a row score."""
    if rank_by == "sum":
        return row["utility_conf"] + row["originality_conf"]
    return row[f"{rank_by}_conf"]


def _select_top_k_per_item(rows: list[dict], s: TopKPerItemStrategy) -> list[dict]:
    """For each ``item_id``, rank by ``rank_by`` and keep top ``k_per_item``.

    If an item has fewer than ``k`` candidates after Phase 1, keep all of
    them — we never fabricate. Tiebreaker resolves ties on the primary
    score; ``tiebreaker="none"`` falls back to Python's stable order.
    """
    if not rows:
        return []
    if s.tiebreaker == "none":
        key = lambda r: -_rank_score(r, s.rank_by)
    else:
        key = lambda r: (-_rank_score(r, s.rank_by), -r[f"{s.tiebreaker}_conf"])

    # group indices by item_id (preserves first-seen order across groups)
    by_item: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_item[str(r["item_id"])].append(i)

    kept_set: set[int] = set()
    for _item_id, idxs in by_item.items():
        ordered = sorted(idxs, key=lambda i: key(rows[i]))
        kept_set.update(ordered[: s.k_per_item])

    reason = f"top{s.k_per_item}_per_item_rank_{s.rank_by}"
    if s.tiebreaker != "none":
        reason += f"_tb_{s.tiebreaker}"

    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        r["kept"] = i in kept_set
        r["keep_reason"] = reason if r["kept"] else "outranked_in_item"
        out.append(r)
    return out


def apply_strategy(rows: list[dict], strategy: SelectStrategy) -> list[dict]:
    """Dispatch on the strategy variant (the only place that knows them all)."""
    if isinstance(strategy, ThresholdStrategy):
        return _select_threshold(rows, strategy)
    if isinstance(strategy, TopFractionStrategy):
        return _select_top_fraction(rows, strategy)
    if isinstance(strategy, TopFractionIntersectionStrategy):
        return _select_intersection(rows, strategy)
    if isinstance(strategy, TopKPerItemStrategy):
        return _select_top_k_per_item(rows, strategy)
    raise ValueError(f"Unknown brainstorm-select strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------


SELECT_COLUMNS = [
    "item_id", "task_type", "domain", "split", "source_id", "sample_idx", "seed",
    "object", "object_source",
    "method", "prompt_id", "prompt_sha256", "prompt_input_hash",
    "model_name", "model_revision", "decoding_json",
    "raw_response", "parsed_response", "output_hash",
    # Score + decision tail
    "use_text",
    "utility_logit_yes", "utility_logit_no", "utility_conf",
    "originality_logit_yes", "originality_logit_no", "originality_conf",
    "advantages_text", "drawbacks_text",
    "kept", "keep_reason",
]


# ---------------------------------------------------------------------------
# Phase-2 scoring (model-backed)
# ---------------------------------------------------------------------------


def _explode_uses_from_infer_csv(infer_csv: Path) -> list[dict]:
    """One row per (item, sample, parsed_use). Carries provenance forward."""
    rows: list[dict] = []
    with infer_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                uses = json.loads(r.get("parsed_response", "[]"))
            except json.JSONDecodeError:
                uses = []
            for use in uses:
                rows.append({**r, "use_text": use})
    return rows


# A "scorer" returns the score dict for one use. Default is the model-backed
# one below; tests inject a stub so the strategy pathway is unit-testable.
def _model_scorer(loader: Any, max_new_tokens: int, skip_chain: bool):
    """Closure that builds the four Phase-2 messages and probes the model.

    Captures ``loader`` + chain settings so the per-row call site stays a
    pure ``(row) -> dict`` function (matches the test stub signature).
    """
    from bridge_steering.methods import (
        BaselineMethod, forced_choice_logprobs,
    )
    gen = BaselineMethod()

    def _score(row: dict) -> dict:
        obj = row["object"]
        purpose = row["use_text"]

        if skip_chain:
            advantages_text = ""
            drawbacks_text = ""
        else:
            adv_msgs = AUT_BS_ADVANTAGES_V1.render(object=obj, purpose=purpose)
            drw_msgs = AUT_BS_DRAWBACKS_V1.render(object=obj, purpose=purpose)
            advantages_text = gen.generate(
                loader, adv_msgs, max_new_tokens=max_new_tokens,
            ).strip()
            drawbacks_text = gen.generate(
                loader, drw_msgs, max_new_tokens=max_new_tokens,
            ).strip()

        util_msgs = AUT_BS_UTIL_EVAL_V1.render(
            object=obj, purpose=purpose,
            advantages=advantages_text, drawbacks=drawbacks_text,
        )
        orig_msgs = AUT_BS_ORIG_EVAL_V1.render(object=obj, purpose=purpose)

        u_yes, u_no = forced_choice_logprobs(loader, util_msgs)
        o_yes, o_no = forced_choice_logprobs(loader, orig_msgs)

        return {
            "advantages_text": advantages_text,
            "drawbacks_text":  drawbacks_text,
            "utility_logit_yes": u_yes, "utility_logit_no": u_no,
            "utility_conf": u_yes - u_no,
            "originality_logit_yes": o_yes, "originality_logit_no": o_no,
            "originality_conf": o_yes - o_no,
        }

    return _score


def _batched_model_score(
    loader: Any,
    rows: list[dict],
    max_new_tokens: int,
    skip_chain: bool,
    batch_size: int = 32,
) -> list[dict]:
    """4-pass batched scorer for the brainstorm-select Phase 2.

    Replaces ``[scorer(r) for r in rows]`` with four batched model passes
    (advantages → drawbacks → utility logprobs → originality logprobs).
    Per-row output is bit-equivalent to ``_model_scorer`` up to fp noise.
    """
    from bridge_steering.methods import (
        BaselineMethod, forced_choice_logprobs_batch,
    )
    gen = BaselineMethod()
    n = len(rows)
    if n == 0:
        return []

    # Build adv / drw message lists (per row, independent of generated text)
    adv_msgs_list = [
        AUT_BS_ADVANTAGES_V1.render(object=r["object"], purpose=r["use_text"])
        for r in rows
    ]
    drw_msgs_list = [
        AUT_BS_DRAWBACKS_V1.render(object=r["object"], purpose=r["use_text"])
        for r in rows
    ]

    if skip_chain:
        advantages_texts = [""] * n
        drawbacks_texts = [""] * n
    else:
        # Pass 1: advantages (batched generation)
        advantages_texts: list[str] = []
        for s in range(0, n, batch_size):
            chunk = adv_msgs_list[s : s + batch_size]
            outs = gen.generate_batch(loader, chunk, max_new_tokens=max_new_tokens)
            advantages_texts.extend(t.strip() for t in outs)
        # Pass 2: drawbacks (batched generation)
        drawbacks_texts: list[str] = []
        for s in range(0, n, batch_size):
            chunk = drw_msgs_list[s : s + batch_size]
            outs = gen.generate_batch(loader, chunk, max_new_tokens=max_new_tokens)
            drawbacks_texts.extend(t.strip() for t in outs)

    # Pass 3: utility yes/no (depends on adv+drw texts)
    util_msgs_list = [
        AUT_BS_UTIL_EVAL_V1.render(
            object=rows[i]["object"], purpose=rows[i]["use_text"],
            advantages=advantages_texts[i], drawbacks=drawbacks_texts[i],
        )
        for i in range(n)
    ]
    util_results = forced_choice_logprobs_batch(
        loader, util_msgs_list, batch_size=batch_size,
    )
    # Pass 4: originality yes/no (independent)
    orig_msgs_list = [
        AUT_BS_ORIG_EVAL_V1.render(object=r["object"], purpose=r["use_text"])
        for r in rows
    ]
    orig_results = forced_choice_logprobs_batch(
        loader, orig_msgs_list, batch_size=batch_size,
    )

    out: list[dict] = []
    for i in range(n):
        u_yes, u_no = util_results[i]
        o_yes, o_no = orig_results[i]
        out.append({
            "advantages_text": advantages_texts[i],
            "drawbacks_text":  drawbacks_texts[i],
            "utility_logit_yes": u_yes, "utility_logit_no": u_no,
            "utility_conf": u_yes - u_no,
            "originality_logit_yes": o_yes, "originality_logit_no": o_no,
            "originality_conf": o_yes - o_no,
        })
    return out


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(ctx: RunContext, scorer: Any = None) -> StageResult:
    """Run Phase 2 (select) over Phase 1 (aut_infer) outputs.

    For each method's ``aut_infer/{method}_results.csv``, explode parsed
    uses into rows, score utility + originality, apply the configured
    strategy, and write ``{method_slug}_selected_results.csv``.

    ``scorer`` is an injection point: pass a ``(row) -> dict`` function to
    bypass the model (used by the integration test). Default: build the
    model-backed scorer from ``ctx``.
    """
    cfg = ctx.cfg
    t0 = time.time()
    if cfg.brainstorm_select is None:
        # Soft-skip: DEFAULT_ORDER includes this stage, but configs without a
        # `brainstorm_select:` block (every non-brainstorm run) shouldn't crash
        # on it. Downstream stages don't depend on its outputs.
        print("[aut_brainstorm_select] cfg.brainstorm_select not set; skipping.")
        return StageResult(
            name="aut_brainstorm_select",
            elapsed_s=time.time() - t0,
            notes={"skipped": True, "reason": "cfg.brainstorm_select not set"},
        )
    bs = cfg.brainstorm_select

    infer_dir = ctx.stage_dir("aut_infer")
    stage_dir = ctx.stage_dir("aut_brainstorm_select")
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Build the scorer once (loads the model lazily inside the closure).
    # Default = batched: a 4-pass model loop (~30-80× faster than per-row).
    # Tests inject a per-row ``scorer`` to bypass the model — that path falls
    # back to the original list comprehension.
    use_batched = scorer is None
    loader = None
    if use_batched:
        from bridge_steering.models import HFLoader
        loader = HFLoader(model_name_or_path=cfg.model_name)
        loader.load()

    outputs: list[dict] = []
    n_inputs_total = 0
    n_kept_total = 0

    for csv_path in sorted(infer_dir.glob("*_results.csv")):
        source_method_slug = csv_path.stem.replace("_results", "")
        out_slug = bs.method_slug or source_method_slug

        use_rows = _explode_uses_from_infer_csv(csv_path)
        n_inputs_total += len(use_rows)

        if use_batched:
            score_dicts = _batched_model_score(
                loader, use_rows, bs.chain_max_new_tokens, bs.skip_chain,
            )
            scored = [
                {**r, **sd, "method": out_slug}
                for r, sd in zip(use_rows, score_dicts)
            ]
        else:
            scored = [{**r, **scorer(r), "method": out_slug} for r in use_rows]
        decided = apply_strategy(scored, bs.strategy)
        # Only kept rows go to the canonical CSV — downstream stages
        # (aut_novelty / aut_ocsai / aut_judge) score every row they read,
        # so writing rejected rows would defeat the filter. Audit info
        # lives in the manifest (n_input_rows / n_kept / kept_fraction).
        kept_rows = [r for r in decided if r["kept"]]
        n_kept_total += len(kept_rows)

        out_path = stage_dir / f"{out_slug}_selected_results.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SELECT_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(kept_rows)
        outputs.append(hashed_output(out_path, rows=len(kept_rows)))

    inputs = {p.stem: hashed_input(p) for p in sorted(infer_dir.glob("*_results.csv"))}

    return StageResult(
        name="aut_brainstorm_select",
        inputs=inputs,
        outputs=outputs,
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "strategy": _strategy_to_dict(bs.strategy),
                "skip_chain": bs.skip_chain,
                "chain_max_new_tokens": bs.chain_max_new_tokens,
                "method_slug": bs.method_slug,
            },
            "brainstorm_select": {
                "strategy": _strategy_to_dict(bs.strategy),
                "skip_chain": bs.skip_chain,
                "n_input_rows": n_inputs_total,
                "n_kept": n_kept_total,
                "kept_fraction_observed": (
                    n_kept_total / n_inputs_total if n_inputs_total else 0.0
                ),
            },
        },
    )


def _strategy_to_dict(s: SelectStrategy) -> dict:
    """Strategy → JSON-serializable dict (for the manifest)."""
    from dataclasses import asdict
    return asdict(s)
