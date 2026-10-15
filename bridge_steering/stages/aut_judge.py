"""Stage: LLM-as-a-judge for AUT replies (novelty + usability).

Backend-agnostic — works against vLLM-hosted models (Qwen, Prometheus)
and OpenAI-compatible cloud endpoints (OpenAI, Together, DeepInfra,
Anthropic-via-OpenAI-compat, …). The backend is selected via
``cfg.aut_judge.backend.type`` (``vllm`` or ``api``).

Reads the canonical AUT replies CSV (brainstorm-select output if
configured, else aut_infer output). Per-row CSV carries both the
inference prompt (forwarded) and the judge prompt + judge model, so a
single row records the full provenance of one score.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from ..config import JudgeConfig, RunConfig
from ..prompts.aut import JUDGE_NOVELTY_USABILITY_V5, build_judge_prompt
from ..repro import RunContext, StageResult, hashed_input, hashed_output, sha256_str

from .aut_ocsai import _explode_uses, _method_slug_from_path, _replies_glob
from .._timing import timed
from ._judge_client import completion_kwargs, judge_model_name, make_judge_client


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------


JUDGE_COLUMNS = [
    "item_id", "task_type", "domain", "split", "source_id", "sample_idx", "seed",
    "object", "method", "use_text",
    "inference_prompt_id", "inference_prompt_sha256",
    "judge_prompt_id", "judge_prompt_sha256", "judge_prompt_input_hash",
    "judge_model", "judge_model_revision", "judge_decoding_json",
    "novelty", "usability", "overall",
    "raw_response", "output_hash", "error",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _coerce_score_1_to_10(v: Any) -> int | None:
    """Clamp to [1, 10] as int; None on garbage. Ported from run_vllm_judge."""
    try:
        if v is None:
            return None
        if isinstance(v, str):
            m = re.search(r"-?\d+(?:\.\d+)?", v)
            if not m:
                return None
            x = float(m.group())
        else:
            x = float(v)
        if x != x:  # NaN
            return None
        return int(max(1, min(10, round(x))))
    except (TypeError, ValueError):
        return None


def parse_judge_response(raw: str) -> tuple[int | None, int | None]:
    """Return ``(novelty, usability)`` from the model's raw text.

    Tries strict JSON first; falls back to regex on novelty/usability fields.
    """
    raw = (raw or "").strip()
    if raw.startswith("{") or "{" in raw:
        # Trim wrappers if present.
        if not raw.startswith("{"):
            i = raw.find("{")
            j = raw.rfind("}")
            if i != -1 and j != -1 and j > i:
                raw = raw[i : j + 1]
        try:
            d = json.loads(raw)
            scores = d.get("scores", d)
            if isinstance(scores, dict):
                return (
                    _coerce_score_1_to_10(scores.get("novelty")),
                    _coerce_score_1_to_10(scores.get("usability")),
                )
        except json.JSONDecodeError:
            pass

    n_m = re.search(r'(?i)"?novelty"?\s*[:=]\s*"?(-?\d+(?:\.\d+)?)"?', raw)
    u_m = re.search(r'(?i)"?usability"?\s*[:=]\s*"?(-?\d+(?:\.\d+)?)"?', raw)
    n = _coerce_score_1_to_10(n_m.group(1)) if n_m else None
    u = _coerce_score_1_to_10(u_m.group(1)) if u_m else None
    return n, u


# ---------------------------------------------------------------------------
# Use-text canonicalization
# ---------------------------------------------------------------------------


# Ported from ``scripts/run_aut_inference._strip_inner_commas``. The OLD
# pipeline strips inner commas from each parsed use before writing
# ``{method}_uses.txt`` (so the judge sees ``"object, use_with_no_inner_commas"``
# split on the first comma). When NEW pipelines were judging the same
# byte-identical replies but with commas preserved, the judge saw slightly
# different strings → judge scores diverged by ±0.02-0.03 on novelty/usability.
# Applying the same canonicalization here closes that gap for bit-exact judge
# parity with OLD-pipeline runs.
def _strip_inner_commas(text: str) -> str:
    """Remove inner commas so judge sees the same text as OLD's _uses.txt."""
    text = re.sub(r",\s+(and|or|but|so|yet|nor)\b", r" \1", text)
    text = text.replace(", ", " ").replace(",", "")
    return text


# ---------------------------------------------------------------------------
# Async judge call (one row)
# ---------------------------------------------------------------------------


async def _judge_one_async(
    client, model_name: str, item_row: dict, jc: JudgeConfig, allow_archived: bool,
) -> dict:
    # Strip inner commas before sending to the judge, matching OLD pipeline.
    canonical_use = _strip_inner_commas(item_row.get("use_text", ""))
    msgs, spec, prompt_input = build_judge_prompt(
        jc.prompt_id, item_row.get("object", ""), canonical_use,
        allow_archived=allow_archived,
    )
    last_raw, err = "", ""
    for attempt in range(jc.max_retries):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name, messages=msgs,
                    **completion_kwargs(
                        jc.backend, jc.max_tokens, jc.temperature, jc.seed,
                    ),
                ),
                timeout=jc.timeout_s,
            )
            last_raw = (resp.choices[0].message.content or "").strip()
            n, u = parse_judge_response(last_raw)
            if n is not None and u is not None:
                return {
                    "novelty": n, "usability": u,
                    "overall": round((n + u) / 2),
                    "raw_response": last_raw, "spec": spec,
                    "prompt_input": prompt_input, "error": "",
                }
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

    # Out of retries
    return {
        "novelty": None, "usability": None, "overall": None,
        "raw_response": last_raw, "spec": spec,
        "prompt_input": prompt_input, "error": err or "parse_failed",
    }


# ---------------------------------------------------------------------------
# CSV row builder
# ---------------------------------------------------------------------------


def _row(item_row: dict, verdict: dict, jc: JudgeConfig) -> dict:
    spec = verdict["spec"]
    decoding = {"temperature": jc.temperature, "max_tokens": jc.max_tokens,
                "max_concurrent": jc.max_concurrent, "seed": jc.seed}
    return {
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
        "inference_prompt_id": item_row.get("prompt_id", ""),
        "inference_prompt_sha256": item_row.get("prompt_sha256", ""),
        "judge_prompt_id": spec.id,
        "judge_prompt_sha256": "sha256:" + spec.sha256(),
        "judge_prompt_input_hash": "sha256:" + sha256_str(verdict["prompt_input"]),
        "judge_model": _judge_model_id(jc),
        "judge_model_revision": "",
        "judge_decoding_json": json.dumps(decoding, sort_keys=True),
        "novelty": verdict["novelty"],
        "usability": verdict["usability"],
        "overall": verdict["overall"],
        "raw_response": verdict["raw_response"],
        "output_hash": "sha256:" + sha256_str(verdict["raw_response"] or ""),
        "error": verdict["error"],
    }


def _judge_model_id(jc: JudgeConfig) -> str:
    return jc.backend.model


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


async def _run_async(rows: list[dict], jc: JudgeConfig, allow_archived: bool,
                     judge_async: Callable | None = None) -> list[dict]:
    """Score every row. ``judge_async`` is overridable for tests (no network)."""
    if judge_async is None:
        client = make_judge_client(jc.backend)
        model = judge_model_name(jc.backend)
        sema = asyncio.Semaphore(jc.max_concurrent)

        async def _one(r):
            async with sema:
                v = await _judge_one_async(client, model, r, jc, allow_archived)
                return _row(r, v, jc)
    else:
        # Test path: judge_async(r) -> verdict dict
        async def _one(r):
            v = await judge_async(r)
            return _row(r, v, jc)

    return await asyncio.gather(*(_one(r) for r in rows))


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JUDGE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(ctx: RunContext, judge_async: Callable | None = None) -> StageResult:
    cfg = ctx.cfg
    t0 = time.time()
    if cfg.aut_judge is None:
        raise ValueError("aut_judge requires cfg.aut_judge to be set")
    jc = cfg.aut_judge

    replies = _replies_glob(ctx)
    stage_dir = ctx.stage_dir("aut_judge")
    stage_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict] = []
    inputs: dict[str, dict] = {}
    n_total = 0
    n_failures = 0

    for path in replies:
        method = _method_slug_from_path(path)
        item_rows = _explode_uses(path)
        n_total += len(item_rows)
        with timed(f"aut_judge: {method}", n=len(item_rows)):
            scored = asyncio.run(
                _run_async(item_rows, jc, cfg.allow_archived_prompts, judge_async)
            )
        n_failures += sum(1 for r in scored if r["novelty"] is None)
        out_path = stage_dir / f"{method}_judge_scores.csv"
        _write_csv(scored, out_path)
        outputs.append(hashed_output(out_path, rows=len(scored)))
        inputs[method] = hashed_input(path)

    judge_spec = JUDGE_NOVELTY_USABILITY_V5 if jc.prompt_id == JUDGE_NOVELTY_USABILITY_V5.id else None

    return StageResult(
        name="aut_judge",
        inputs=inputs,
        outputs=outputs,
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "backend_type": jc.backend.type,
                "model": jc.backend.model,
                "max_concurrent": jc.max_concurrent,
                "prompt_id": jc.prompt_id,
            },
            "n_judged_rows": n_total,
            "n_failures": n_failures,
        },
        extra_prompts={"judge_prompt": (judge_spec.manifest_entry()
                                        if judge_spec is not None else {"id": jc.prompt_id})},
    )
