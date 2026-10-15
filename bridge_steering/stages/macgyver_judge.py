"""Stage: LLM-as-a-judge for MacGyver replies (additive 5-pt rubric).

Same backend-agnostic shape as ``aut_judge``: vLLM or API. The prompt is
``prompts.macgyver.JUDGE_ADDITIVE_5PT_V1`` and the parser extracts an
integer 0-5 from the model's final line (``Score: 4``).
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
from ..prompts.macgyver import MACGYVER_JUDGE_ADDITIVE_5PT_V1, build_judge_prompt
from ..repro import RunContext, StageResult, hashed_input, hashed_output, sha256_str

from .._timing import timed
from ._judge_client import completion_kwargs, judge_model_name, make_judge_client


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------


JUDGE_COLUMNS = [
    "item_id", "task_type", "domain", "split", "source_id", "sample_idx", "seed",
    "problem_id", "method", "model_response",
    "inference_prompt_id", "inference_prompt_sha256",
    "judge_prompt_id", "judge_prompt_sha256", "judge_prompt_input_hash",
    "judge_model", "judge_model_revision", "judge_decoding_json",
    "quality_score", "raw_response", "output_hash", "error",
]


_SCORE_LINE_RE = re.compile(r"[Ss]core\s*:\s*(\d)")
_FRAC_RE = re.compile(r"\b([0-5])\s*/\s*5\b")


def parse_macgyver_score(raw: str) -> int | None:
    """Extract a 0-5 integer from the judge response. ``None`` on failure."""
    raw = raw or ""
    m = _SCORE_LINE_RE.search(raw)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 5:
            return v
    m = _FRAC_RE.search(raw)
    if m:
        return int(m.group(1))
    return None


async def _judge_one_async(
    client, model_name: str, item_row: dict, jc: JudgeConfig, allow_archived: bool,
) -> dict:
    # Pass the rendered user message (problem + MACGYVER_USER_SUFFIX for the
    # default style) to the judge, matching OLD scripts/score_macgyver_quality.py
    # which reads CSV's user_prompt column. Fallback to problem_text for CSVs
    # produced before user_prompt was added to the MG inference schema.
    judge_user_prompt = item_row.get("user_prompt") or item_row.get("problem_text", "")
    msgs, spec, prompt_input = build_judge_prompt(
        jc.prompt_id,
        judge_user_prompt,
        item_row.get("raw_response", "") or item_row.get("parsed_response", ""),
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
            score = parse_macgyver_score(last_raw)
            if score is not None:
                return {
                    "quality_score": score, "raw_response": last_raw,
                    "spec": spec, "prompt_input": prompt_input, "error": "",
                }
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

    return {
        "quality_score": None, "raw_response": last_raw,
        "spec": spec, "prompt_input": prompt_input,
        "error": err or "score_parse_failed",
    }


def _row(item_row: dict, verdict: dict, jc: JudgeConfig) -> dict:
    spec = verdict["spec"]
    decoding = {"temperature": jc.temperature, "max_tokens": jc.max_tokens,
                "max_concurrent": jc.max_concurrent, "seed": jc.seed}
    return {
        "item_id": item_row["item_id"],
        "task_type": item_row.get("task_type", "macgyver"),
        "domain": item_row.get("domain", "macgyver"),
        "split": item_row.get("split", "eval"),
        "source_id": item_row.get("source_id", item_row["item_id"]),
        "sample_idx": item_row.get("sample_idx", 0),
        "seed": item_row.get("seed", ""),
        "problem_id": item_row.get("problem_id", item_row["item_id"]),
        "method": item_row.get("method", ""),
        "model_response": item_row.get("raw_response", ""),
        "inference_prompt_id": item_row.get("prompt_id", ""),
        "inference_prompt_sha256": item_row.get("prompt_sha256", ""),
        "judge_prompt_id": spec.id,
        "judge_prompt_sha256": "sha256:" + spec.sha256(),
        "judge_prompt_input_hash": "sha256:" + sha256_str(verdict["prompt_input"]),
        "judge_model": jc.backend.model,
        "judge_model_revision": "",
        "judge_decoding_json": json.dumps(decoding, sort_keys=True),
        "quality_score": verdict["quality_score"],
        "raw_response": verdict["raw_response"],
        "output_hash": "sha256:" + sha256_str(verdict["raw_response"] or ""),
        "error": verdict["error"],
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JUDGE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


async def _run_async(item_rows: list[dict], jc: JudgeConfig, allow_archived: bool,
                     judge_async: Callable | None = None) -> list[dict]:
    if judge_async is None:
        client = make_judge_client(jc.backend)
        model = judge_model_name(jc.backend)
        sema = asyncio.Semaphore(jc.max_concurrent)

        async def _one(r):
            async with sema:
                v = await _judge_one_async(client, model, r, jc, allow_archived)
                return _row(r, v, jc)
    else:
        async def _one(r):
            v = await judge_async(r)
            return _row(r, v, jc)

    return await asyncio.gather(*(_one(r) for r in item_rows))


def run(ctx: RunContext, judge_async: Callable | None = None) -> StageResult:
    cfg = ctx.cfg
    t0 = time.time()
    if cfg.macgyver_judge is None:
        raise ValueError("macgyver_judge requires cfg.macgyver_judge to be set")
    jc = cfg.macgyver_judge

    infer_dir = ctx.stage_dir("macgyver_infer")
    stage_dir = ctx.stage_dir("macgyver_judge")
    stage_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict] = []
    inputs: dict[str, dict] = {}
    n_total = 0
    n_failures = 0

    for path in sorted(infer_dir.glob("*_results.csv")):
        method = path.stem.replace("_results", "")
        with path.open("r", encoding="utf-8") as f:
            item_rows = list(csv.DictReader(f))
        n_total += len(item_rows)
        with timed(f"macgyver_judge: {method}", n=len(item_rows)):
            scored = asyncio.run(
                _run_async(item_rows, jc, cfg.allow_archived_prompts, judge_async)
            )
        n_failures += sum(1 for r in scored if r["quality_score"] is None)
        out_path = stage_dir / f"{method}_judge_scores.csv"
        _write_csv(scored, out_path)
        outputs.append(hashed_output(out_path, rows=len(scored)))
        inputs[method] = hashed_input(path)

    return StageResult(
        name="macgyver_judge",
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
        extra_prompts={"judge_prompt": MACGYVER_JUDGE_ADDITIVE_5PT_V1.manifest_entry()},
    )
