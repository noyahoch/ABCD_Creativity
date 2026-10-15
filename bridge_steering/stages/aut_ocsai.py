"""Stage: OCSAI scoring (originality) for AUT replies.

Reads the canonical AUT replies CSV (brainstorm-select output if
configured, else aut_infer output) and posts ONE row per (item, sample)
to the OCSAI HTTP endpoint with ``(object, full_reply)``.

We score the whole multi-use reply as one OpenScoring input — that is what
the legacy ``scripts/run_ocsai_scoring.py`` does and what the davinci3
model on openscoring.du.edu/llm reliably grades. Calling OCSAI with one
parsed use per call (the prior bridge_steering behaviour) caused it to return
``NaN`` for ~30 % of inputs.

Classified as external-service: the format half is deterministic but the
HTTP call is auditable-only. We save the raw response text per batch
into the manifest's per-batch audit list.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import time
from pathlib import Path
from typing import Any

from ..config import OCSAIConfig, RunConfig
from ..prompts.aut import OCSAI_REQUEST_V1
from ..repro import RunContext, StageResult, hashed_input, hashed_output, sha256_str


# ---------------------------------------------------------------------------
# Path resolution: brainstorm-select output if present, else aut_infer
# ---------------------------------------------------------------------------


def aut_replies_dir(ctx: RunContext) -> Path:
    """The directory aut_ocsai / aut_judge consume.

    One rule, one place: ``brainstorm_select`` takes precedence.
    """
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
# CSV columns
# ---------------------------------------------------------------------------


OCSAI_BASE_COLUMNS = [
    "item_id", "task_type", "domain", "split", "source_id", "sample_idx", "seed",
    "object", "method", "reply_text",
    "ocsai_model", "ocsai_request_hash", "ocsai_request_prompt_id", "ocsai_request_prompt_sha256",
    "ocsai_originality", "raw_response_path", "output_hash",
]


# ---------------------------------------------------------------------------
# Row materialization — ONE OCSAI score per (item, sample) reply, not per
# parsed use. Matches scripts/run_ocsai_scoring.py (the reliable path).
# ---------------------------------------------------------------------------


def _load_replies(replies_csv: Path) -> list[dict]:
    """Return one row per inference reply, with ``reply_text`` populated.

    aut_infer rows expose the full multi-use response under ``raw_response``;
    brainstorm-select rows store the per-row final output under ``use_text``.
    Either way we feed the *whole* reply to OCSAI.
    """
    out: list[dict] = []
    with replies_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            reply = r.get("raw_response") or r.get("use_text") or r.get("reply") or ""
            if not reply:
                continue
            r = dict(r)
            r["reply_text"] = reply
            out.append(r)
    return out


def _explode_uses(replies_csv: Path) -> list[dict]:
    """Return one row per parsed use. Kept for downstream stages
    (``aut_judge``, ``aut_novelty``) that score per-use rather than
    per-reply. Used by import; not by this stage's own scoring path."""
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


def _row(item_row: dict, originality: float | None, raw_path: str, raw: str,
         oc: OCSAIConfig) -> dict:
    spec = OCSAI_REQUEST_V1
    req_payload = json.dumps({"url": oc.url, "model": oc.model,
                              "object": item_row.get("object", ""),
                              "reply": item_row.get("reply_text", "")}, sort_keys=True)
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
        "reply_text": item_row.get("reply_text", ""),
        "ocsai_model": oc.model,
        "ocsai_request_hash": "sha256:" + sha256_str(req_payload),
        "ocsai_request_prompt_id": spec.id,
        "ocsai_request_prompt_sha256": "sha256:" + spec.sha256(),
        "ocsai_originality": originality,
        "raw_response_path": raw_path,
        "output_hash": "sha256:" + sha256_str(raw),
    }


# ---------------------------------------------------------------------------
# Sidecar audit storage — raw OCSAI responses are written once per unique
# response (keyed by sha256), and rows reference them by relative path. This
# avoids duplicating the same JSON across every row in a batch.
# ---------------------------------------------------------------------------


def _write_audit(stage_dir: Path, raw_text: str) -> str:
    """Write ``raw_text`` to ``audit/<sha256>.json`` (idempotent).

    Returns the path relative to ``stage_dir`` (the directory the CSV lives in),
    so readers can resolve it without knowing the stage directory.
    """
    sha = sha256_str(raw_text)
    audit_dir = stage_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    p = audit_dir / f"{sha}.json"
    if not p.exists():
        p.write_text(raw_text, encoding="utf-8")
    return f"audit/{sha}.json"


# ---------------------------------------------------------------------------
# Score-call abstraction (swappable for tests via cfg-injected fake)
# ---------------------------------------------------------------------------


def _default_score_batch(rows: list[dict], oc: OCSAIConfig) -> list[tuple[float | None, str]]:
    """Default scorer — calls the real OCSAI endpoint with retry.

    Returns ``(originality, raw_response_text)`` per row. ``None`` originality
    on failure; raw text is always saved (audit trail).

    Retries transient failures up to ``oc.max_attempts`` with linear backoff
    ``oc.retry_backoff_s * attempt``. After all retries exhausted, returns the
    error tuple for every row in the batch.
    """
    from bridge_steering.ocsai import format_ocsai_row, score_ocsai_llm

    # Pass just the OBJECT as the OCSAI prompt — that's the academically-
    # standard input format for openscoring.du.edu (object/cue as prompt,
    # response as the use). The OLD ``scripts/run_ocsai_scoring.py:217`` path
    # passed the full rendered AUT user prompt instead, which OCSAI doesn't
    # know how to interpret correctly. Keeping object-only here is the
    # correct behavior; numerical divergence from OLD on this metric is
    # OLD doing it wrong, not NEW.
    inputs = [format_ocsai_row(r.get("object", ""), r.get("reply_text", "")) for r in rows]
    last_err: Exception | None = None
    for attempt in range(1, max(1, oc.max_attempts) + 1):
        try:
            data = score_ocsai_llm(inputs, url=oc.url, model=oc.model, timeout=oc.timeout_s)
            scores = data.get("scores", []) if isinstance(data, dict) else []
            # The endpoint echoes a "cite" array (paper + software citation) in
            # every response. Drop it before hashing/storing — it's static
            # boilerplate, not per-call data.
            if isinstance(data, dict) and "cite" in data:
                data = {k: v for k, v in data.items() if k != "cite"}
            raw = json.dumps(data, sort_keys=True)
            out: list[tuple[float | None, str]] = []
            for i, _r in enumerate(rows):
                s = scores[i] if i < len(scores) else None
                val = s.get("originality") if isinstance(s, dict) else s
                try:
                    val = float(val) if val is not None else None
                except (TypeError, ValueError):
                    val = None
                out.append((val, raw))
            return out
        except Exception as e:
            last_err = e
            if attempt < max(1, oc.max_attempts):
                time.sleep(oc.retry_backoff_s * attempt)

    # All attempts failed — return error tuple for every row so the stage still
    # produces a CSV (audit string captures the final exception).
    err_text = f"OCSAI_ERROR: {type(last_err).__name__}: {last_err}"
    return [(None, err_text)] * len(rows)


def _write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(ctx: RunContext, score_batch=None) -> StageResult:
    """Stage entry. ``score_batch`` is overridable for tests."""
    cfg = ctx.cfg
    t0 = time.time()
    if cfg.ocsai is None:
        raise ValueError("aut_ocsai requires cfg.ocsai to be set")
    oc = cfg.ocsai

    score_fn = score_batch or _default_score_batch
    replies = _replies_glob(ctx)
    stage_dir = ctx.stage_dir("aut_ocsai")
    stage_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict] = []
    inputs: dict[str, dict] = {}
    n_rows_total = 0
    n_failures_total = 0

    for path in replies:
        method = _method_slug_from_path(path)
        item_rows = _load_replies(path)
        n_rows_total += len(item_rows)

        # Slice into batches up-front so concurrency just maps over them.
        batches = [
            item_rows[start : start + oc.batch_size]
            for start in range(0, len(item_rows), oc.batch_size)
        ]
        if oc.max_concurrency > 1 and len(batches) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(oc.max_concurrency, len(batches))
            ) as ex:
                batch_results = list(ex.map(lambda b: score_fn(b, oc), batches))
        else:
            batch_results = [score_fn(b, oc) for b in batches]

        scored_rows: list[dict] = []
        for batch, results in zip(batches, batch_results):
            for r, (orig, raw) in zip(batch, results):
                if orig is None:
                    n_failures_total += 1
                raw_path = _write_audit(stage_dir, raw)
                scored_rows.append(_row(r, orig, raw_path, raw, oc))

        out_path = stage_dir / f"{method}_ocsai_scores.csv"
        _write_csv(scored_rows, out_path, OCSAI_BASE_COLUMNS)
        outputs.append(hashed_output(out_path, rows=len(scored_rows)))
        inputs[method] = hashed_input(path)

    return StageResult(
        name="aut_ocsai",
        inputs=inputs,
        outputs=outputs,
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "url": oc.url, "model": oc.model, "batch_size": oc.batch_size,
                "max_attempts": oc.max_attempts,
                "retry_backoff_s": oc.retry_backoff_s,
                "max_concurrency": oc.max_concurrency,
                "reads_from": str(aut_replies_dir(ctx).name),
            },
            "n_scored_rows": n_rows_total,
            "n_failures": n_failures_total,
            "ocsai_request_prompt": OCSAI_REQUEST_V1.manifest_entry(),
        },
    )
