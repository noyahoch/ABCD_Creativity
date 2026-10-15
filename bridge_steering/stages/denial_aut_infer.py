"""Stage: AUT inference under denial prompting (Lu et al. 2024 adapted).

Iterative loop per AUT eval item:

  round 0: standard inference prompt (resolved from cfg.aut.prompts.inference,
           byte-identical to what aut_infer's baseline method produces — the
           result will closely match `baseline_results.csv` modulo RNG order).
  round t (1..T):
       1. reviewer extracts a list of short approach phrases from
          round-(t-1)'s reply (vLLM Qwen by default).
       2. per-item RNG picks one approach not yet in this item's
          constraint set; appends to the running constraint list.
       3. inference with the denial-augmented prompt (cfg.denial.aut_denial_prompt_id)
          rendering ``{constraints_block}`` to the cumulative deny list.

Writes ``denial_round{t}_results.csv`` into ``aut_infer/`` (NOT a separate
stage dir) so the existing ``aut_ocsai``/``aut_judge`` discovery picks the
files up alongside the standard ``baseline_results.csv``. Each round becomes
its own ``method`` row in the downstream judge/ocsai/aggregate CSVs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path

from .._timing import timed
from ..config import RunConfig, VllmJudgeBackend
from ..managed_vllm import managed_vllm
from ..datasets import ABCDDataset
from ..prompts import resolve
from ..prompts.aut import aut_object, build_user_prompt
from ..repro import RunContext, StageResult, hashed_input, hashed_output, sha256_str

from ._denial_common import (
    format_constraints_block,
    per_item_rng,
    review_batch,
    select_next_constraint,
)
from .aut_infer import (
    AUT_INFER_COLUMNS,
    _build_user_messages,
    _decoding_payload,
    _row as _aut_row,
    _write_csv,
    parse_uses_from_reply,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_kw):  # type: ignore[no-redef]
        return it


def _denial_row(
    item: dict, sample_idx: int, cfg: RunConfig, method_slug: str, spec,
    prompt_input: str, reply: str,
) -> dict:
    """Same column schema as aut_infer but with method=`denial_round{t}`."""
    return _aut_row(item, sample_idx, cfg, "eval", method_slug, spec, prompt_input, reply)


def run(ctx: RunContext) -> StageResult:
    """Iterative denial-prompting inference for AUT."""
    from bridge_steering.methods import BaselineMethod
    from bridge_steering.models import HFLoader

    cfg = ctx.cfg
    t0 = time.time()
    if cfg.denial is None:
        # Soft-skip: DEFAULT_ORDER includes this stage, but configs without a
        # `denial:` block (i.e. every non-denial run) should not crash on it.
        # Stages downstream (aut_ocsai, aut_judge, aggregate) discover denial
        # rounds by scanning aut_infer/ for denial_round*_results.csv; with no
        # such files written, they simply have no extra rows to score.
        print("[denial_aut_infer] cfg.denial not set; skipping.")
        return StageResult(
            name="denial_aut_infer",
            elapsed_s=time.time() - t0,
            notes={"skipped": True, "reason": "cfg.denial not set"},
        )
    dcfg = cfg.denial

    # --- Data ---
    dataset = ABCDDataset.load(cfg.aut.abcd_data)
    split = dataset.split(cfg.aut.eval_seed, cfg.aut.eval_holdout)
    eval_items = split.eval_items

    # --- Resolve prompts ---
    # Round 0: standard AUT inference prompt (whatever cfg.aut.prompts.inference
    # picks — typically aut_user_a_paper_infer for the paper-replication track).
    round0_specs: list = []
    round0_user_strs: list[str] = []
    round0_msgs: list[list[dict]] = []
    for item in eval_items:
        msgs, spec, user_str = _build_user_messages(item, cfg, "inference")
        round0_msgs.append(msgs)
        round0_specs.append(spec)
        round0_user_strs.append(user_str)
    # Rounds 1+: denial-augmented inference prompt
    denial_spec = resolve(dcfg.aut_denial_prompt_id, allow_archived=cfg.allow_archived_prompts)
    # Reviewer prompt (used at end of every round t to extract approaches
    # from round-t's solutions — feeds round (t+1)).
    reviewer_spec = resolve(dcfg.reviewer_aut_prompt_id, allow_archived=cfg.allow_archived_prompts)

    # --- Decoding ---
    if cfg.decoding_aut.num_inferences > 1:
        raise ValueError(
            "denial_aut_infer assumes num_inferences=1 (one solution per round per item)."
        )

    # --- Load model + baseline method (round 0 + denial-augmented rounds
    # all share the same BaselineMethod — no steering, just an augmented
    # text prompt for rounds 1+) ---
    loader = HFLoader(model_name_or_path=cfg.model_name)
    with timed("denial_aut_infer: load model"):
        loader.load()
    method = BaselineMethod()
    if not hasattr(method, "slug"):
        method.slug = "baseline"

    # --- Generate ---
    # Outputs go into the aut_infer/ stage dir so downstream discovery
    # (aut_ocsai, aut_judge) picks them up automatically.
    stage_dir = ctx.stage_dir("aut_infer")
    stage_dir.mkdir(parents=True, exist_ok=True)
    # Also create our own stage dir for the manifest.
    manifest_dir = ctx.stage_dir("denial_aut_infer")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    gen_kwargs = {
        "max_new_tokens": cfg.decoding_aut.max_new_tokens,
        "temperature": cfg.decoding_aut.temperature,
        "do_sample": cfg.decoding_aut.temperature > 0,
        **{k: v for k, v in {"top_p": cfg.decoding_aut.top_p,
                             "min_p": cfg.decoding_aut.min_p}.items()
           if v is not None},
    }
    batch_size = max(1, int(cfg.decoding_aut.batch_size))

    if cfg.decoding_aut.gen_seed is not None and cfg.decoding_aut.temperature > 0:
        import torch as _torch
        s = int(cfg.decoding_aut.gen_seed)
        _torch.manual_seed(s)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(s)

    # Per-item state across rounds: the running constraint list.
    constraints_per_item: list[list[str]] = [[] for _ in eval_items]
    # Replies from the previous round, used to feed the reviewer.
    prev_replies: list[str] = [""] * len(eval_items)
    outputs: list[dict] = []

    for t in range(dcfg.rounds + 1):
        method_slug = f"denial_round{t}"
        rows: list[dict] = []

        # Build per-item (messages, spec, rendered_user_str) for this round.
        if t == 0:
            per_item = list(zip(eval_items, round0_msgs, round0_specs, round0_user_strs))
        else:
            # Reviewer pass: extract approaches from round (t-1) replies.
            # JUDGE TIME-MULTIPLEX — see denial_macgyver_infer for the full
            # rationale: release the generator, run the reviewer on an
            # ephemeral judge pinned inside this worker's own GPUs, tear it
            # down, reload the generator. RNG state is untouched.
            ephemeral = (
                isinstance(dcfg.reviewer_backend, VllmJudgeBackend)
                and cfg.vllm_server is not None
                and cfg.vllm_server.enable
            )
            if ephemeral:
                loader.unload()
            with timed(f"denial_aut_infer: reviewer round {t} (incl. judge lifecycle)",
                       n=len(prev_replies)):
                with (managed_vllm(cfg) if ephemeral else contextlib.nullcontext()):
                    extracted = asyncio.run(
                        review_batch(dcfg, reviewer_spec, "aut_response", prev_replies)
                    )
            if ephemeral:
                with timed(f"denial_aut_infer: reload generator round {t}"):
                    loader.load()
            # Update constraints per item; build denial-augmented prompts.
            per_item = []
            for i, (item, cur_constraints, approaches, base_user) in enumerate(
                zip(eval_items, constraints_per_item, extracted, round0_user_strs)
            ):
                rng = per_item_rng(cfg.seed, item["id"], t)
                new = select_next_constraint(approaches, cur_constraints, rng)
                if new is not None:
                    cur_constraints.append(new)
                # Augmented prompt; reuse the original full_instruction body.
                msgs = denial_spec.render(
                    full_instruction=str(item.get("A", "")),
                    constraints_block=format_constraints_block(cur_constraints),
                )
                per_item.append((item, msgs, denial_spec, msgs[-1]["content"]))

        # Generate this round's replies (batched).
        n_items = len(per_item)
        round_replies: list[str] = [None] * n_items  # type: ignore[list-item]
        with timed(f"denial_aut_infer: round {t}", n=n_items):
            if getattr(method, "supports_batch", False) and batch_size > 1:
                n_batches = (n_items + batch_size - 1) // batch_size
                bar = tqdm(range(0, n_items, batch_size), total=n_batches,
                           desc=f"denial_aut_infer:round{t}", unit="batch")
                for start in bar:
                    chunk = per_item[start:start + batch_size]
                    batch_msgs = [m for (_, m, _, _) in chunk]
                    replies = method.generate_batch(loader, batch_msgs, **gen_kwargs)
                    for j, ((item, _msgs, spec, user_str), reply) in enumerate(
                        zip(chunk, replies)
                    ):
                        rows.append(_denial_row(item, 0, cfg, method_slug, spec, user_str, reply))
                        round_replies[start + j] = reply
            else:
                for j, (item, msgs, spec, user_str) in enumerate(
                    tqdm(per_item, desc=f"denial_aut_infer:round{t}", unit="item")
                ):
                    reply = method.generate(loader, msgs, **gen_kwargs)
                    rows.append(_denial_row(item, 0, cfg, method_slug, spec, user_str, reply))
                    round_replies[j] = reply

        csv_path = stage_dir / f"{method_slug}_results.csv"
        _write_csv(rows, csv_path)
        outputs.append(hashed_output(csv_path, rows=len(rows)))
        prev_replies = [(r or "") for r in round_replies]

    return StageResult(
        name="denial_aut_infer",
        inputs={"abcd_data": hashed_input(cfg.aut.abcd_data)},
        outputs=outputs,
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "rounds": dcfg.rounds,
                "round0_prompt_id": round0_specs[0].id if round0_specs else None,
                "denial_prompt_id": dcfg.aut_denial_prompt_id,
                "reviewer_prompt_id": dcfg.reviewer_aut_prompt_id,
                "reviewer_model": dcfg.reviewer_backend.model,
                "batch_size": cfg.decoding_aut.batch_size,
            },
            "model": {"name": cfg.model_name},
            "decoding": _decoding_payload(cfg),
            "n_eval_items": len(eval_items),
            "eval_split": {
                "seed": cfg.aut.eval_seed,
                "holdout": cfg.aut.eval_holdout,
                "abcd_sha256": dataset.sha256,
            },
        },
    )
