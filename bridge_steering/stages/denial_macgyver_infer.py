"""Stage: MacGyver inference under denial prompting (Lu et al. 2024 adapted).

Mirrors ``denial_aut_infer`` but operates on the MacGyver dataset. Per-round
``denial_round{t}_results.csv`` files are written into ``macgyver_infer/`` so
``macgyver_novelty`` / ``macgyver_judge`` discover them as additional methods
alongside ``baseline_results.csv``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

from .._timing import timed
from ..config import MacGyverInferenceConfig, RunConfig, VllmJudgeBackend
from ..managed_vllm import managed_vllm
from ..datasets import MacGyverDataset
from ..prompts import resolve
from ..prompts.macgyver import build_user_prompt
from ..repro import RunContext, StageResult, hashed_input, hashed_output

from ._denial_common import (
    format_constraints_block,
    per_item_rng,
    review_batch,
    select_next_constraint,
)
from .macgyver_infer import (
    MAC_INFER_COLUMNS,
    _build_user_messages,
    _decoding_payload,
    _row as _mac_row,
    _write_csv,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_kw):  # type: ignore[no-redef]
        return it


def run(ctx: RunContext) -> StageResult:
    from bridge_steering.methods import BaselineMethod
    from bridge_steering.models import HFLoader

    cfg = ctx.cfg
    t0 = time.time()
    if cfg.denial is None:
        # Soft-skip: see denial_aut_infer for rationale.
        print("[denial_macgyver_infer] cfg.denial not set; skipping.")
        return StageResult(
            name="denial_macgyver_infer",
            elapsed_s=time.time() - t0,
            notes={"skipped": True, "reason": "cfg.denial not set"},
        )
    if cfg.macgyver_data is None:
        raise ValueError("denial_macgyver_infer requires cfg.macgyver_data")
    if cfg.decoding_macgyver is None:
        raise ValueError("denial_macgyver_infer requires cfg.decoding_macgyver")
    dcfg = cfg.denial

    mc = cfg.macgyver or MacGyverInferenceConfig()
    dataset = MacGyverDataset.load(cfg.macgyver_data, mc.subset, max_items=mc.max_items)

    if cfg.decoding_macgyver.num_inferences > 1:
        raise ValueError(
            "denial_macgyver_infer assumes num_inferences=1 (one solution per round per item)."
        )

    # --- Resolve prompts ---
    # Round 0: standard MacGyver inference prompt (resolved by prompt_style).
    round0_specs: list = []
    round0_user_strs: list[str] = []
    round0_msgs: list[list[dict]] = []
    items = list(dataset.items)
    for item in items:
        msgs, spec, user_str = _build_user_messages(item, cfg)
        round0_msgs.append(msgs)
        round0_specs.append(spec)
        round0_user_strs.append(user_str)
    denial_spec = resolve(
        dcfg.macgyver_denial_prompt_id, allow_archived=cfg.allow_archived_prompts
    )
    reviewer_spec = resolve(
        dcfg.reviewer_macgyver_prompt_id, allow_archived=cfg.allow_archived_prompts
    )

    # --- Model ---
    loader = HFLoader(model_name_or_path=cfg.model_name)
    with timed("denial_macgyver_infer: load model"):
        loader.load()
    method = BaselineMethod()
    if not hasattr(method, "slug"):
        method.slug = "baseline"

    # Output dirs: write per-round CSVs into macgyver_infer/, keep manifest
    # in our own stage_dir.
    out_dir = ctx.stage_dir("macgyver_infer")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = ctx.stage_dir("denial_macgyver_infer")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    gen_kwargs = {
        "max_new_tokens": cfg.decoding_macgyver.max_new_tokens,
        "temperature": cfg.decoding_macgyver.temperature,
        "do_sample": cfg.decoding_macgyver.temperature > 0,
        **{k: v for k, v in {"top_p": cfg.decoding_macgyver.top_p,
                             "min_p": cfg.decoding_macgyver.min_p}.items()
           if v is not None},
    }
    batch_size = max(1, int(cfg.decoding_macgyver.batch_size))

    if cfg.decoding_macgyver.gen_seed is not None and cfg.decoding_macgyver.temperature > 0:
        import torch as _torch
        s = int(cfg.decoding_macgyver.gen_seed)
        _torch.manual_seed(s)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(s)

    constraints_per_item: list[list[str]] = [[] for _ in items]
    prev_replies: list[str] = [""] * len(items)
    outputs: list[dict] = []

    for t in range(dcfg.rounds + 1):
        method_slug = f"denial_round{t}"
        rows: list[dict] = []

        if t == 0:
            per_item = list(zip(items, round0_msgs, round0_specs, round0_user_strs))
        else:
            # JUDGE TIME-MULTIPLEX (2026-07-19): generations and judgements
            # never share the GPUs. On a 45 GiB L40S the Qwen-32B-AWQ reviewer
            # (18.1 GiB weights) cannot coexist with a 24B half-shard (KV
            # starvation, confirmed twice), and a dedicated judge card caps the
            # campaign at 1-wide. So each reviewer round: release the HF
            # generator -> bring the judge up on THIS worker's own GPUs
            # (VLLM_CUDA_VISIBLE_DEVICES, set by the driver, pins it inside
            # the pair) -> review -> tear the judge down -> reload the
            # generator (~7 s from page cache). The torch RNG is untouched by
            # unload/reload, so generation streams are identical to an
            # uninterrupted run. managed_vllm() honours reuse_existing, so a
            # pre-started long-lived server (old regime) is used as-is.
            ephemeral = (
                isinstance(dcfg.reviewer_backend, VllmJudgeBackend)
                and cfg.vllm_server is not None
                and cfg.vllm_server.enable
            )
            if ephemeral:
                loader.unload()
            with timed(f"denial_macgyver_infer: reviewer round {t} (incl. judge lifecycle)",
                       n=len(prev_replies)):
                with (managed_vllm(cfg) if ephemeral else contextlib.nullcontext()):
                    extracted = asyncio.run(
                        review_batch(dcfg, reviewer_spec, "macgyver_response", prev_replies)
                    )
            if ephemeral:
                with timed(f"denial_macgyver_infer: reload generator round {t}"):
                    loader.load()
            per_item = []
            for item, cur_constraints, approaches in zip(items, constraints_per_item, extracted):
                rng = per_item_rng(cfg.seed, item.get("id", item.get("problem_id", "")), t)
                new = select_next_constraint(approaches, cur_constraints, rng)
                if new is not None:
                    cur_constraints.append(new)
                msgs = denial_spec.render(
                    problem=item["problem"],
                    constraints_block=format_constraints_block(cur_constraints),
                )
                per_item.append((item, msgs, denial_spec, msgs[-1]["content"]))

        n_items = len(per_item)
        round_replies: list[str] = [None] * n_items  # type: ignore[list-item]
        with timed(f"denial_macgyver_infer: round {t}", n=n_items):
            if getattr(method, "supports_batch", False) and batch_size > 1:
                n_batches = (n_items + batch_size - 1) // batch_size
                bar = tqdm(range(0, n_items, batch_size), total=n_batches,
                           desc=f"denial_macgyver_infer:round{t}", unit="batch")
                for start in bar:
                    chunk = per_item[start:start + batch_size]
                    batch_msgs = [m for (_, m, _, _) in chunk]
                    replies = method.generate_batch(loader, batch_msgs, **gen_kwargs)
                    for j, ((item, _msgs, spec, user_str), reply) in enumerate(
                        zip(chunk, replies)
                    ):
                        rows.append(_mac_row(item, 0, cfg, method_slug, spec, user_str, reply))
                        round_replies[start + j] = reply
            else:
                for j, (item, msgs, spec, user_str) in enumerate(
                    tqdm(per_item, desc=f"denial_macgyver_infer:round{t}", unit="item")
                ):
                    reply = method.generate(loader, msgs, **gen_kwargs)
                    rows.append(_mac_row(item, 0, cfg, method_slug, spec, user_str, reply))
                    round_replies[j] = reply

        csv_path = out_dir / f"{method_slug}_results.csv"
        _write_csv(rows, csv_path)
        outputs.append(hashed_output(csv_path, rows=len(rows)))
        prev_replies = [(r or "") for r in round_replies]

    return StageResult(
        name="denial_macgyver_infer",
        inputs={"macgyver_data": hashed_input(cfg.macgyver_data)},
        outputs=outputs,
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "rounds": dcfg.rounds,
                "round0_prompt_id": round0_specs[0].id if round0_specs else None,
                "denial_prompt_id": dcfg.macgyver_denial_prompt_id,
                "reviewer_prompt_id": dcfg.reviewer_macgyver_prompt_id,
                "reviewer_model": dcfg.reviewer_backend.model,
                "subset": mc.subset, "max_items": mc.max_items,
                "batch_size": cfg.decoding_macgyver.batch_size,
            },
            "model": {"name": cfg.model_name},
            "decoding": _decoding_payload(cfg),
            "n_items": len(items),
            "dataset_sha256": dataset.sha256,
        },
    )
