"""Stage: MacGyver inference on the filtered eval set.

Uses ``prompts.macgyver.build_user_prompt`` — the single source of truth
for the MacGyver user prompt. Per-row CSV mirrors the AUT inference
schema with MacGyver-specific extras.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .._timing import timed
from ..config import MacGyverInferenceConfig, RunConfig
from ..datasets import MacGyverDataset
from ..prompts.macgyver import build_user_prompt
from ..repro import RunContext, StageResult, hashed_input, hashed_output, sha256_str

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_kw):  # type: ignore[no-redef]
        return it


# ---------------------------------------------------------------------------
# Pure prompt builder (shared with the no-op steering case)
# ---------------------------------------------------------------------------


def _build_user_messages(item: dict, cfg: RunConfig):
    """Return ``(messages, spec_used, rendered_user_str)`` for one item."""
    mc = cfg.macgyver or MacGyverInferenceConfig()
    return build_user_prompt(item, mc.prompt_style)


# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------


MAC_INFER_COLUMNS = [
    "item_id", "task_type", "domain", "split", "source_id", "sample_idx", "seed",
    "problem_id", "problem_text",
    # Rendered user message (problem + MACGYVER_USER_SUFFIX for default style).
    # Mirrors OLD scripts/run_macgyver_inference.py's user_prompt column. The
    # judge consumes this so it sees the same context the generator saw.
    "user_prompt",
    "method", "prompt_id", "prompt_sha256", "prompt_input_hash",
    "model_name", "model_revision", "decoding_json",
    "raw_response", "parsed_response", "output_hash",
    # Ground-truth columns from the MacGyver xlsx (carried through for joins).
    "solvable", "unconventional", "ref_solution", "ref_label",
]


def _decoding_payload(cfg: RunConfig) -> dict:
    d = cfg.decoding_macgyver or cfg.decoding_aut
    return {
        "temperature": d.temperature, "top_p": d.top_p, "min_p": d.min_p,
        "max_new_tokens": d.max_new_tokens, "gen_seed": d.gen_seed,
        "num_inferences": d.num_inferences, "batch_size": d.batch_size,
    }


def _row(item, sample_idx, cfg, method_slug, spec, prompt_input, reply):
    decoding = _decoding_payload(cfg)
    return {
        "item_id": item["id"],
        "task_type": item.get("task_type", "macgyver"),
        "domain": item.get("domain", "macgyver"),
        "split": "eval",
        "source_id": item.get("source_id", item["id"]),
        "sample_idx": sample_idx,
        "seed": cfg.seed,
        "problem_id": item.get("problem_id", item["id"]),
        "problem_text": item.get("problem_text", item.get("problem", "")),
        "user_prompt": prompt_input,
        "method": method_slug,
        "prompt_id": spec.id,
        "prompt_sha256": "sha256:" + spec.sha256(),
        "prompt_input_hash": "sha256:" + sha256_str(prompt_input),
        "model_name": cfg.model_name,
        "model_revision": "",
        "decoding_json": json.dumps(decoding, sort_keys=True),
        "raw_response": reply,
        "parsed_response": (reply or "").strip(),
        "output_hash": "sha256:" + sha256_str(reply or ""),
        "solvable": item.get("solvable", True),
        "unconventional": item.get("unconventional", True),
        "ref_solution": item.get("solution", ""),
        "ref_label": item.get("label", ""),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MAC_INFER_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _build_methods(cfg: RunConfig, default_vectors_path: Path | None = None) -> list[Any]:
    """Same pattern as aut_infer: baseline + optional steered variants.

    If ``macgyver_steering.vectors`` is unset and ``default_vectors_path``
    exists on disk, use that — so the steer_extract output gets picked up
    automatically without a YAML edit.
    """
    from bridge_steering.methods import BaselineMethod, SteeredMethod

    methods: list[Any] = []
    s = cfg.macgyver_steering
    if s is None:
        methods.append(BaselineMethod())
        return methods
    if s.include_baseline:
        methods.append(BaselineMethod())
    vectors_path = s.vectors
    if vectors_path is None and default_vectors_path is not None \
            and default_vectors_path.is_file():
        vectors_path = default_vectors_path
    if vectors_path is None:
        if s.alphas:
            raise ValueError(
                "macgyver_steering.vectors must be set before macgyver_infer runs."
            )
        # baseline-only run (alphas: []) — no steered methods, no vectors needed
        return methods
    from .aut_infer import _append_tau_methods
    _append_tau_methods(methods, s, vectors_path)
    if not s.include_fixed_steered:
        return methods
    if s.steer_layers_group is not None:
        for alpha in s.alphas:
            m = SteeredMethod(
                vectors_path=vectors_path, alpha=alpha,
                layer_indices=s.steer_layers_group,
                steer_mode=s.steer_mode, k_assist=s.k_assist,
                steer_start_offset=s.steer_start_offset,
            )
            m.slug = "steered_" + "_".join(f"L{L}" for L in s.steer_layers_group) + f"_a{alpha}"
            methods.append(m)
    else:
        layers = s.layers if s.layers else [None]
        for L in layers:
            for alpha in s.alphas:
                m = SteeredMethod(
                    vectors_path=vectors_path, alpha=alpha, layer_idx=L,
                    steer_mode=s.steer_mode, k_assist=s.k_assist,
                    steer_start_offset=s.steer_start_offset,
                )
                m.slug = "steered_" + (f"L{L}" if L is not None else "auto") + f"_a{alpha}"
                methods.append(m)
    return methods


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(ctx: RunContext) -> StageResult:
    from bridge_steering.models import HFLoader

    cfg = ctx.cfg
    t0 = time.time()
    if cfg.macgyver_data is None:
        raise ValueError("macgyver_infer requires cfg.macgyver_data")
    if cfg.decoding_macgyver is None:
        raise ValueError("macgyver_infer requires cfg.decoding_macgyver")

    mc = cfg.macgyver or MacGyverInferenceConfig()
    dataset = MacGyverDataset.load(cfg.macgyver_data, mc.subset, max_items=mc.max_items)

    if cfg.decoding_macgyver.num_inferences > 1 and cfg.decoding_macgyver.temperature <= 0:
        raise ValueError("decoding_macgyver.num_inferences > 1 requires temperature > 0")

    loader = HFLoader(model_name_or_path=cfg.model_name)
    with timed("macgyver_infer: load model"):
        loader.load()
    default_vec = ctx.stage_dir("steer_extract") / "steering_vectors.pt"
    methods = _build_methods(cfg, default_vectors_path=default_vec)

    stage_dir = ctx.stage_dir("macgyver_infer")
    stage_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict] = []

    # Pre-build all per-item prompts once (same for every method) so we can
    # batch by chunks of cfg.decoding_macgyver.batch_size.
    prompts_per_item: list[tuple[object, list[dict], object, str]] = []
    for item in dataset.items:
        msgs, spec, user_str = _build_user_messages(item, cfg)
        prompts_per_item.append((item, msgs, spec, user_str))

    gen_kwargs = {
        "max_new_tokens": cfg.decoding_macgyver.max_new_tokens,
        "temperature": cfg.decoding_macgyver.temperature,
        "do_sample": cfg.decoding_macgyver.temperature > 0,
        **{k: v for k, v in {"top_p": cfg.decoding_macgyver.top_p,
                             "min_p": cfg.decoding_macgyver.min_p}.items()
           if v is not None},
    }
    batch_size = max(1, int(cfg.decoding_macgyver.batch_size))

    # Match OLD scripts/run_macgyver_inference.py seeding behavior: seed the
    # global torch RNG once before sampling so generations are reproducible
    # across runs at the configured `gen_seed`.
    if cfg.decoding_macgyver.gen_seed is not None and cfg.decoding_macgyver.temperature > 0:
        import torch as _torch
        s = int(cfg.decoding_macgyver.gen_seed)
        _torch.manual_seed(s)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(s)

    for method in methods:
        rows = []
        n_items = len(prompts_per_item) * cfg.decoding_macgyver.num_inferences
        with timed(f"macgyver_infer: {method.slug}", n=n_items):
            for sample_idx in range(cfg.decoding_macgyver.num_inferences):
                if getattr(method, "supports_batch", False) and batch_size > 1:
                    n_batches = (len(prompts_per_item) + batch_size - 1) // batch_size
                    bar = tqdm(range(0, len(prompts_per_item), batch_size),
                               total=n_batches,
                               desc=f"macgyver_infer:{method.slug}",
                               unit="batch")
                    for start in bar:
                        chunk = prompts_per_item[start:start + batch_size]
                        batch_msgs = [m for (_, m, _, _) in chunk]
                        replies = method.generate_batch(loader, batch_msgs, **gen_kwargs)
                        for (item, _msgs, spec, user_str), reply in zip(chunk, replies):
                            rows.append(_row(item, sample_idx, cfg,
                                             method.slug, spec, user_str, reply))
                else:
                    for item, msgs, spec, user_str in tqdm(
                            prompts_per_item, desc=f"macgyver_infer:{method.slug}", unit="item"):
                        reply = method.generate(loader, msgs, **gen_kwargs)
                        rows.append(_row(item, sample_idx, cfg,
                                         method.slug, spec, user_str, reply))
        csv_path = stage_dir / f"{method.slug}_results.csv"
        _write_csv(rows, csv_path)
        outputs.append(hashed_output(csv_path, rows=len(rows)))

    return StageResult(
        name="macgyver_infer",
        inputs={"macgyver_data": hashed_input(cfg.macgyver_data)},
        outputs=outputs,
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "methods": [m.slug for m in methods],
                "subset": mc.subset, "max_items": mc.max_items,
                "prompt_style": mc.prompt_style,
            },
            "model": {"name": cfg.model_name},
            "decoding": _decoding_payload(cfg),
            "n_items": len(dataset.items),
            "dataset_sha256": dataset.sha256,
        },
    )
