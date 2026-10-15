"""Stage: AUT inference on the held-out eval set.

The bug-fix payload: ``_build_user_messages(item, cfg)`` is the **same**
function ``stages.steer_collect.build_pairs`` calls. They cannot disagree.

Per-row CSV columns are the shared row-id columns + AUT-specific extras +
audit columns (prompt id/hash, decoding, raw + parsed response, output
hash) — per the v3 manifest schema in PLAN.md.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

from .._timing import timed
from ..config import RunConfig, SteeringInferenceConfig, resolve_aut_prompt
from ..datasets import ABCDDataset
from ..prompts.aut import aut_object, build_user_prompt
from ..repro import RunContext, StageResult, hashed_input, hashed_output, sha256_str

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in main deps
    def tqdm(it, **_kw):  # type: ignore[no-redef]
        return it


# ---------------------------------------------------------------------------
# Pure-Python prompt builders (shared with steer_collect via the same prompts module)
# ---------------------------------------------------------------------------


def _resolved_subject(cfg: RunConfig, role: str = "inference") -> str:
    """Resolve the subject for ``role`` (steering | inference)."""
    override = getattr(cfg.aut.prompts, role, None)
    if override is not None and override.subject:
        return override.subject
    return cfg.aut.prompts.default_subject


def _build_user_messages(item: dict, cfg: RunConfig, role: str = "inference"):
    """Return ``(messages, spec_used, rendered_user_str)`` for one item.

    Honors per-role ``prompt_id`` overrides (mirrors the workaround in
    :func:`bridge_steering.stages.steer_collect.build_pairs`). When a role pins a
    spec by id, we render that spec with both ``object`` and
    ``full_instruction`` kwargs so the template can use whichever it
    references. Otherwise, fall back to subject-based resolution.
    """
    override = getattr(cfg.aut.prompts, role, None)
    if override is not None and override.prompt_id:
        spec = resolve_aut_prompt(cfg, role)
        obj, _ = aut_object(item)
        msgs = spec.render(
            object=obj,
            full_instruction=str(item.get("A", "")),
        )
        return msgs, spec, msgs[-1]["content"]
    subject = _resolved_subject(cfg, role)
    return build_user_prompt(item, subject)


# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------


# Stable column order matches the v3 PLAN.md schema.
AUT_INFER_COLUMNS = [
    # Row-id columns
    "item_id", "task_type", "domain", "split", "source_id", "sample_idx", "seed",
    # AUT-specific
    "object", "object_source",
    # Audit
    "method", "prompt_id", "prompt_sha256", "prompt_input_hash",
    "model_name", "model_revision", "decoding_json",
    # ``user_prompt`` is the rendered chat-user string (same one Llama saw),
    # needed by downstream aut_ocsai to pass to the OCSAI endpoint as the
    # ``prompt`` field. OLD-pipeline parity: ``run_ocsai_scoring`` reads it
    # from the inference CSV's ``user_prompt`` column.
    "user_prompt",
    "raw_response", "parsed_response", "output_hash",
]


# Bullet/numbered prefixes and minimum length: ported from
# scripts.run_aut_inference._parse_uses_from_reply.
_PREAMBLE_RE = re.compile(
    r"(?i)^(here are|the following|below are|sure|of course|certainly|"
    r"i('d| would| can| will)|let me|give \d+|(\d+ )?(unconventional|unusual|creative|alternative) uses|"
    r"uses for)",
)
_MIN_USE_WORDS = 4


def parse_uses_from_reply(reply: str) -> list[str]:
    """Extract individual uses from a model reply.

    Drops preamble lines and short fragments. Same rules as the legacy
    script so downstream OCSAI/judge CSVs match.
    """
    out: list[str] = []
    for line in (reply or "").split("\n"):
        line = line.strip()
        line = re.sub(r"^[\-\*\•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip().rstrip(":")
        if not line or len(line.split()) < _MIN_USE_WORDS:
            continue
        if _PREAMBLE_RE.search(line):
            continue
        out.append(line)
    return out


# ---------------------------------------------------------------------------
# Method builder
# ---------------------------------------------------------------------------


def _append_tau_methods(methods: list[Any], s: "SteeringInferenceConfig",
                        vectors_path: Path) -> None:
    """Add one TauSteeredMethod per (alpha, tau) combo when ``s.taus`` is set.

    Shared by aut_infer and macgyver_infer so both stages build identical
    tau arms from the same config fields.
    """
    if not s.taus:
        return
    from bridge_steering.methods import TauSteeredMethod

    if s.distribution is None:
        raise ValueError("steering.taus is set but steering.distribution is None")
    if s.steer_layers_group is None:
        raise ValueError("tau methods require steer_layers_group")
    for alpha in s.alphas:
        for tau in s.taus:
            m = TauSteeredMethod(
                distribution_path=s.distribution,
                tau=tau,
                rank=s.tau_rank,
                vectors_path=vectors_path,
                alpha=alpha,
                layer_indices=s.steer_layers_group,
                steer_mode=s.steer_mode,
                k_assist=s.k_assist,
                steer_start_offset=s.steer_start_offset,
            )
            m.slug = (f"steered_tau{tau}_"
                      + "_".join(f"L{L}" for L in s.steer_layers_group)
                      + f"_a{alpha}")
            methods.append(m)


def _build_methods(cfg: RunConfig, default_vectors_path: Path | None = None) -> list[Any]:
    """Build the method instances to evaluate.

    Always includes ``baseline``. Adds one ``SteeredMethod`` per
    (layer, alpha) combo when ``cfg.aut_steering`` is set.

    If ``aut_steering.vectors`` is unset and ``default_vectors_path`` exists
    on disk, use that — lets the orchestrator wire steer_extract's output
    into aut_infer without mutating the YAML.
    """
    from bridge_steering.methods import BaselineMethod, SteeredMethod

    methods: list[Any] = []
    if cfg.aut_steering is None:
        methods.append(BaselineMethod())
        return methods

    s: SteeringInferenceConfig = cfg.aut_steering
    if s.include_baseline:
        methods.append(BaselineMethod())
    vectors_path = s.vectors
    if vectors_path is None and default_vectors_path is not None \
            and default_vectors_path.is_file():
        vectors_path = default_vectors_path
    if vectors_path is None:
        raise ValueError(
            "aut_steering.vectors must be set (or filled in from the "
            "steer_extract stage's output) before aut_infer runs."
        )
    s = SteeringInferenceConfig(
        vectors=vectors_path,
        alphas=s.alphas,
        layers=s.layers,
        steer_layers_group=s.steer_layers_group,
        steer_mode=s.steer_mode,
        k_assist=s.k_assist,
        steer_start_offset=s.steer_start_offset,
        include_baseline=s.include_baseline,
        include_fixed_steered=s.include_fixed_steered,
        distribution=s.distribution,
        taus=s.taus,
        tau_rank=s.tau_rank,
    )

    _append_tau_methods(methods, s, s.vectors)
    if not s.include_fixed_steered:
        return methods
    if s.steer_layers_group is not None:
        for alpha in s.alphas:
            m = SteeredMethod(
                vectors_path=s.vectors,
                alpha=alpha,
                layer_indices=s.steer_layers_group,
                steer_mode=s.steer_mode,
                k_assist=s.k_assist,
                steer_start_offset=s.steer_start_offset,
            )
            m.slug = "steered_" + "_".join(f"L{L}" for L in s.steer_layers_group) + f"_a{alpha}"
            methods.append(m)
    else:
        layers = s.layers if s.layers else [None]
        for L in layers:
            for alpha in s.alphas:
                m = SteeredMethod(
                    vectors_path=s.vectors,
                    alpha=alpha,
                    layer_idx=L,
                    steer_mode=s.steer_mode,
                    k_assist=s.k_assist,
                    steer_start_offset=s.steer_start_offset,
                )
                m.slug = "steered_" + (f"L{L}" if L is not None else "auto") + f"_a{alpha}"
                methods.append(m)
    return methods


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def _decoding_payload(cfg: RunConfig) -> dict:
    d = cfg.decoding_aut
    return {
        "temperature": d.temperature, "top_p": d.top_p, "min_p": d.min_p,
        "max_new_tokens": d.max_new_tokens, "gen_seed": d.gen_seed,
        "num_inferences": d.num_inferences, "batch_size": d.batch_size,
    }


def _row(
    item: dict, sample_idx: int, cfg: RunConfig, split: str,
    method_slug: str, spec, prompt_input: str, reply: str,
) -> dict:
    obj, obj_src = aut_object(item)
    decoding = _decoding_payload(cfg)
    return {
        "item_id": item["id"],
        "task_type": item.get("task_type", "aut"),
        "domain": item.get("domain", ""),
        "split": split,
        "source_id": item.get("source_id", item["id"]),
        "sample_idx": sample_idx,
        "seed": cfg.seed,
        "object": obj,
        "object_source": obj_src,
        "method": method_slug,
        "prompt_id": spec.id,
        "prompt_sha256": "sha256:" + spec.sha256(),
        "prompt_input_hash": "sha256:" + sha256_str(prompt_input),
        "model_name": cfg.model_name,
        "model_revision": "",          # populated when HF revision is resolvable
        "decoding_json": json.dumps(decoding, sort_keys=True),
        "user_prompt": prompt_input,
        "raw_response": reply,
        "parsed_response": json.dumps(parse_uses_from_reply(reply)),
        "output_hash": "sha256:" + sha256_str(reply or ""),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AUT_INFER_COLUMNS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(ctx: RunContext) -> StageResult:
    """Stage entry point. Loads the model, generates per (item × sample × method)."""
    from bridge_steering.models import HFLoader

    cfg = ctx.cfg
    t0 = time.time()

    # --- Data + split (eval items only) ----------------------------------
    dataset = ABCDDataset.load(cfg.aut.abcd_data)
    split = dataset.split(cfg.aut.eval_seed, cfg.aut.eval_holdout)
    eval_items = split.eval_items

    # --- Prepare per-item prompt resolution (uses prompts.aut.build_user_prompt)
    prompts_per_item: list[tuple[dict, list[dict], object, str]] = []
    for item in eval_items:
        msgs, spec, user_str = _build_user_messages(item, cfg, "inference")
        prompts_per_item.append((item, msgs, spec, user_str))

    # --- Build methods ----------------------------------------------------
    # Auto-resolve steering vectors from the conventional steer_extract output
    # if cfg.aut_steering.vectors is left unset in the YAML.
    default_vec = ctx.stage_dir("steer_extract") / "steering_vectors.pt"
    methods = _build_methods(cfg, default_vectors_path=default_vec)
    if cfg.decoding_aut.num_inferences > 1 and cfg.decoding_aut.temperature <= 0:
        raise ValueError(
            "decoding_aut.num_inferences > 1 requires temperature > 0"
        )

    # === EXPERIMENTAL [r2_extension] START =============================
    if cfg.experimental is not None and cfg.experimental.verbose_stages:
        infer_spec = prompts_per_item[0][2] if prompts_per_item else None
        s = cfg.aut_steering
        print(
            f"[aut_infer] prompt={infer_spec.id if infer_spec else None}  "
            f"methods={[m.slug for m in methods]}  "
            f"alphas={s.alphas if s is not None else None}  "
            f"layers={s.layers if s is not None else None}"
        )
    # === EXPERIMENTAL [r2_extension] END ===============================

    # --- Load model -------------------------------------------------------
    loader = HFLoader(model_name_or_path=cfg.model_name)
    with timed("aut_infer: load model"):
        loader.load()

    # --- Generate ---------------------------------------------------------
    stage_dir = ctx.stage_dir("aut_infer")
    stage_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict] = []

    gen_kwargs = {
        "max_new_tokens": cfg.decoding_aut.max_new_tokens,
        "temperature": cfg.decoding_aut.temperature,
        "do_sample": cfg.decoding_aut.temperature > 0,
        **{k: v for k, v in {"top_p": cfg.decoding_aut.top_p,
                             "min_p": cfg.decoding_aut.min_p}.items()
           if v is not None},
    }
    batch_size = max(1, int(cfg.decoding_aut.batch_size))

    # Seed RNG before sampling so AUT generations are reproducible across runs
    # and match the OLD-pipeline behavior (scripts/run_aut_inference.py:550-559
    # seeded the global torch RNG once per run). Without this, NEW used
    # whatever RNG state happened to be left after model load, producing
    # different sampled tokens vs OLD even at the same `gen_seed`.
    if cfg.decoding_aut.gen_seed is not None and cfg.decoding_aut.temperature > 0:
        import torch as _torch
        s = int(cfg.decoding_aut.gen_seed)
        _torch.manual_seed(s)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(s)

    for method in methods:
        rows: list[dict] = []
        n_items = len(prompts_per_item) * cfg.decoding_aut.num_inferences
        with timed(f"aut_infer: {method.slug}", n=n_items):
            for sample_idx in range(cfg.decoding_aut.num_inferences):
                if getattr(method, "supports_batch", False) and batch_size > 1:
                    n_batches = (len(prompts_per_item) + batch_size - 1) // batch_size
                    bar = tqdm(range(0, len(prompts_per_item), batch_size),
                               total=n_batches,
                               desc=f"aut_infer:{method.slug}",
                               unit="batch")
                    for start in bar:
                        chunk = prompts_per_item[start:start + batch_size]
                        batch_msgs = [m for (_, m, _, _) in chunk]
                        replies = method.generate_batch(loader, batch_msgs, **gen_kwargs)
                        for (item, _msgs, spec, user_str), reply in zip(chunk, replies):
                            rows.append(_row(item, sample_idx, cfg, "eval",
                                             method.slug, spec, user_str, reply))
                else:
                    for item, msgs, spec, user_str in tqdm(
                            prompts_per_item, desc=f"aut_infer:{method.slug}", unit="item"):
                        reply = method.generate(loader, msgs, **gen_kwargs)
                        rows.append(_row(item, sample_idx, cfg, "eval",
                                         method.slug, spec, user_str, reply))
        csv_path = stage_dir / f"{method.slug}_results.csv"
        _write_csv(rows, csv_path)
        outputs.append(hashed_output(csv_path, rows=len(rows)))

    return StageResult(
        name="aut_infer",
        inputs={"abcd_data": hashed_input(cfg.aut.abcd_data)},
        outputs=outputs,
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "methods": [m.slug for m in methods],
                "num_inferences": cfg.decoding_aut.num_inferences,
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
