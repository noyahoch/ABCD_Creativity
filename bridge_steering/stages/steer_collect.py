"""Stage: collect bridge-steering activations.

Replaces the activation-collection half of ``scripts/run_bridge_steering.py``.
The big fixes relative to the legacy script:

* The AUT user prompt is built by ``bridge_steering.prompts.aut.build_user_prompt``
  — the SAME function the inference stage uses. The bug where steering and
  inference disagreed on the prompt cannot reoccur.
* ``cfg.steering.probe_layers`` is read from the typed config, not a
  hardcoded ``range(10, 25)``.
* The resolved AUT prompt id + sha256 are written into the stage manifest,
  so months later you can tell which prompt produced which activations.

The actual model forward passes + caching reuse the existing
``bridge_steering.steering.cache.ResidualStreamCache`` and
``bridge_steering.steering.completion_sources.SOURCE_REGISTRY``. We only own the
glue + the prompt resolution.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

from ..config import RunConfig, resolve_aut_prompt
from ..datasets import ABCDDataset
from ..prompts.aut import build_user_prompt
from ..repro import RunContext, StageResult, hashed_input, hashed_output


# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without a model)
# ---------------------------------------------------------------------------


def format_d_completion(item: dict, default_source: str = "D_banal") -> str:
    """Teacher-forced D completion.

    * If ``item[default_source]`` is a list[str] (e.g. ``D_banal``, ``C``):
      formatted as one bullet per element, matching
      ``bridge_steering.data_prep.format_D_completion``.
    * If ``item[default_source]`` is a single string (e.g. ``D_generic``,
      the prose-paragraph control from
      ``data/abcd_aut_v3_bridges_with_c_dgeneric.json``): returned
      as-is, with no ``"- "`` bullet prefix, and a one-time warning that
      bullet-join was skipped.
    """
    if default_source not in item:
        raise ValueError(
            f"Item {item.get('id', '?')!r} missing default_source field "
            f"{default_source!r}; available keys: {sorted(item)}"
        )
    value = item[default_source]
    if isinstance(value, str):
        warnings.warn(
            f"format_d_completion: default_source={default_source!r} is a "
            f"string; returning prose as-is without '- ' bullet prefix. "
            f"(Item {item.get('id', '?')!r}.)",
            stacklevel=2,
        )
        return value
    return "\n".join(f"- {use}" for use in value)


def build_pairs(
    items: list[dict],
    cfg: RunConfig,
) -> list[dict[str, Any]]:
    """Build the contrastive-pair dicts the CompletionSource expects.

    Each pair carries:
      * ``id``, ``source_id``, ``domain`` — row IDs (manifest + CSV provenance)
      * ``user_prompt`` — rendered user string
      * ``prompt_msgs`` — full chat-messages list
      * ``d_completion`` — teacher-forced D text
      * ``b_list`` — JSON ``B_list`` (CompletionSource needs this)
      * ``_aut_prompt_spec`` — the PromptSpec used (manifest provenance)
    """
    spec = resolve_aut_prompt(cfg, "steering")
    subject = (
        cfg.aut.prompts.steering.subject
        if cfg.aut.prompts.steering is not None and cfg.aut.prompts.steering.subject
        else cfg.aut.prompts.default_subject
    )
    pairs: list[dict[str, Any]] = []
    for item in items:
        b_list = item.get("B_list")
        if not isinstance(b_list, list) or len(b_list) == 0:
            raise ValueError(
                f"Bridge steering requires non-empty JSON B_list on every row "
                f"(id={item.get('id', '?')!r}). No fallback to top-level B."
            )
        msgs, spec_used, user_str = build_user_prompt(item, subject)
        # If the user pinned a prompt_id override, prefer it over subject.
        if (cfg.aut.prompts.steering is not None
                and cfg.aut.prompts.steering.prompt_id):
            spec_used = spec  # the override-resolved spec
            # Re-render with the pinned spec; assume same placeholders.
            msgs = spec_used.render(
                object=item.get("A_OBJECT") or "",
                full_instruction=str(item.get("A", "")),
            )
            user_str = msgs[-1]["content"]
        pair = {
            "id": item["id"],
            "source_id": item.get("source_id", item["id"]),
            "domain": item.get("domain", ""),
            "user_prompt": user_str,
            "prompt_msgs": msgs,
            "d_completion": format_d_completion(item, cfg.steering.default_source),
            "b_list": b_list,
            "_aut_prompt_spec": spec_used,
        }
        # Optional: gold creative completions for CAA-style C-vs-D baseline
        # (source = c_creative_bullets). Attached only when the dataset
        # carries a per-item 'C' field; sources that don't need it ignore it.
        c_field = item.get("C")
        if isinstance(c_field, list) and len(c_field) > 0:
            pair["c_list"] = list(c_field)
        pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# Model-backed activation collection (port of run_bridge_steering loop)
# ---------------------------------------------------------------------------


def _find_subsequence(haystack: list[int], needle: list[int], start_at: int = 0) -> int:
    if not needle:
        return -1
    n = len(needle)
    upper = len(haystack) - n + 1
    for i in range(max(0, start_at), max(0, upper)):
        if haystack[i : i + n] == needle:
            return i
    return -1


def _tokenize_full_sequence(prompt_msgs, completion, tokenizer, collect_span: str):
    """Port of run_bridge_steering.tokenize_full_sequence."""
    from bridge_steering.chat_template import apply_chat_template_safe

    prompt_text = apply_chat_template_safe(tokenizer, prompt_msgs, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    assistant_start = int(prompt_ids.shape[1])

    full_msgs = prompt_msgs + [{"role": "assistant", "content": completion}]
    full_text = apply_chat_template_safe(tokenizer, full_msgs, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
    full_ids_list = full_ids[0].tolist()
    span_start = assistant_start
    span_len = full_ids.shape[1] - assistant_start

    if collect_span == "assistant_body":
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        body_start = _find_subsequence(full_ids_list, completion_ids, start_at=assistant_start)
        if body_start >= 0:
            span_start = body_start
            span_len = len(completion_ids)

    return {
        "full_ids": full_ids,
        "assistant_start": span_start,
        "n_completion_tokens": span_len,
    }


def _collect_single_completion(
    prompt_msgs, completion, tokenizer, model, device, cache,
    probe_layers, effective_window, collect_span,
):
    """One teacher-forced forward pass; return per-layer mean activation over
    the first ``effective_window`` completion tokens (whole completion if 0)."""
    import torch

    tok = _tokenize_full_sequence(prompt_msgs, completion, tokenizer, collect_span)
    inp = tok["full_ids"].to(device)
    n_comp = tok["n_completion_tokens"]
    w = n_comp if effective_window <= 0 else min(effective_window, n_comp)

    model(inp, use_cache=False)
    start = tok["assistant_start"]
    positions = list(range(start, start + w))
    vecs = {}
    for L in probe_layers:
        vecs[L] = cache.get_positions(L, positions).mean(dim=0).cpu()
    cache.clear()
    return vecs


def _min_b_completion_tokens(train_pairs, source, tokenizer, model, collect_span):
    """Minimum tokenized length over all B completions across all pairs.
    Used to cap the B window when --window > 0."""
    min_comp = float("inf")
    for p in train_pairs:
        for b_text in source.get_b_completions(p, model, tokenizer):
            tok = _tokenize_full_sequence(p["prompt_msgs"], b_text, tokenizer, collect_span)
            min_comp = min(min_comp, tok["n_completion_tokens"])
    return int(min_comp) if min_comp != float("inf") else 0


def collect_activations(train_pairs, source, model, tokenizer, device,
                        probe_layers, w_d, w_b, collect_span, steer_target):
    """Collect per-layer B and D activation lists across train pairs.

    Same alignment rule as the legacy script: if a B-source returns N
    completions per pair, D is appended N times so ``len(acts_B) == len(acts_D)``.
    """
    import torch
    from bridge_steering.steering import ResidualStreamCache
    from tqdm import tqdm

    acts_D = {L: [] for L in probe_layers}
    acts_B = {L: [] for L in probe_layers}
    cache = ResidualStreamCache(model, probe_layers, steer_target=steer_target)
    cache.register()
    try:
        with torch.no_grad():
            for p in tqdm(train_pairs, desc="Caching activations"):
                vecs_D = _collect_single_completion(
                    p["prompt_msgs"], p["d_completion"], tokenizer,
                    model, device, cache, probe_layers, w_d, collect_span,
                )
                for b_text in source.get_b_completions(p, model, tokenizer):
                    for L in probe_layers:
                        acts_D[L].append(vecs_D[L])
                    vecs_B = _collect_single_completion(
                        p["prompt_msgs"], b_text, tokenizer,
                        model, device, cache, probe_layers, w_b, collect_span,
                    )
                    for L in probe_layers:
                        acts_B[L].append(vecs_B[L])
    finally:
        cache.remove()

    return acts_B, acts_D


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(ctx: RunContext) -> StageResult:
    """The stage entry point. Returns inputs/outputs/notes for the manifest."""
    from bridge_steering.models import HFLoader
    from bridge_steering.steering.completion_sources import SOURCE_REGISTRY
    from bridge_steering.steering.hook_targets import validate_steer_scale, validate_steer_target
    from bridge_steering.data_prep import TaskType
    import torch

    cfg = ctx.cfg
    t0 = time.time()

    # --- Data + split (eval indices are owned by the steer stage --------
    dataset = ABCDDataset.load(cfg.aut.abcd_data)
    split = dataset.split(cfg.aut.eval_seed, cfg.aut.eval_holdout)
    train_items = split.train_items
    train_subset_size = cfg.aut.train_subset_size
    if train_subset_size is not None:
        if train_subset_size < 1:
            raise ValueError(
                f"aut.train_subset_size must be >= 1 when set, got {train_subset_size}"
            )
        if train_subset_size > len(train_items):
            raise ValueError(
                f"aut.train_subset_size={train_subset_size} exceeds available "
                f"train items ({len(train_items)}) after eval_holdout="
                f"{cfg.aut.eval_holdout}"
            )
        train_items = train_items[:train_subset_size]
    train_pairs = build_pairs(train_items, cfg)
    spec_used = train_pairs[0]["_aut_prompt_spec"] if train_pairs else None

    # === EXPERIMENTAL [r2_extension] START =============================
    if cfg.experimental is not None and cfg.experimental.verbose_stages:
        print(
            f"[steer_collect] prompt={spec_used.id if spec_used else None}  "
            f"bridge_source={cfg.steering.bridge_source}  "
            f"default_source={cfg.steering.default_source}  "
            f"probe_layers={cfg.steering.probe_layers}  "
            f"n_train={len(train_pairs)}"
        )
    # === EXPERIMENTAL [r2_extension] END ===============================

    # --- Steering source + target --------------------------------------
    steer_target = validate_steer_target(cfg.steering.steer_target)
    steer_scale = validate_steer_scale(cfg.steering.steer_scale)
    source_cls = SOURCE_REGISTRY[cfg.steering.bridge_source]
    # Build a flat all-config dict so source classes that read from
    # argparse-style kwargs (mean_diff_beta, fixed_b_list_mode, ...) work.
    source_config = {
        **cfg.steering.source_kwargs,
        **cfg.steering.method_kwargs,
    }
    source = source_cls(source_config)
    source.setup(train_pairs, TaskType.AUT)

    # --- Model -----------------------------------------------------------
    loader = HFLoader(model_name_or_path=cfg.model_name)
    model, tokenizer = loader.load()
    device = next(model.parameters()).device
    n_layers = len(model.model.layers)
    probe_layers = [L for L in cfg.steering.probe_layers if 0 <= L < n_layers]
    if not probe_layers:
        raise ValueError(
            f"No probe layers from cfg.steering.probe_layers={cfg.steering.probe_layers} "
            f"fall in [0, {n_layers}) for model {cfg.model_name!r}."
        )

    # --- Window resolution ----------------------------------------------
    if cfg.steering.window <= 0:
        w_d, w_b = 0, 0
    else:
        min_d = min(
            _tokenize_full_sequence(p["prompt_msgs"], p["d_completion"], tokenizer,
                                    cfg.steering.collect_span)["n_completion_tokens"]
            for p in train_pairs
        )
        w_d = min(cfg.steering.window, min_d)
        min_b = _min_b_completion_tokens(train_pairs, source, tokenizer, model,
                                         cfg.steering.collect_span)
        w_b = min(cfg.steering.window, min_b)

    # --- Collect ---------------------------------------------------------
    acts_B, acts_D = collect_activations(
        train_pairs, source, model, tokenizer, device,
        probe_layers, w_d, w_b, cfg.steering.collect_span, steer_target,
    )

    # --- Save activations cache -----------------------------------------
    stage_dir = ctx.stage_dir("steer_collect")
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_path = stage_dir / "activations.pt"
    torch.save({
        "acts_B": acts_B,
        "acts_D": acts_D,
        "probe_layers": probe_layers,
        "steer_target": steer_target,
        "steer_scale": steer_scale,
        "window": cfg.steering.window,
        "collect_span": cfg.steering.collect_span,
        "bridge_source": cfg.steering.bridge_source,
        "default_source": cfg.steering.default_source,
        "source_kwargs": cfg.steering.source_kwargs,
        "model_name": cfg.model_name,
        "abcd_data_path": str(cfg.aut.abcd_data.resolve()),
        "abcd_sha256": dataset.sha256,
        "eval_indices": split.eval_indices,
        "train_indices": split.train_indices,
        "eval_seed": cfg.aut.eval_seed,
        "eval_holdout": cfg.aut.eval_holdout,
        "train_subset_size": train_subset_size,
        "n_train_used": len(train_items),
        "aut_prompt_id": spec_used.id if spec_used else None,
        "aut_prompt_sha256": ("sha256:" + spec_used.sha256()) if spec_used else None,
    }, out_path)

    result = StageResult(
        name="steer_collect",
        inputs={
            "abcd_data": hashed_input(cfg.aut.abcd_data),
            "model": {"name": cfg.model_name},
        },
        outputs=[hashed_output(out_path)],
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "bridge_source": cfg.steering.bridge_source,
                "default_source": cfg.steering.default_source,
                "probe_layers": probe_layers,
                "window": cfg.steering.window,
                "effective_window_D": w_d,
                "effective_window_B": w_b,
                "collect_span": cfg.steering.collect_span,
                "steer_target": steer_target,
                "steer_scale": steer_scale,
            },
            "model": {"name": cfg.model_name, "n_layers": n_layers},
            "n_train_pairs": len(train_pairs),
            "n_acts_per_layer_B": len(next(iter(acts_B.values()))) if acts_B else 0,
            "n_acts_per_layer_D": len(next(iter(acts_D.values()))) if acts_D else 0,
            "eval_split": {
                "seed": cfg.aut.eval_seed,
                "holdout": cfg.aut.eval_holdout,
                "n_eval": len(split.eval_indices),
                "abcd_sha256": dataset.sha256,
            },
            "train_subset": {
                "size_config": train_subset_size,
                "n_used": len(train_items),
                "n_available": len(split.train_items),
            },
        },
    )
    return result
