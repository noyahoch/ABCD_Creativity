"""Stage: Olson-style activation collection (prompt-induced creativity).

Adapts the recipe of Olson et al. (2024) — arXiv:2412.06060 — to AUT.

Mechanical contrast vs. ``steer_collect``:

* ``steer_collect``: same prompt, two teacher-forced completions (gold B
  mechanism vs gold D banal). Activations are read while the model is
  teacher-forced through known text. Contrast is in the *completion*.
* ``olson_collect``: two different prompts (creative-instructed vs
  boring-instructed), each followed by *model-generated* text. Activations
  are read in a teacher-forced re-pass of (prompt + generation), one
  re-pass per prompt side. Contrast is in the *prompt*.

The output schema matches ``steer_collect``'s ``activations.pt`` so the
downstream ``steer_extract`` stage consumes Olson activations unchanged —
``steer_extract`` selects the input directory based on whether
``cfg.olson`` is set on the run config.

Faithfulness note: Olson's original paper uses Fan et al. (2018) writing
prompts and GPT-4o-generated "uncreative" rewrites, not instruction
prefixes on AUT items. The instruction-prefix shape used here is the
standard community translation when applying Olson-style contrast to
AUT; the underlying structural claim (contrast is prompt-induced, not
completion-induced) is preserved. Swap the prompt ids in ``cfg.olson``
to use any other realization.
"""

from __future__ import annotations

import time
from typing import Any

from ..config import RunConfig
from ..datasets import ABCDDataset
from ..prompts import PromptSpec, resolve
from ..repro import RunContext, StageResult, hashed_input, hashed_output
from .steer_collect import _collect_single_completion, _tokenize_full_sequence


# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without a model)
# ---------------------------------------------------------------------------


# Word swaps applied to ``item['A']`` (the dataset's full instruction) when
# rendering the BORING side of an Olson pair. Without these, an outer
# "Be conventional..." instruction would still wrap an inner "Give 8
# unconventional uses for X..." pulled from the dataset, making the boring
# prompt internally contradictory. Keeping the substitution table tiny and
# explicit so it's auditable.
_BORING_A_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("unconventional", "conventional"),
    ("unusual", "ordinary"),
)


def _to_boring_full_instruction(a: str) -> str:
    """Return a boring-coherent rewrite of the dataset's A field for the
    Olson boring side. Case-sensitive; the dataset uses lowercase markers."""
    out = a
    for src, dst in _BORING_A_SUBSTITUTIONS:
        out = out.replace(src, dst)
    return out


def build_olson_pairs(
    items: list[dict],
    creative_spec: PromptSpec,
    boring_spec: PromptSpec,
) -> list[dict[str, Any]]:
    """Build Olson contrast pairs.

    Each pair carries:
      * ``id``, ``source_id``, ``domain`` — row IDs (manifest provenance).
      * ``creative_msgs`` / ``boring_msgs`` — rendered chat-message lists.
      * ``creative_user`` / ``boring_user`` — bare user strings (for logging).
      * ``_creative_spec`` / ``_boring_spec`` — PromptSpec references for
        manifest hashing.

    Both renders receive ``object=…`` AND ``full_instruction=…`` so the
    underlying PromptSpec can use either placeholder. For the boring side,
    ``full_instruction`` is passed through :func:`_to_boring_full_instruction`
    so any embedded creative markers ("unconventional" / "unusual") are
    rewritten to their boring counterparts. This makes the rendered boring
    prompt internally coherent when the spec uses ``{full_instruction}``
    (v2_paper); harmless when the spec only uses ``{object}`` (v1).

    Note: unlike :func:`bridge_steering.stages.steer_collect.build_pairs` there is
    no teacher-forced completion in the pair dict — the completion side is
    produced by the model at collection time.
    """
    from ..prompts.aut import aut_object

    pairs: list[dict[str, Any]] = []
    for item in items:
        obj, _src = aut_object(item)
        full_a = str(item.get("A", ""))
        full_a_boring = _to_boring_full_instruction(full_a)
        creative_msgs = creative_spec.render(
            object=obj, full_instruction=full_a,
        )
        boring_msgs = boring_spec.render(
            object=obj, full_instruction=full_a_boring,
        )
        pairs.append({
            "id": item["id"],
            "source_id": item.get("source_id", item["id"]),
            "domain": item.get("domain", ""),
            "creative_msgs": creative_msgs,
            "boring_msgs": boring_msgs,
            "creative_user": creative_msgs[-1]["content"],
            "boring_user": boring_msgs[-1]["content"],
            "_creative_spec": creative_spec,
            "_boring_spec": boring_spec,
        })
    return pairs


# ---------------------------------------------------------------------------
# Model-backed activation collection
# ---------------------------------------------------------------------------


def _generate_under_prompt(
    prompt_msgs: list[dict[str, str]],
    tokenizer,
    model,
    device,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    seed: int | None,
) -> str:
    """Run a one-shot generation under ``prompt_msgs``; return the new text.

    Deterministic given ``seed`` + decoding config. Used per (item, side) so
    re-runs reproduce the same generations and thus the same steering vector.
    """
    import torch
    from bridge_steering.chat_template import apply_chat_template_safe

    prompt_text = apply_chat_template_safe(
        tokenizer, prompt_msgs, add_generation_prompt=True
    )
    # Pass attention_mask explicitly. Llama's tokenizer sets pad == eos, so
    # without an explicit mask HF emits a warning and the model can't tell
    # padding from content — generations drift slightly. Cheaper to be explicit.
    tokenized = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    if seed is not None:
        torch.manual_seed(int(seed))
    do_sample = bool(temperature and temperature > 0.0)
    gen_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        out = model.generate(prompt_ids, **gen_kwargs)
    new_tokens = out[0][prompt_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def collect_olson_activations(
    train_pairs: list[dict[str, Any]],
    model,
    tokenizer,
    device,
    *,
    probe_layers: list[int],
    window: int,
    max_new_tokens: int,
    gen_temperature: float,
    gen_top_p: float | None,
    gen_seed: int | None,
    steer_target: str,
):
    """Generate model responses under each prompt, then teacher-force re-pass
    to collect activations over the first ``window`` generated-token positions.

    Returns ``(acts_B, acts_D, generations)`` where acts_B is the creative
    side, acts_D is the boring side, and generations is a list of dicts
    recording what the model produced (for manifest provenance).
    """
    import torch
    from bridge_steering.steering import ResidualStreamCache
    from tqdm import tqdm

    acts_B: dict[int, list] = {L: [] for L in probe_layers}
    acts_D: dict[int, list] = {L: [] for L in probe_layers}
    generations: list[dict[str, str]] = []

    cache = ResidualStreamCache(model, probe_layers, steer_target=steer_target)
    cache.register()
    try:
        for i, p in enumerate(tqdm(train_pairs, desc="Olson generate+collect")):
            # Use seed + offset per side so each (item, side) is reproducible
            seed_c = (gen_seed + 2 * i) if gen_seed is not None else None
            seed_b = (gen_seed + 2 * i + 1) if gen_seed is not None else None

            creative_text = _generate_under_prompt(
                p["creative_msgs"], tokenizer, model, device,
                max_new_tokens=max_new_tokens, temperature=gen_temperature,
                top_p=gen_top_p, seed=seed_c,
            )
            boring_text = _generate_under_prompt(
                p["boring_msgs"], tokenizer, model, device,
                max_new_tokens=max_new_tokens, temperature=gen_temperature,
                top_p=gen_top_p, seed=seed_b,
            )

            # Teacher-force re-pass with the model's own generations.
            # Cap window per item like steer_collect: min(window, completion_len).
            with torch.no_grad():
                tok_c = _tokenize_full_sequence(
                    p["creative_msgs"], creative_text, tokenizer, "assistant_full",
                )
                w_c = tok_c["n_completion_tokens"] if window <= 0 \
                    else min(window, tok_c["n_completion_tokens"])
                vecs_C = _collect_single_completion(
                    p["creative_msgs"], creative_text, tokenizer, model, device,
                    cache, probe_layers, w_c, "assistant_full",
                )
                tok_b = _tokenize_full_sequence(
                    p["boring_msgs"], boring_text, tokenizer, "assistant_full",
                )
                w_b = tok_b["n_completion_tokens"] if window <= 0 \
                    else min(window, tok_b["n_completion_tokens"])
                vecs_B_neg = _collect_single_completion(
                    p["boring_msgs"], boring_text, tokenizer, model, device,
                    cache, probe_layers, w_b, "assistant_full",
                )

            for L in probe_layers:
                acts_B[L].append(vecs_C[L])
                acts_D[L].append(vecs_B_neg[L])
            generations.append({
                "id": p["id"],
                "creative_text": creative_text,
                "boring_text": boring_text,
            })
    finally:
        cache.remove()

    return acts_B, acts_D, generations


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(ctx: RunContext) -> StageResult:
    from bridge_steering.models import HFLoader
    from bridge_steering.steering.hook_targets import validate_steer_scale, validate_steer_target
    import torch

    cfg: RunConfig = ctx.cfg
    if cfg.olson is None:
        raise ValueError(
            "olson_collect: cfg.olson is None. Set the `olson:` block in the "
            "run config to use this stage."
        )
    t0 = time.time()

    # --- Resolve the two Olson prompts ----------------------------------
    creative_spec = resolve(
        cfg.olson.creative_prompt_id, allow_archived=cfg.allow_archived_prompts,
    )
    boring_spec = resolve(
        cfg.olson.boring_prompt_id, allow_archived=cfg.allow_archived_prompts,
    )
    if creative_spec.id == boring_spec.id:
        raise ValueError(
            f"olson_collect: creative_prompt_id and boring_prompt_id must "
            f"differ (both = {creative_spec.id!r})"
        )

    # --- Data + split ---------------------------------------------------
    dataset = ABCDDataset.load(cfg.aut.abcd_data)
    split = dataset.split(cfg.aut.eval_seed, cfg.aut.eval_holdout)
    train_pairs = build_olson_pairs(split.train_items, creative_spec, boring_spec)

    if cfg.experimental is not None and cfg.experimental.verbose_stages:
        print(
            f"[olson_collect] creative={creative_spec.id}  "
            f"boring={boring_spec.id}  "
            f"probe_layers={cfg.olson.probe_layers}  "
            f"n_train={len(train_pairs)}  "
            f"max_new_tokens={cfg.olson.max_new_tokens}  window={cfg.olson.window}"
        )

    # --- Validate hook target / scale -----------------------------------
    steer_target = validate_steer_target(cfg.olson.steer_target)
    steer_scale = validate_steer_scale(cfg.olson.steer_scale)

    # --- Model ----------------------------------------------------------
    loader = HFLoader(model_name_or_path=cfg.model_name)
    model, tokenizer = loader.load()
    device = next(model.parameters()).device
    n_layers = len(model.model.layers)
    probe_layers = [L for L in cfg.olson.probe_layers if 0 <= L < n_layers]
    if not probe_layers:
        raise ValueError(
            f"No probe layers from cfg.olson.probe_layers={cfg.olson.probe_layers} "
            f"fall in [0, {n_layers}) for model {cfg.model_name!r}."
        )

    # --- Collect --------------------------------------------------------
    acts_B, acts_D, generations = collect_olson_activations(
        train_pairs, model, tokenizer, device,
        probe_layers=probe_layers,
        window=cfg.olson.window,
        max_new_tokens=cfg.olson.max_new_tokens,
        gen_temperature=cfg.olson.gen_temperature,
        gen_top_p=cfg.olson.gen_top_p,
        gen_seed=cfg.olson.gen_seed,
        steer_target=steer_target,
    )

    # --- Save (same schema as steer_collect/activations.pt) -------------
    stage_dir = ctx.stage_dir("olson_collect")
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_path = stage_dir / "activations.pt"
    torch.save({
        "acts_B": acts_B,                 # creative side
        "acts_D": acts_D,                 # boring side
        "probe_layers": probe_layers,
        "steer_target": steer_target,
        "steer_scale": steer_scale,
        "window": cfg.olson.window,
        "collect_span": "assistant_full",
        # Provenance fields kept compatible with steer_collect's schema so
        # steer_extract reads them without changes.
        "collector": "olson",
        "bridge_source": "olson_creative",
        "default_source": "olson_boring",
        "source_kwargs": {
            "creative_prompt_id": cfg.olson.creative_prompt_id,
            "boring_prompt_id": cfg.olson.boring_prompt_id,
            "max_new_tokens": cfg.olson.max_new_tokens,
            "gen_temperature": cfg.olson.gen_temperature,
            "gen_top_p": cfg.olson.gen_top_p,
            "gen_seed": cfg.olson.gen_seed,
        },
        "model_name": cfg.model_name,
        "abcd_data_path": str(cfg.aut.abcd_data.resolve()),
        "abcd_sha256": dataset.sha256,
        "eval_indices": split.eval_indices,
        "train_indices": split.train_indices,
        "eval_seed": cfg.aut.eval_seed,
        "eval_holdout": cfg.aut.eval_holdout,
        "aut_prompt_id": creative_spec.id,
        "aut_prompt_sha256": "sha256:" + creative_spec.sha256(),
        "boring_prompt_id": boring_spec.id,
        "boring_prompt_sha256": "sha256:" + boring_spec.sha256(),
        # Light generation provenance — useful for debugging without
        # bloating the artifact. Strip if disk size matters.
        "generation_preview": [
            {"id": g["id"],
             "creative_preview": g["creative_text"][:200],
             "boring_preview": g["boring_text"][:200]}
            for g in generations[:5]
        ],
    }, out_path)

    return StageResult(
        name="olson_collect",
        inputs={
            "abcd_data": hashed_input(cfg.aut.abcd_data),
            "model": {"name": cfg.model_name},
        },
        outputs=[hashed_output(out_path)],
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "creative_prompt_id": creative_spec.id,
                "boring_prompt_id": boring_spec.id,
                "probe_layers": probe_layers,
                "window": cfg.olson.window,
                "max_new_tokens": cfg.olson.max_new_tokens,
                "gen_temperature": cfg.olson.gen_temperature,
                "gen_top_p": cfg.olson.gen_top_p,
                "gen_seed": cfg.olson.gen_seed,
                "steer_target": steer_target,
                "steer_scale": steer_scale,
            },
            "model": {"name": cfg.model_name, "n_layers": n_layers},
            "n_train_pairs": len(train_pairs),
            "n_acts_per_layer": len(next(iter(acts_B.values()))) if acts_B else 0,
            "eval_split": {
                "seed": cfg.aut.eval_seed,
                "holdout": cfg.aut.eval_holdout,
                "n_eval": len(split.eval_indices),
                "abcd_sha256": dataset.sha256,
            },
            "prompts": {
                "creative": creative_spec.manifest_entry(),
                "boring": boring_spec.manifest_entry(),
            },
        },
    )
