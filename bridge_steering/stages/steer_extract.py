"""Stage: turn collected activations into steering vectors.

Pure-numeric stage — no GPU, no model. Reads
``<run_dir>/steer_collect/activations.pt``, dispatches to
``bridge_steering.steering.extractors.EXTRACTOR_REGISTRY[cfg.steering.method]``, and
writes ``<run_dir>/steer_extract/steering_vectors.pt`` plus per-layer
``stats.csv``.

Splitting collect ↔ extract lets you swap extractors (mean_diff vs.
b_only vs. b_perp) without re-running the model. Each (collect, extract)
pair gets its own manifest and is byte-identical for the same inputs.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import RunConfig
from ..repro import RunContext, StageResult, hashed_input, hashed_output


def run(ctx: RunContext) -> StageResult:
    import torch
    from bridge_steering.steering import select_best_layer
    from bridge_steering.steering.extractors import EXTRACTOR_REGISTRY

    cfg = ctx.cfg
    t0 = time.time()

    # --- Load the collected activations ----------------------------------
    # When cfg.olson is set, read from the Olson collector's output instead
    # of steer_collect's. Both produce the same activations.pt schema, so
    # the rest of this stage is identical.
    collect_stage = "olson_collect" if cfg.olson is not None else "steer_collect"
    collect_dir = ctx.stage_dir(collect_stage)
    acts_path = collect_dir / "activations.pt"
    if not acts_path.is_file():
        raise FileNotFoundError(
            f"steer_extract: activations not found at {acts_path}. "
            f"Run the {collect_stage} stage first (or use --with-deps)."
        )
    cache = torch.load(acts_path, map_location="cpu", weights_only=False)
    acts_B: dict[int, list] = cache["acts_B"]
    acts_D: dict[int, list] = cache["acts_D"]
    probe_layers: list[int] = cache["probe_layers"]

    # --- Build the extractor (carries CLI-style kwargs) ------------------
    extractor_cls = EXTRACTOR_REGISTRY[cfg.steering.method]
    extractor = extractor_cls({
        **cfg.steering.method_kwargs,
        **cfg.steering.source_kwargs,
    })

    # --- Compute vectors -------------------------------------------------
    result = extractor.compute_vectors(acts_B, acts_D, probe_layers)
    steer_layer = select_best_layer(result.stats_df)
    v_steer = result.v_bridge[steer_layer].clone()

    # === EXPERIMENTAL [r2_extension] START =============================
    if cfg.experimental is not None and cfg.experimental.verbose_stages:
        print(
            f"[steer_extract] method={cfg.steering.method}  "
            f"selected_layer={steer_layer}  "
            f"||v||={float(v_steer.norm().item()):.4f}"
        )
    # === EXPERIMENTAL [r2_extension] END ===============================

    # --- Save steering checkpoint ---------------------------------------
    stage_dir = ctx.stage_dir("steer_extract")
    stage_dir.mkdir(parents=True, exist_ok=True)
    pt_path = stage_dir / "steering_vectors.pt"
    csv_path = stage_dir / "stats.csv"

    save_dict = {
        # Vectors
        "steer_layer": steer_layer,
        "v_steer": v_steer,
        "v_bridge": result.v_bridge,
        "stats_df": result.stats_df,
        # Strategy
        "method": cfg.steering.method,
        "bridge_source": cache.get("bridge_source"),
        "default_source": cache.get("default_source"),
        # Collection settings (forwarded from steer_collect manifest)
        "probe_layers": probe_layers,
        "window": cache.get("window"),
        "collect_span": cache.get("collect_span"),
        "steer_target": cache.get("steer_target"),
        "steer_scale": cache.get("steer_scale"),
        # Provenance (forwarded so steering_vectors.pt is self-describing)
        "model_name": cache.get("model_name"),
        "abcd_data_path": cache.get("abcd_data_path"),
        "abcd_sha256": cache.get("abcd_sha256"),
        "eval_indices": cache.get("eval_indices"),
        "train_indices": cache.get("train_indices"),
        "eval_seed": cache.get("eval_seed"),
        "eval_holdout": cache.get("eval_holdout"),
        # AUT prompt provenance — the bug-fix payload
        "aut_prompt_id": cache.get("aut_prompt_id"),
        "aut_prompt_sha256": cache.get("aut_prompt_sha256"),
    }
    save_dict.update(result.metadata)
    torch.save(save_dict, pt_path)
    result.stats_df.to_csv(csv_path, index=False)

    return StageResult(
        name="steer_extract",
        inputs={"activations": hashed_input(acts_path)},
        outputs=[
            hashed_output(pt_path),
            hashed_output(csv_path, rows=len(result.stats_df)),
        ],
        elapsed_s=time.time() - t0,
        notes={
            "stage_config": {
                "method": cfg.steering.method,
                "method_kwargs": cfg.steering.method_kwargs,
                "probe_layers": probe_layers,
            },
            "selected_layer": steer_layer,
            "v_steer_norm": float(v_steer.norm().item()),
            "extractor_metadata_keys": sorted(result.metadata),
        },
    )
