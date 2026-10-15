"""Stages registry.

Each stage is a plain function ``run(ctx: RunContext) -> StageResult``.
``pipeline.run_stage(name, ctx)`` is the only call path that executes a
stage — it wraps with timing, manifest writing, and error handling.

To add a stage:
  1. Create ``stages/my_stage.py`` exposing ``def run(ctx) -> StageResult``.
  2. Add one line to ``STAGES`` below.
  3. Insert into ``DEFAULT_ORDER`` at the correct position.
"""

from __future__ import annotations

from typing import Callable

from ..repro import RunContext, StageResult
from . import (
    aggregate, aut_brainstorm_select, aut_infer, aut_judge, aut_novelty, aut_ocsai,
    denial_aut_infer, denial_macgyver_infer,
    macgyver_infer, macgyver_judge, macgyver_novelty,
    olson_collect, steer_collect, steer_extract,
)


StageFn = Callable[[RunContext], StageResult]


STAGES: dict[str, StageFn] = {
    "steer_collect":          steer_collect.run,
    "olson_collect":          olson_collect.run,
    "steer_extract":          steer_extract.run,
    "aut_infer":              aut_infer.run,
    "aut_brainstorm_select":  aut_brainstorm_select.run,
    "aut_ocsai":              aut_ocsai.run,
    "aut_novelty":            aut_novelty.run,
    "aut_judge":              aut_judge.run,
    "macgyver_infer":         macgyver_infer.run,
    "macgyver_novelty":       macgyver_novelty.run,
    "macgyver_judge":         macgyver_judge.run,
    "denial_aut_infer":       denial_aut_infer.run,
    "denial_macgyver_infer":  denial_macgyver_infer.run,
    "aggregate":              aggregate.run,
}


DEFAULT_ORDER: list[str] = [
    # All HF-generator inference up front so the L40S holds Llama on its own.
    "steer_collect",
    "steer_extract",
    "aut_infer",
    "aut_brainstorm_select",
    "macgyver_infer",
    # Denial-prompting stages (Lu et al. 2024) write per-round CSVs into the
    # aut_infer/ and macgyver_infer/ stage dirs alongside the regular
    # baseline_results.csv, so the standard ocsai/novelty/judge discovery
    # picks each round up as a separate "method" row. Both stages time-
    # multiplex generator and reviewer: the HF generator is unloaded around
    # each reviewer round and an ephemeral vLLM judge takes the freed GPU.
    "denial_aut_infer",
    "denial_macgyver_infer",
    # GPU-free / API-only post-processing.
    # NB: ``aut_novelty`` is NOT in the default order — the paper reports
    # OCSAI as the AUT originality signal, not n-gram originality. The
    # stage stays registered in STAGES so a caller can still ``--only
    # aut_novelty`` for inspection, but it doesn't run by default.
    "aut_ocsai",
    "macgyver_novelty",
    # All vLLM-served judges next, so managed_vllm spins Qwen-32B-AWQ up once
    # for a single contiguous slice instead of fighting Llama for VRAM.
    "aut_judge",
    "macgyver_judge",
    "aggregate",
]


# Upstream prerequisites: ``--only X`` requires these stages to have
# usable outputs unless the user passes ``--with-deps``.
PREREQUISITES: dict[str, tuple[str, ...]] = {
    "steer_collect":         (),
    "olson_collect":         (),
    "steer_extract":         ("steer_collect",),  # OR ("olson_collect",) — pipeline picks based on cfg.olson
    "aut_infer":             (),    # steering is optional; aut_steering pulls vectors in
    "aut_brainstorm_select": ("aut_infer",),
    "aut_ocsai":             ("aut_infer",),       # OR ("aut_brainstorm_select",) — both supplied
    "aut_novelty":           ("aut_infer",),       # OR ("aut_brainstorm_select",); pure-local
    "aut_judge":             ("aut_infer",),
    "macgyver_infer":        (),
    "macgyver_novelty":      ("macgyver_infer",),
    "macgyver_judge":        ("macgyver_infer",),
    "denial_aut_infer":      (),
    "denial_macgyver_infer": (),
    "aggregate":             (),    # opportunistic — picks up whatever exists
}
