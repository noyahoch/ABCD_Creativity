"""Orchestrator: the only call path that runs a stage.

Both ``cli.main`` and ``pipeline.run_pipeline`` go through
:func:`run_stage` — manifest writing, timing, and error handling are
guaranteed because they live in the wrapper, not in each stage.

Resume is sha-verified: a stage is "done" iff its ``_manifest.json``
exists, is success-status, and every recorded output file still has the
recorded sha256. Tampered outputs fail loudly unless ``--force-rerun``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from .config import RunConfig, VllmJudgeBackend
from .managed_vllm import managed_vllm
from .repro import RunContext, StageResult, read_manifest, sha256_file, write_manifest
from .stages import DEFAULT_ORDER, PREREQUISITES, STAGES


def _vllm_stages_in(cfg: RunConfig, stages: list[str]) -> list[int]:
    """Indices into ``stages`` whose backend resolves to vLLM.

    Judges only (aut_judge, macgyver_judge). The denial-prompting stages
    (denial_aut_infer, denial_macgyver_infer) are deliberately NOT listed:
    since 2026-07-19 they manage their own EPHEMERAL judge — the HF generator
    is unloaded for each reviewer round, the judge spins up on the freed
    GPU(s) via managed_vllm(), then tears down and the generator reloads.
    A pipeline-lifetime server here would co-reside with the generator, which
    a 24B-class model cannot afford (KV starvation on a 45 GiB card,
    confirmed 2026-07-18).
    """
    out: list[int] = []
    for i, s in enumerate(stages):
        if s == "aut_judge" and cfg.aut_judge is not None and \
                isinstance(cfg.aut_judge.backend, VllmJudgeBackend):
            out.append(i)
        elif s == "macgyver_judge" and cfg.macgyver_judge is not None and \
                isinstance(cfg.macgyver_judge.backend, VllmJudgeBackend):
            out.append(i)
    return out


class PipelineError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Single execution wrapper
# ---------------------------------------------------------------------------


def run_stage(name: str, ctx: RunContext) -> StageResult:
    """The ONLY way a stage is executed. Wraps with timing + manifest writing.

    The stage function itself is called with ``ctx`` and nothing else.
    Stage functions that accept test-injectable callables expose them via
    their own module-level entry points; the production path uses defaults.
    """
    if name not in STAGES:
        raise KeyError(f"unknown stage: {name!r}. Allowed: {sorted(STAGES)}")
    fn = STAGES[name]
    stage_dir = ctx.stage_dir(name)
    stage_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = fn(ctx)
        if not isinstance(result, StageResult):
            raise TypeError(
                f"stage {name!r} returned {type(result).__name__}; "
                "expected StageResult"
            )
        result.name = name
        if result.elapsed_s == 0.0:
            result.elapsed_s = time.time() - t0
        write_manifest(ctx, stage_dir, result, status="success")
        return result
    except Exception as e:
        empty = StageResult(name=name, elapsed_s=time.time() - t0)
        write_manifest(ctx, stage_dir, empty, status="failed", error=repr(e))
        raise


# ---------------------------------------------------------------------------
# Resume + dependency rules
# ---------------------------------------------------------------------------


def stage_is_done(ctx: RunContext, name: str, verify_outputs: bool = True) -> bool:
    """A stage is done iff its manifest is success AND output shas still match.

    With ``verify_outputs=False`` we only check the manifest status — useful
    for ``--dry-run`` and listing without re-hashing.
    """
    stage_dir = ctx.stage_dir(name)
    manifest_path = stage_dir / "_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        m = read_manifest(stage_dir)
    except Exception:
        return False
    if m.get("status") != "success":
        return False
    if not verify_outputs:
        return True
    for out in m.get("outputs", []):
        p = Path(out["path"])
        if not p.is_file():
            return False
        expected = (out.get("sha256") or "").removeprefix("sha256:")
        if expected and sha256_file(p) != expected:
            return False
    return True


def _prereqs_satisfied(ctx: RunContext, name: str) -> list[str]:
    """Return the list of missing/stale prerequisite stages.

    Special case for ``steer_extract``: it accepts EITHER ``steer_collect``
    (bridge collector) OR ``olson_collect`` (Olson collector) as its
    upstream. When ``cfg.olson`` is set, only ``olson_collect`` counts;
    otherwise only ``steer_collect`` counts. PREREQUISITES lists only the
    bridge default, so we override the resolution here.
    """
    prereqs = PREREQUISITES.get(name, ())
    if name == "steer_extract":
        if getattr(ctx.cfg, "olson", None) is not None:
            prereqs = ("olson_collect",)
        else:
            prereqs = ("steer_collect",)
    return [p for p in prereqs
            if not stage_is_done(ctx, p, verify_outputs=True)]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _aut_outputs_wanted(cfg: RunConfig) -> bool:
    """True iff any downstream consumer of aut_infer's CSV is configured.

    When false (e.g. the MacGyver-only ``ask_novelty`` method), running
    aut_infer is pure waste — no judge, no OCSAI, no brainstorm-select, and
    no aut_steering means nothing will read its rows. Gating aut_infer on
    this lets a config opt out of the AUT side entirely without per-config
    launcher plumbing.
    """
    return (
        cfg.aut_judge is not None
        or cfg.ocsai is not None
        or cfg.brainstorm_select is not None
        or cfg.aut_steering is not None
    )


def _filter_to_enabled(cfg: RunConfig, stages: list[str]) -> list[str]:
    """Drop stages whose config gate is off (e.g. macgyver without macgyver_data).

    Olson swap: when ``cfg.olson`` is set, ``steer_collect`` is replaced by
    ``olson_collect`` (and ``olson_collect`` is added if not already present),
    so the plain command ``python -m bridge_steering pipeline --config olson.yaml``
    Just Works without ``--only olson_collect,...``.
    """
    olson_on = getattr(cfg, "olson", None) is not None
    out: list[str] = []
    aut_wanted = _aut_outputs_wanted(cfg)
    for s in stages:
        if olson_on and s == "steer_collect":
            # Swap bridge collector → Olson collector exactly in place so the
            # default ordering is preserved.
            s = "olson_collect"
        if s == "aut_infer" and not aut_wanted:
            continue
        if s == "aut_brainstorm_select" and cfg.brainstorm_select is None:
            continue
        if s == "aut_ocsai" and cfg.ocsai is None:
            continue
        if s == "aut_novelty" and (cfg.originality is None or cfg.originality.ngram_index is None):
            continue
        if s == "aut_judge" and cfg.aut_judge is None:
            continue
        if s == "macgyver_infer" and cfg.macgyver_data is None:
            continue
        if s == "macgyver_novelty" and (
            cfg.macgyver_data is None
            or cfg.originality is None
            or cfg.originality.ngram_index is None
        ):
            continue
        if s == "macgyver_judge" and (cfg.macgyver_data is None or cfg.macgyver_judge is None):
            continue
        # Denial-prompting stages are opt-in via cfg.denial — without these
        # gates, DEFAULT_ORDER drags them into every full-pipeline run and
        # crashes at the stage's own `cfg.denial is None` guard, wasting all
        # prior HF/judge inference. Mirror the macgyver_* pattern: also gate
        # denial_macgyver_infer on macgyver_data, since its rounds need it.
        if s == "denial_aut_infer" and cfg.denial is None:
            continue
        if s == "denial_macgyver_infer" and (
            cfg.denial is None or cfg.macgyver_data is None
        ):
            continue
        if olson_on and s == "olson_collect":
            # Defensive: never emit olson_collect twice if caller already had it.
            if "olson_collect" in out:
                continue
        out.append(s)
    return out


def _expand_with_deps(ctx: RunContext, requested: list[str]) -> list[str]:
    """Expand requested stages to include missing prerequisite stages, in order."""
    want = list(requested)
    out: list[str] = []
    for s in DEFAULT_ORDER:
        if s in want:
            out.append(s)
            continue
        # Pull in upstream stages whose outputs are missing and downstream of
        # which the requested stages depend, transitively.
        if any(t in want and s in _ancestors(t) for t in DEFAULT_ORDER):
            if not stage_is_done(ctx, s):
                out.append(s)
    return out


def _ancestors(name: str) -> set[str]:
    """All transitive prerequisites of ``name``."""
    out: set[str] = set()
    stack = list(PREREQUISITES.get(name, ()))
    while stack:
        s = stack.pop()
        if s in out:
            continue
        out.add(s)
        stack.extend(PREREQUISITES.get(s, ()))
    return out


def run_pipeline(
    ctx: RunContext,
    only: Iterable[str] | None = None,
    with_deps: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> list[StageResult]:
    """Run the pipeline (or a subset). Returns one StageResult per stage executed.

    * ``only``: run just these stages (otherwise all of ``DEFAULT_ORDER``).
    * ``with_deps``: also run upstream prerequisites missing from disk.
    * ``resume``: skip stages whose manifest is success + outputs verified.
    * ``dry_run``: print plan, don't execute.

    With strict ``--only`` (no ``--with-deps``), missing prerequisites are an
    error before any work runs.
    """
    cfg = ctx.cfg
    requested = list(only) if only else list(DEFAULT_ORDER)
    requested = _filter_to_enabled(cfg, requested)
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        raise PipelineError(f"unknown stage(s): {unknown}")

    # Compute missing prereqs once; how we react depends on the mode.
    # Only flag prereqs that are neither on disk nor scheduled to run earlier
    # in this same invocation — otherwise the full-pipeline path trips on
    # stages that *will* be produced before their dependents.
    requested_set = set(requested)
    missing_per_stage: dict[str, list[str]] = {}
    for s in requested:
        missing = [p for p in _prereqs_satisfied(ctx, s)
                   if p not in requested_set]
        if missing:
            missing_per_stage[s] = missing

    if with_deps:
        to_run = _expand_with_deps(ctx, requested)
    elif missing_per_stage and not dry_run:
        lines = [f"  {stage} requires: {missing}"
                 for stage, missing in missing_per_stage.items()]
        raise PipelineError(
            "missing prerequisites — re-invoke with `--with-deps` or run "
            "those stages first:\n" + "\n".join(lines)
        )
    else:
        to_run = requested

    if resume:
        to_run = [s for s in to_run if not stage_is_done(ctx, s)]

    # The n-gram index is a one-off prerequisite that is not shipped; fail
    # before any GPU work rather than after inference.
    novelty = [s for s in to_run if s in ("aut_novelty", "macgyver_novelty")]
    index = cfg.originality.ngram_index if cfg.originality else None
    if novelty and index is not None and not Path(index).exists():
        msg = (f"n-gram index {index} not found (needed by {novelty}). Build it once with "
               f"`python scripts/build_ngram_index.py`, or leave these stages out via --only.")
        if not dry_run:
            raise PipelineError(msg)
        print(f"[dry-run] WARNING: {msg}")

    if dry_run:
        print(f"[dry-run] run_id={cfg.run_id}  run_dir={ctx.run_dir}")
        print(f"[dry-run] git={ctx.git_sha}{' (dirty)' if ctx.git_dirty else ''}")
        print(f"[dry-run] config_hash={ctx.config_hash}")
        if missing_per_stage and not with_deps:
            print("[dry-run] WARNING: missing prerequisites (would fail without "
                  "--with-deps):")
            for stage, missing in missing_per_stage.items():
                print(f"[dry-run]   {stage} requires: {missing}")
        print(f"[dry-run] stages to run, in order: {to_run}")
        return []

    # Start vLLM only around the contiguous slice of stages that need it, so
    # GPU-heavy inference stages above the slice still get the full device.
    vllm_idxs = _vllm_stages_in(cfg, to_run)
    if vllm_idxs and cfg.vllm_server is not None and cfg.vllm_server.enable:
        first, last = vllm_idxs[0], vllm_idxs[-1]
        pre, mid, post = to_run[:first], to_run[first:last + 1], to_run[last + 1:]
        # Anything in mid that is NOT a vLLM judge would run while the server
        # holds ~85% of VRAM — almost certainly OOMs an HF generator stage.
        # Force the failure to surface here instead of as a confusing vLLM
        # crash later. The DEFAULT_ORDER reorder puts judges last so this
        # should never trigger; the assertion documents the invariant.
        # Only HF-generator stages compete with vLLM for VRAM. CPU/HTTP stages
        # (aut_ocsai, *_novelty, aggregate) are fine inside the vLLM window.
        # NB: denial_* stages are NOT vllm stages anymore (2026-07-19): they
        # run their own ephemeral judge around each reviewer round, with the
        # HF generator unloaded, so they must stay OUTSIDE the slice.
        _GPU_HEAVY_NON_VLLM = {
            "steer_collect", "olson_collect", "steer_extract",
            "aut_infer", "aut_brainstorm_select", "macgyver_infer",
        }
        vllm_idx_set = set(vllm_idxs)
        bad = [
            to_run[i] for i in range(first, last + 1)
            if i not in vllm_idx_set and to_run[i] in _GPU_HEAVY_NON_VLLM
        ]
        if bad:
            raise PipelineError(
                "DEFAULT_ORDER puts HF-generator stage(s) between vLLM stages: "
                f"{bad}. Move them outside the vLLM slice (before the first "
                "vLLM stage or after the last one) so they don't fight vLLM "
                "for VRAM."
            )
    else:
        pre, mid, post = to_run, [], []

    results: list[StageResult] = []
    for s in pre:
        results.append(run_stage(s, ctx))
        # Each HF-using stage builds its own HFLoader; without this release,
        # the previous model's bf16 weights stay in the CUDA caching allocator
        # and the next loader fails (CPU offload or OOM). Hit hardest on big
        # models: Mistral-Small-24B at bf16 (~48 GB) only fits when the
        # previous load's allocation is actually freed.
        _release_cuda_cache()
    if mid:
        _release_cuda_cache()  # let vLLM see GPU memory freed by upstream HF stages
        with managed_vllm(cfg):
            for s in mid:
                results.append(run_stage(s, ctx))
                _release_cuda_cache()
    for s in post:
        results.append(run_stage(s, ctx))
        _release_cuda_cache()
    return results


def _release_cuda_cache() -> None:
    """Drop unreferenced CUDA allocations from prior stages so vLLM can grab VRAM.

    Each HF-using stage's loader goes out of scope when the stage function
    returns, but PyTorch's caching allocator keeps the memory reserved
    until somebody asks it to release. vLLM's startup-time free-memory
    check sees the cached chunks as "in use" and refuses to start.
    """
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
