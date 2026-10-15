"""Regression test: probe layers come from cfg.steering.probe_layers.

In the legacy script ``run_bridge_steering.py:560`` had a hardcoded
``list(range(10, 25, 1))``. In the redesign the value lives in the typed
config; this test asserts the config value reaches the activation cache.
"""

from __future__ import annotations

from pathlib import Path

import torch

from bridge_steering.config import RunConfig
from bridge_steering.repro import build_run_context


def _make_collect_cache_with_layers(cfg: RunConfig, layers: list[int]) -> Path:
    """Pre-build a steer_collect/activations.pt for a custom layer set."""
    ctx = build_run_context(cfg)
    collect_dir = ctx.stage_dir("steer_collect")
    collect_dir.mkdir(parents=True, exist_ok=True)
    d_model = 4
    acts_B = {L: [torch.ones(d_model)] for L in layers}
    acts_D = {L: [torch.zeros(d_model)] for L in layers}
    torch.save({
        "acts_B": acts_B, "acts_D": acts_D, "probe_layers": layers,
        "steer_target": "residual", "steer_scale": "fixed",
        "window": 0, "collect_span": "assistant_full",
        "bridge_source": "fixed", "default_source": "D_banal",
        "model_name": "fake", "abcd_data_path": str(cfg.aut.abcd_data),
        "abcd_sha256": "deadbeef",
        "eval_indices": [], "train_indices": [0],
        "eval_seed": 1, "eval_holdout": 0,
        "aut_prompt_id": "aut_user_object_only_v1",
        "aut_prompt_sha256": "sha256:00",
    }, collect_dir / "activations.pt")
    return ctx.stage_dir("steer_extract") / "steering_vectors.pt"


def test_probe_layers_567(base_run_cfg):
    from bridge_steering.stages.steer_extract import run

    base_run_cfg.steering.probe_layers = [5, 6, 7]
    pt = _make_collect_cache_with_layers(base_run_cfg, [5, 6, 7])
    run(build_run_context(base_run_cfg))
    state = torch.load(pt, map_location="cpu", weights_only=False)
    assert sorted(state["v_bridge"]) == [5, 6, 7]
    assert state["probe_layers"] == [5, 6, 7]


def test_probe_layers_single_layer(base_run_cfg):
    """A change to cfg actually changes the probed layer set."""
    from bridge_steering.stages.steer_extract import run

    base_run_cfg.steering.probe_layers = [12]
    pt = _make_collect_cache_with_layers(base_run_cfg, [12])
    run(build_run_context(base_run_cfg))
    state = torch.load(pt, map_location="cpu", weights_only=False)
    assert sorted(state["v_bridge"]) == [12]


def test_probe_layers_default_matches_plan(base_run_cfg):
    """SteeringConfig default = list(range(10, 25)) per the plan."""
    assert base_run_cfg.steering.probe_layers == list(range(10, 25))
