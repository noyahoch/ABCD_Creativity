"""Unit tests for the steer_extract stage.

Synthetic activations only — no GPU, no model.

Verifies that each registered extractor (mean_diff, b_only, b_perp) produces
the mathematically expected vector on hand-crafted B / D activation lists,
and that the saved checkpoint carries the full reproducibility payload.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bridge_steering.config import RunConfig
from bridge_steering.repro import build_run_context


@pytest.fixture
def cache_with_acts(base_run_cfg: RunConfig, tmp_path: Path) -> Path:
    """Pre-build a steer_collect/activations.pt at the run dir."""
    ctx = build_run_context(base_run_cfg)
    collect_dir = ctx.stage_dir("steer_collect")
    collect_dir.mkdir(parents=True, exist_ok=True)

    # Hand-crafted: 3 pairs × 1 B variant per pair, d_model=4, layers [12, 13]
    d_model = 4
    layers = [12, 13]
    # Layer 12: B centroid = (1,1,1,1), D centroid = (1,0,0,0) so v = (0,1,1,1)
    acts_B = {12: [torch.tensor([1.0, 1.0, 1.0, 1.0]) for _ in range(3)],
              13: [torch.tensor([2.0, 0.0, 0.0, 0.0]) for _ in range(3)]}
    acts_D = {12: [torch.tensor([1.0, 0.0, 0.0, 0.0]) for _ in range(3)],
              13: [torch.tensor([0.0, 0.0, 0.0, 0.0]) for _ in range(3)]}
    torch.save({
        "acts_B": acts_B, "acts_D": acts_D, "probe_layers": layers,
        "steer_target": "residual", "steer_scale": "fixed",
        "window": 0, "collect_span": "assistant_full",
        "bridge_source": "fixed", "default_source": "D_banal",
        "model_name": "fake-model",
        "abcd_data_path": str(base_run_cfg.aut.abcd_data),
        "abcd_sha256": "deadbeef",
        "eval_indices": [0], "train_indices": [1, 2],
        "eval_seed": 7, "eval_holdout": 1,
        "aut_prompt_id": "aut_user_object_only_v1",
        "aut_prompt_sha256": "sha256:abc",
    }, collect_dir / "activations.pt")
    return collect_dir / "activations.pt"


def _run_with(cfg: RunConfig, method: str) -> Path:
    from bridge_steering.stages.steer_extract import run

    cfg.steering.method = method
    ctx = build_run_context(cfg)
    result = run(ctx)
    assert result.name == "steer_extract"
    return ctx.stage_dir("steer_extract") / "steering_vectors.pt"


def test_mean_diff_equals_centroid_difference(base_run_cfg, cache_with_acts):
    pt_path = _run_with(base_run_cfg, "mean_diff")
    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    # B = (1,1,1,1), D = (1,0,0,0) ⇒ v = (0,1,1,1)
    v12 = state["v_bridge"][12]
    assert torch.allclose(v12, torch.tensor([0.0, 1.0, 1.0, 1.0]))
    # B = (2,0,0,0), D = 0 ⇒ v = (2,0,0,0)
    v13 = state["v_bridge"][13]
    assert torch.allclose(v13, torch.tensor([2.0, 0.0, 0.0, 0.0]))
    # Checkpoint carries the AUT prompt provenance forward — the bug fix.
    assert state["aut_prompt_id"] == "aut_user_object_only_v1"
    assert state["aut_prompt_sha256"] == "sha256:abc"


def test_b_only_returns_centroid_B(base_run_cfg, cache_with_acts):
    pt_path = _run_with(base_run_cfg, "b_only")
    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    # mu_B rescaled to |B-D| (default): (1,1,1,1) * sqrt(3)/2
    assert torch.allclose(state["v_bridge"][12], torch.tensor([1.0, 1.0, 1.0, 1.0]) * 3 ** 0.5 / 2)
    assert torch.allclose(state["v_bridge"][13], torch.tensor([2.0, 0.0, 0.0, 0.0]))


def test_b_perp_is_orthogonal_to_centroid_D(base_run_cfg, cache_with_acts):
    pt_path = _run_with(base_run_cfg, "b_perp")
    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    # Layer 12: D = (1,0,0,0). B_perp should have a 0 in dim 0.
    v12 = state["v_bridge"][12].float()
    d12 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cos = torch.nn.functional.cosine_similarity(v12.unsqueeze(0), d12.unsqueeze(0)).item()
    assert abs(cos) < 1e-6, f"v12 should be perpendicular to D, cos={cos}"


def test_stats_csv_has_one_row_per_layer(base_run_cfg, cache_with_acts):
    import pandas as pd
    _run_with(base_run_cfg, "mean_diff")
    csv = base_run_cfg.run_dir() / "steer_extract" / "stats.csv"
    df = pd.read_csv(csv)
    assert sorted(df["layer"].tolist()) == [12, 13]


def test_missing_activations_raises(base_run_cfg):
    from bridge_steering.stages.steer_extract import run
    ctx = build_run_context(base_run_cfg)
    with pytest.raises(FileNotFoundError, match="activations not found"):
        run(ctx)
