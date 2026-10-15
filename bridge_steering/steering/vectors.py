"""
Steering vector computation: weighted mean-difference bridge vectors and diagnostics.

Given activations from contrastive pairs (B = mechanism-mode, D = listing-mode),
compute per-layer centroids and v_bridge = mu_B - beta * mu_D (beta=1 is classic),
plus geometry and separability diagnostics.
"""

from __future__ import annotations

from typing import Any

import torch
import pandas as pd


class BridgeVectorComputer:
    """
    Holds activations from contrastive pairs and computes per-layer bridge
    vectors (centroids, v_bridge, stats).
    """

    def __init__(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        layers: list[int],
    ) -> None:
        self.acts_B = acts_B
        self.acts_D = acts_D
        self.layers = layers

    def compute(
        self,
        beta: float = 1.0,
    ) -> tuple[
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        pd.DataFrame,
    ]:
        """
        Compute per-layer centroids, v_bridge, and diagnostic stats.

        Args:
            beta: Weight on D centroid for mean-diff: v = mu_B - beta * mu_D (default 1.0).

        Returns:
            (centroids_B, centroids_D, v_bridge, stats_df)
        """
        centroids_B, centroids_D, stats_df = compute_bridge_vectors(
            self.acts_B, self.acts_D, self.layers, beta=beta
        )
        v_bridge = {
            L: centroids_B[L] - beta * centroids_D[L] for L in self.layers
        }
        return centroids_B, centroids_D, v_bridge, stats_df

    @staticmethod
    def select_best_layer(
        stats_df: pd.DataFrame,
        frac_positive_threshold: float = 1.0,
    ) -> int:
        """Select best layer index by relative signal among fully-separating layers."""
        return select_best_layer(stats_df, frac_positive_threshold)


def compute_bridge_vectors(
    acts_B: dict[int, list[torch.Tensor]],
    acts_D: dict[int, list[torch.Tensor]],
    layers: list[int],
    beta: float = 1.0,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], pd.DataFrame]:
    """
    Compute per-layer centroids and diagnostic stats for weighted mean-diff.

    For each layer L:
      centroids_B[L] = mean of acts_B[L]
      centroids_D[L] = mean of acts_D[L]
      v = mu_B - beta * mu_D  (used for separability stats; v_bridge dict filled by caller)

    Geometry diagnostics (per layer):
      cos(mu_B, mu_D), norms, ||mu_B - mu_D||, ||proj_parallel(mu_B onto mu_D)||,
      ||mu_B orthogonal to mu_D||.

    Args:
        acts_B: {layer: [tensor(d_model), ...]} for mechanism-mode condition.
        acts_D: {layer: [tensor(d_model), ...]} for listing-mode condition.
        layers: Layers to process.
        beta: Weight on D in steering direction v = mu_B - beta * mu_D.

    Returns:
        (centroids_B, centroids_D, stats_df)
    """
    centroids_B: dict[int, torch.Tensor] = {}
    centroids_D: dict[int, torch.Tensor] = {}
    layer_stats: list[dict[str, Any]] = []

    for L in layers:
        stack_D = torch.stack(acts_D[L])  # (N, d_model)
        stack_B = torch.stack(acts_B[L])  # (N, d_model)

        mu_D = stack_D.mean(dim=0)
        mu_B = stack_B.mean(dim=0)
        v = mu_B - beta * mu_D

        centroids_D[L] = mu_D
        centroids_B[L] = mu_B

        mu_B_f = mu_B.float()
        mu_D_f = mu_D.float()
        d_norm_sq = float(mu_D_f.dot(mu_D_f).item())
        if d_norm_sq > 1e-20:
            scale = float(mu_B_f.dot(mu_D_f).item()) / d_norm_sq
            proj_parallel = scale * mu_D_f
            par_norm = proj_parallel.norm().item()
            orth = mu_B_f - proj_parallel
            orth_norm = orth.norm().item()
        else:
            par_norm = 0.0
            orth_norm = mu_B_f.norm().item()

        cos_sim = torch.nn.functional.cosine_similarity(
            mu_B.unsqueeze(0).float(), mu_D.unsqueeze(0).float()
        ).item()
        v_norm = v.float().norm().item()
        mu_D_norm = mu_D.float().norm().item()
        mu_B_norm = mu_B.float().norm().item()
        diff_classic = mu_B - mu_D
        diff_classic_norm = diff_classic.float().norm().item()

        diffs = (stack_B - stack_D).float()
        v_unit = (v / (v.norm() + 1e-8)).float()
        projs = (diffs @ v_unit).numpy()

        rel_signal = v_norm / (mu_D_norm + 1e-8)
        cv = float(projs.std()) / (float(projs.mean()) + 1e-8)

        layer_stats.append({
            "layer": L,
            "beta": float(beta),
            "cos_centroids": round(cos_sim, 4),
            "|v_bridge|": round(v_norm, 2),
            "|mu_D|": round(mu_D_norm, 2),
            "|mu_B|": round(mu_B_norm, 2),
            "|mu_B_minus_mu_D|": round(diff_classic_norm, 2),
            "mu_B_par_D": round(par_norm, 4),
            "mu_B_orth_D": round(orth_norm, 4),
            "rel_signal": round(rel_signal, 4),
            "CV": round(cv, 4),
            "proj_mean": round(float(projs.mean()), 2),
            "proj_std": round(float(projs.std()), 2),
            "proj_min": round(float(projs.min()), 2),
            "frac_positive": round(float((projs > 0).mean()), 3),
        })

    stats_df = pd.DataFrame(layer_stats)
    return centroids_B, centroids_D, stats_df


def select_best_layer(
    stats_df: pd.DataFrame,
    frac_positive_threshold: float = 1.0,
) -> int:
    """
    Select the best layer for steering by relative signal among fully-separating layers.

    Picks the layer with highest |v_bridge| / |mu_D| (rel_signal) among those
    where frac_positive >= frac_positive_threshold. Falls back to all layers if none qualify.

    Returns:
        Best layer index (int).
    """
    candidates = stats_df[stats_df["frac_positive"] >= frac_positive_threshold]
    if candidates.empty:
        candidates = stats_df
    best_row = candidates.sort_values("rel_signal", ascending=False).iloc[0]
    return int(best_row["layer"])
