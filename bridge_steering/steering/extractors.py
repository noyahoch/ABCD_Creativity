"""
Vector extractors: control how collected activations are turned into
per-layer steering vectors.

This is one of two composable strategy axes in the steering framework:
  --b-source  (CompletionSource)  x  --method  (VectorExtractor)

Any source can be combined with any extractor.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch

from .vectors import BridgeVectorComputer


EXTRACTOR_REGISTRY: dict[str, type[VectorExtractor]] = {}


def register_extractor(cls: type[VectorExtractor]) -> type[VectorExtractor]:
    """Class decorator that adds a VectorExtractor subclass to the registry."""
    EXTRACTOR_REGISTRY[cls.name] = cls
    return cls


@dataclass
class ExtractionResult:
    """Standardised output from any VectorExtractor."""

    v_bridge: dict[int, torch.Tensor]
    stats_df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorExtractor(ABC):
    """
    Base class for steering-vector extraction methods.

    Subclasses implement ``compute_vectors`` which receives the already-collected
    per-layer activation lists (acts_B, acts_D) and returns an
    ``ExtractionResult``.

    Activation *collection* is handled by the script's collection loop together
    with a ``CompletionSource``; the extractor is only responsible for the
    aggregation / computation stage.
    """

    name: str  # registry key, set on subclass

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        """Compute per-layer steering vectors from collected activations.

        Args:
            acts_B: ``{layer: [tensor(d_model), ...]}`` B-condition activations.
            acts_D: ``{layer: [tensor(d_model), ...]}`` D-condition activations.
            probe_layers: Layer indices present in both dicts.

        Returns:
            An ``ExtractionResult`` with ``v_bridge``, ``stats_df``, and
            optional ``metadata``.
        """
        ...

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add method-specific CLI arguments.  Override in subclasses."""


# --------------------------------------------------------------------------- #
# Concrete extractors
# --------------------------------------------------------------------------- #


@register_extractor
class MeanDiffExtractor(VectorExtractor):
    """Weighted centroid difference v = mu_B - beta * mu_D (beta=1 is classic B-D)."""

    name = "mean_diff"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mean-diff-beta",
            type=float,
            default=1.0,
            metavar="BETA",
            help="For mean_diff: steering direction v = mu_B - BETA * mu_D (default 1.0)",
        )

    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        beta = float(self.config.get("mean_diff_beta", 1.0))
        computer = BridgeVectorComputer(acts_B, acts_D, probe_layers)
        centroids_B, centroids_D, v_bridge, stats_df = computer.compute(beta=beta)

        return ExtractionResult(
            v_bridge=v_bridge,
            stats_df=stats_df,
            metadata={
                "centroids_B": centroids_B,
                "centroids_D": centroids_D,
                "mean_diff_beta": beta,
            },
        )


@register_extractor
class BOnlyExtractor(VectorExtractor):
    """B-only extractor: use centroid_B directly as the steering vector.

    This intentionally ignores D when constructing v_bridge, which is useful
    for ablations where no D subtraction/projection is desired.
    """

    name = "b_only"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--no-scale-to-diff",
            action="store_true",
            help="Do not scale b_only (mu_B) to (B-D) norm (alpha not comparable to mean_diff)",
        )

    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        v_bridge: dict[int, torch.Tensor] = {}
        layer_stats: list[dict[str, Any]] = []
        centroids_B: dict[int, torch.Tensor] = {}
        centroids_D: dict[int, torch.Tensor] = {}
        # Default ON (mirrors b_perp): rescale mu_B to the (B-D) norm so alpha is
        # comparable to mean_diff / b_perp — the +B/-D decomposition then isolates
        # DIRECTION rather than raw magnitude.
        scale_to_diff = not self.config.get("no_scale_to_diff", False)

        for L in probe_layers:
            stack_B = torch.stack(acts_B[L]).float()
            stack_D = torch.stack(acts_D[L]).float()
            centroid_B = stack_B.mean(dim=0)
            centroid_D = stack_D.mean(dim=0)
            centroids_B[L] = centroid_B
            centroids_D[L] = centroid_D

            norm_b = centroid_B.norm().item()
            norm_diff = (centroid_B - centroid_D).norm().item()
            if scale_to_diff and norm_b > 1e-8 and norm_diff > 1e-8:
                v = centroid_B * (norm_diff / norm_b)
            else:
                v = centroid_B
            v_bridge[L] = v.to(acts_B[L][0].dtype)

            layer_stats.append(
                {
                    "layer": L,
                    "rel_signal": round(v.norm().item(), 4),
                    "frac_positive": 1.0,
                    "|b_only|": round(norm_b, 2),
                    "|b-d|": round(norm_diff, 2),
                    "scaled_to_diff": scale_to_diff,
                }
            )

        stats_df = pd.DataFrame(layer_stats)
        return ExtractionResult(
            v_bridge=v_bridge,
            stats_df=stats_df,
            metadata={
                "centroids_B": centroids_B,
                "centroids_D": centroids_D,
                "b_only": True,
                "scaled_to_diff": scale_to_diff,
            },
        )


@register_extractor
class DOnlyExtractor(VectorExtractor):
    """D-only extractor: use -centroid_D directly as the steering vector.

    Symmetric ablation to BOnlyExtractor. Where BOnly drops the -D term
    from ``v = mu_B - mu_D``, DOnly drops the +B term. Steering with a
    positive scale along ``v = -mu_D`` pushes activations AWAY from the
    banal (D) cluster without any pull toward the bridge (B) cluster.

    Tests whether the steering signal lives in the +B direction or the
    -D direction (or only in the B-D contrast itself).
    """

    name = "d_only"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--no-scale-to-diff",
            action="store_true",
            help="Do not scale d_only (-mu_D) to (B-D) norm (alpha not comparable to mean_diff)",
        )

    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        v_bridge: dict[int, torch.Tensor] = {}
        layer_stats: list[dict[str, Any]] = []
        centroids_B: dict[int, torch.Tensor] = {}
        centroids_D: dict[int, torch.Tensor] = {}
        # Default ON (mirrors b_perp): rescale -mu_D to the (B-D) norm so alpha is
        # comparable to mean_diff / b_perp — see BOnlyExtractor.
        scale_to_diff = not self.config.get("no_scale_to_diff", False)

        for L in probe_layers:
            stack_B = torch.stack(acts_B[L]).float()
            stack_D = torch.stack(acts_D[L]).float()
            centroid_B = stack_B.mean(dim=0)
            centroid_D = stack_D.mean(dim=0)
            centroids_B[L] = centroid_B
            centroids_D[L] = centroid_D

            norm_d = centroid_D.norm().item()
            norm_diff = (centroid_B - centroid_D).norm().item()
            if scale_to_diff and norm_d > 1e-8 and norm_diff > 1e-8:
                v = (-centroid_D) * (norm_diff / norm_d)
            else:
                v = -centroid_D
            v_bridge[L] = v.to(acts_D[L][0].dtype)

            layer_stats.append(
                {
                    "layer": L,
                    "rel_signal": round(v.norm().item(), 4),
                    "frac_positive": 1.0,
                    "|d_only|": round(norm_d, 2),
                    "|b-d|": round(norm_diff, 2),
                    "scaled_to_diff": scale_to_diff,
                }
            )

        stats_df = pd.DataFrame(layer_stats)
        return ExtractionResult(
            v_bridge=v_bridge,
            stats_df=stats_df,
            metadata={
                "centroids_B": centroids_B,
                "centroids_D": centroids_D,
                "d_only": True,
                "scaled_to_diff": scale_to_diff,
            },
        )


@register_extractor
class BPerpExtractor(VectorExtractor):
    """B orthogonal to D: project out the D component from B.

    v_bridge[L] = B_perp = centroid_B - proj_D(centroid_B)
                = centroid_B - (centroid_B · d_hat) * d_hat

    where d_hat = centroid_D / ||centroid_D||.

    This keeps only the part of B that is orthogonal to D — the
    mechanism-reasoning signal that is NOT shared with listing mode.
    Unlike mean_diff (B - D), this removes the D direction entirely
    rather than subtracting the D magnitude along D.

    Scaled to ||B - D|| norm so alpha is comparable to mean_diff.
    """

    name = "b_perp"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--no-scale-to-diff",
            action="store_true",
            help="Do not scale B_perp to (B-D) norm (alpha not comparable to mean_diff)",
        )

    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        v_bridge: dict[int, torch.Tensor] = {}
        layer_stats: list[dict[str, Any]] = []
        scale_to_diff = not self.config.get("no_scale_to_diff", False)

        for L in probe_layers:
            stack_B = torch.stack(acts_B[L]).float()
            stack_D = torch.stack(acts_D[L]).float()
            centroid_B = stack_B.mean(dim=0)
            centroid_D = stack_D.mean(dim=0)

            d_norm_sq = centroid_D.dot(centroid_D)
            if d_norm_sq > 1e-16:
                proj_D_B = (centroid_B.dot(centroid_D) / d_norm_sq) * centroid_D
            else:
                proj_D_B = torch.zeros_like(centroid_B)
            b_perp = centroid_B - proj_D_B

            norm_perp = b_perp.norm().item()
            diff = centroid_B - centroid_D
            norm_diff = diff.norm().item()

            if scale_to_diff and norm_perp > 1e-8 and norm_diff > 1e-8:
                v_bridge[L] = (b_perp * (norm_diff / norm_perp)).to(acts_B[L][0].dtype)
                rel_signal = norm_diff / (centroid_D.norm().item() + 1e-8)
            else:
                v_bridge[L] = b_perp.to(acts_B[L][0].dtype)
                rel_signal = norm_perp / (centroid_D.norm().item() + 1e-8)

            cos_perp_diff = torch.nn.functional.cosine_similarity(
                b_perp.unsqueeze(0), diff.unsqueeze(0)
            ).item()

            layer_stats.append({
                "layer": L,
                "rel_signal": round(rel_signal, 4),
                "frac_positive": 1.0,
                "|b_perp|": round(norm_perp, 2),
                "|b-d|": round(norm_diff, 2),
                "cos_perp_vs_diff": round(cos_perp_diff, 4),
            })

        stats_df = pd.DataFrame(layer_stats)

        return ExtractionResult(
            v_bridge=v_bridge,
            stats_df=stats_df,
            metadata={
                "centroids_B": {L: torch.stack(acts_B[L]).mean(dim=0) for L in probe_layers},
                "centroids_D": {L: torch.stack(acts_D[L]).mean(dim=0) for L in probe_layers},
                "b_perp": True,
            },
        )
