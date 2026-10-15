"""Bridge activation-steering utilities."""

from pathlib import Path
from typing import Any

from .cache import ResidualStreamCache
from .completion_sources import (
    CompletionSource,
    CrossObjectTrainPermutedB,
    FixedCompletion,
    MultiBConcatenated,
    SOURCE_REGISTRY,
    register_source,
)
from .extractors import (
    ExtractionResult,
    EXTRACTOR_REGISTRY,
    MeanDiffExtractor,
    VectorExtractor,
    register_extractor,
)
from .hook_targets import (
    STEER_SCALE_FIXED,
    STEER_SCALE_RELATIVE_NORM,
    STEER_TARGET_ATTN_OUT,
    STEER_TARGET_RESIDUAL,
    validate_steer_scale,
    validate_steer_target,
)
from .hooks import MultiSteeringHook, SteeringHook, steered_generate, steered_generate_batch
from .vectors import (
    BridgeVectorComputer,
    compute_bridge_vectors,
    select_best_layer,
)

__all__ = [
    # Completion sources (--b-source axis)
    "CompletionSource",
    "FixedCompletion",
    "MultiBConcatenated",
    "CrossObjectTrainPermutedB",
    "SOURCE_REGISTRY",
    "register_source",
    # Vector extractors (--method axis)
    "VectorExtractor",
    "ExtractionResult",
    "MeanDiffExtractor",
    "EXTRACTOR_REGISTRY",
    "register_extractor",
    # Original exports
    "BridgeVectorComputer",
    "ResidualStreamCache",
    "STEER_SCALE_FIXED",
    "STEER_SCALE_RELATIVE_NORM",
    "STEER_TARGET_ATTN_OUT",
    "STEER_TARGET_RESIDUAL",
    "MultiSteeringHook",
    "SteeringHook",
    "steered_generate",
    "steered_generate_batch",
    "validate_steer_scale",
    "validate_steer_target",
    "compute_bridge_vectors",
    "load_steering_vectors",
    "select_best_layer",
]


def load_steering_vectors(path: str | Path) -> tuple[int, Any]:
    """
    Load steering vectors saved by the steer_extract stage.

    The file is a torch checkpoint with keys ``steer_layer`` (int) and
    ``v_steer`` (tensor of shape ``(d_model,)``).

    To tell how the vector was computed, inspect the checkpoint:
      - ``state["method"]``: e.g. "mean_diff", "b_perp"
      - ``state["use_pca"]``: legacy key; False for checkpoints from current code
      - ``state["b_source"]``: includes "fixed", "cross_object_train_b", etc.
      - ``state["steer_target"]``: ``"residual"`` (default) or ``"attn_out"``
      - ``state["steer_scale"]``: ``"fixed"`` (default) or ``"relative_norm"``
      - ``state["eval_indices"]``, ``state["abcd_sha256"]``: eval split contract (when present)

    Returns:
        (layer_idx, v_steer) ready to pass to ``steered_generate``.
    """
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state["steer_layer"]), state["v_steer"]
