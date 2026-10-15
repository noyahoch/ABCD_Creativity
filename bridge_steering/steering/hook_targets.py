"""
Where to read/write activations for bridge steering.

* ``residual`` — decoder layer output (full block, residual stream).
* ``attn_out`` — output of the attention output projection (``o_proj`` / equivalent)
  before the residual add; often tighter norms than the full stream.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

STEER_TARGET_RESIDUAL = "residual"
STEER_TARGET_ATTN_OUT = "attn_out"

STEER_TARGET_CHOICES = frozenset({STEER_TARGET_RESIDUAL, STEER_TARGET_ATTN_OUT})

# How steering delta is scaled w.r.t. coefficient ``alpha`` (``c`` in notes).
# ``fixed``:  Δh = c * v
# ``relative_norm``: Δh = v * (c * ||h|| / ||v||) per token — stabilizes vs RMSNorm.
STEER_SCALE_FIXED = "fixed"
STEER_SCALE_RELATIVE_NORM = "relative_norm"

STEER_SCALE_CHOICES = frozenset({STEER_SCALE_FIXED, STEER_SCALE_RELATIVE_NORM})


def validate_steer_target(name: str) -> str:
    if name not in STEER_TARGET_CHOICES:
        raise ValueError(
            f"steer_target must be one of {sorted(STEER_TARGET_CHOICES)}, got {name!r}"
        )
    return name


def validate_steer_scale(name: str) -> str:
    if name not in STEER_SCALE_CHOICES:
        raise ValueError(
            f"steer_scale must be one of {sorted(STEER_SCALE_CHOICES)}, got {name!r}"
        )
    return name


def _decoder_layers(model: Any) -> Any:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise TypeError(
            "Model has no ``model.layers``; expected a HF CausalLM with stacked decoder blocks."
        )
    return layers


def _attention_submodule(layer: Any) -> Any:
    for name in ("self_attn", "attention", "self_attention"):
        m = getattr(layer, name, None)
        if m is not None:
            return m
    raise TypeError(
        f"Decoder layer {type(layer).__name__!r} has no self_attn / attention / self_attention."
    )


def _attention_output_proj(attn: Any) -> nn.Module:
    for name in ("o_proj", "dense", "wo", "proj"):
        m = getattr(attn, name, None)
        if isinstance(m, nn.Module):
            return m
    raise TypeError(
        f"Attention module {type(attn).__name__!r} has no o_proj-like Linear output."
    )


def resolve_steer_hook_module(model: Any, layer_idx: int, steer_target: str) -> Any:
    """
    Return the ``nn.Module`` to hook for the given layer and target.

    * ``residual``: ``model.model.layers[layer_idx]`` (post-block hidden states).
    * ``attn_out``: output projection inside that layer's attention submodule.
    """
    validate_steer_target(steer_target)
    layers = _decoder_layers(model)
    layer = layers[layer_idx]
    if steer_target == STEER_TARGET_RESIDUAL:
        return layer
    attn = _attention_submodule(layer)
    return _attention_output_proj(attn)


def tensor_from_hook_output(output: Any) -> Any:
    """Normalize hook ``output`` to a (B, T, D) tensor."""
    if isinstance(output, tuple):
        return output[0]
    return output
