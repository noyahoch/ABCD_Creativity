"""
Capture hidden states at decoder layers (residual stream or attention ``o_proj`` output).

Used by ``run_bridge_steering.py`` to average activations over completion tokens.
"""

from __future__ import annotations

from typing import Any

import torch

from .hook_targets import STEER_TARGET_RESIDUAL, tensor_from_hook_output, resolve_steer_hook_module


class ResidualStreamCache:
    """
    Register forward hooks on an HF CausalLM to capture tensors for one forward
    pass at a time.

    * ``steer_target="residual"`` (default): layer output (residual stream).
    * ``steer_target="attn_out"``: attention output projection (before residual add).

    After ``model(...)``, each probed layer's tensor is stored; use
    ``get_positions`` to slice token positions, then ``clear`` before the next
    forward (hooks remain registered).
    """

    def __init__(
        self,
        model: Any,
        layers: list[int],
        steer_target: str = STEER_TARGET_RESIDUAL,
    ) -> None:
        self.model = model
        self.layers = layers
        self.steer_target = steer_target
        self._hooks: list[Any] = []
        self._cache: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module: Any, input: Any, output: Any) -> None:
            hidden = tensor_from_hook_output(output)
            self._cache[layer_idx] = hidden.detach()

        return hook_fn

    def register(self) -> ResidualStreamCache:
        self.remove()
        for L in self.layers:
            mod = resolve_steer_hook_module(self.model, L, self.steer_target)
            h = mod.register_forward_hook(self._make_hook(L))
            self._hooks.append(h)
        return self

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._cache.clear()

    def clear(self) -> None:
        """Drop cached tensors; keep hooks (call before the next forward)."""
        self._cache.clear()

    def get_positions(self, layer_idx: int, positions: list[int]) -> torch.Tensor:
        """
        Return hidden states at ``positions`` for batch index 0.

        Shape: ``(len(positions), d_model)``.
        """
        hidden = self._cache[layer_idx]
        return hidden[0, positions, :]
