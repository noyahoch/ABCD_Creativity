"""
Activation-steering hook: adds α × v_steer during generation.

Injection site is configurable: full decoder layer output (residual stream) or
attention ``o_proj`` output (before the residual add).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from bridge_steering.chat_template import apply_chat_template_safe

from .hook_targets import (
    STEER_SCALE_FIXED,
    STEER_TARGET_RESIDUAL,
    resolve_steer_hook_module,
    validate_steer_scale,
)


class SteeringHook:
    """
    Hook that adds α * v_steer at layer L during generation.

    * ``steer_target="residual"``: add to the decoder layer output (default).
    * ``steer_target="attn_out"``: add to the attention output projection output.

    * ``steer_scale="fixed"`` (default): ``Δh = α v``.
    * ``steer_scale="relative_norm"``: ``Δh = v * (α ||h|| / ||v||)`` at each steered
      token (helps RMSNorm-heavy models such as Mistral).

    Modes:
      - "last_prompt_only": inject only at position last_prompt_pos on the first (prefill) pass.
      - "all_new_tokens": inject at every generated token + last prompt token on prefill.
      - "all": inject at every position on every forward pass.
      - "first_k_assistant_tokens": inject only for the first K decode steps,
         plus optionally at last prompt token on prefill.
    """

    def __init__(
        self,
        model: Any,
        layer_idx: int,
        v_steer: torch.Tensor,
        alpha: float,
        mode: str = "all_new_tokens",
        last_prompt_pos: int = -1,
        k_assist: int = 16,
        steer_start_offset: int = 0,
        also_inject_last_prompt: bool = True,
        steer_target: str = STEER_TARGET_RESIDUAL,
        steer_scale: str = STEER_SCALE_FIXED,
    ) -> None:
        self.model = model
        self.layer_idx = layer_idx
        self.steer_target = steer_target
        self.steer_scale = validate_steer_scale(str(steer_scale))
        self.device = next(model.parameters()).device
        self.v_steer = v_steer.to(self.device).to(next(model.parameters()).dtype)
        self.alpha = float(alpha)
        self.mode = mode
        self.last_prompt_pos = int(last_prompt_pos)
        self.k_assist = int(k_assist)
        self.steer_start_offset = max(0, int(steer_start_offset))
        self.also_inject_last_prompt = bool(also_inject_last_prompt)

        self._hook: Any = None
        self._decode_step: int = 0

    def _steering_delta(self, h: torch.Tensor) -> torch.Tensor:
        """Δh for current hidden slice ``h`` with shape (..., hidden_dim)."""
        # Move v to h's device on demand: with sharded models (device_map="auto"),
        # the steered layer may live on a different GPU than self.device
        # (which is pinned to the embedding's GPU in __init__).
        v = self.v_steer if self.v_steer.device == h.device else self.v_steer.to(h.device)
        if self.steer_scale == STEER_SCALE_FIXED:
            return self.alpha * v
        v_n = v.float().norm().clamp(min=1e-8)
        norms = h.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (self.alpha * norms / v_n).to(h.dtype)
        return scale * v

    def _hook_fn(self, module: Any, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        is_prefill = hidden.shape[1] > 1

        if self.mode == "last_prompt_only":
            if is_prefill and self.last_prompt_pos >= 0:
                hidden = hidden.clone()
                sl = hidden[:, self.last_prompt_pos, :]
                sl.add_(self._steering_delta(sl))

        elif self.mode == "all_new_tokens":
            hidden = hidden.clone()
            if is_prefill:
                if self.last_prompt_pos >= 0:
                    sl = hidden[:, self.last_prompt_pos, :]
                    sl.add_(self._steering_delta(sl))
            else:
                self._decode_step += 1
                if self._decode_step > self.steer_start_offset:
                    hidden.add_(self._steering_delta(hidden))

        elif self.mode == "all":
            hidden = hidden + self._steering_delta(hidden)

        elif self.mode == "first_k_assistant_tokens":
            hidden = hidden.clone()
            if is_prefill:
                if self.also_inject_last_prompt and self.last_prompt_pos >= 0:
                    sl = hidden[:, self.last_prompt_pos, :]
                    sl.add_(self._steering_delta(sl))
            else:
                self._decode_step += 1
                step_in_window = self._decode_step - self.steer_start_offset
                if 1 <= step_in_window <= self.k_assist:
                    hidden.add_(self._steering_delta(hidden))

        else:
            raise ValueError(f"Unknown SteeringHook mode: {self.mode!r}")

        if rest is not None:
            return (hidden,) + rest
        return hidden

    def register(self) -> "SteeringHook":
        """Register hook on the target submodule (layer or attention o_proj)."""
        self.remove()
        self._decode_step = 0
        mod = resolve_steer_hook_module(self.model, self.layer_idx, self.steer_target)
        self._hook = mod.register_forward_hook(self._hook_fn)
        return self

    def remove(self) -> None:
        """Remove the hook if registered."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self) -> "SteeringHook":
        return self.register()

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.remove()


class MultiSteeringHook:
    """Register several :class:`SteeringHook` instances (one per layer) for the same forward pass."""

    def __init__(self, hooks: Sequence[SteeringHook]) -> None:
        self._hooks = list(hooks)

    def register(self) -> "MultiSteeringHook":
        for h in self._hooks:
            h.register()
        return self

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()

    def __enter__(self) -> "MultiSteeringHook":
        return self.register()

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.remove()


def _normalize_layer_vector_pairs(
    layer_idx: int | None,
    v_steer: torch.Tensor | None,
    layer_vector_pairs: list[tuple[int, torch.Tensor]] | None,
) -> list[tuple[int, torch.Tensor]]:
    if layer_vector_pairs is not None:
        if not layer_vector_pairs:
            raise ValueError("layer_vector_pairs must be non-empty")
        if layer_idx is not None or v_steer is not None:
            raise ValueError(
                "Pass either layer_vector_pairs=... OR layer_idx and v_steer, not both"
            )
        return list(layer_vector_pairs)
    if layer_idx is None or v_steer is None:
        raise TypeError(
            "steered_generate requires layer_idx and v_steer unless layer_vector_pairs is set"
        )
    return [(layer_idx, v_steer)]


@torch.no_grad()
def steered_generate(
    loader: Any,
    prompt_msgs: list[dict],
    layer_idx: int | None = None,
    v_steer: torch.Tensor | None = None,
    alpha: float = 1.0,
    mode: str = "all_new_tokens",
    max_new_tokens: int = 384,
    do_sample: bool = False,
    temperature: float = 1.0,
    k_assist: int = 16,
    steer_start_offset: int = 0,
    also_inject_last_prompt: bool = True,
    steer_target: str = STEER_TARGET_RESIDUAL,
    steer_scale: str = STEER_SCALE_FIXED,
    layer_vector_pairs: list[tuple[int, torch.Tensor]] | None = None,
    **generate_kwargs: Any,
) -> str:
    """
    Run generation with activation steering applied via SteeringHook.

    Args:
        loader: HFLoader instance (or any object with .load() returning (model, tokenizer)).
        prompt_msgs: List of {"role": ..., "content": ...} messages.
        layer_idx: Layer at which to inject the steering vector (single-layer path).
        v_steer: Steering direction tensor (single-layer path).
        layer_vector_pairs: If set, inject at multiple layers in one forward pass:
            ``[(L0, v0), (L1, v1), ...]`` (same ``alpha`` / mode / target for all).
            Mutually exclusive with ``layer_idx`` / ``v_steer``.
        alpha: Steering strength multiplier.
        mode: One of "last_prompt_only", "all_new_tokens", "all", "first_k_assistant_tokens".
        max_new_tokens: Max tokens to generate.
        do_sample: Whether to use sampling.
        temperature: Sampling temperature.
        k_assist: Used when mode="first_k_assistant_tokens".
        steer_start_offset: Skip this many initial generated tokens before steering starts.
        also_inject_last_prompt: Also inject at last prompt token during prefill.
        steer_target: "residual" (layer out) or "attn_out" (attention o_proj out).
        steer_scale: "fixed" or "relative_norm" (Δh = v * (α||h||/||v||) per token).

    Returns:
        Generated text string.
    """
    model, tokenizer = loader.load()

    text = apply_chat_template_safe(
        tokenizer, prompt_msgs, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    last_prompt_pos = inputs["input_ids"].shape[1] - 1

    pairs = _normalize_layer_vector_pairs(layer_idx, v_steer, layer_vector_pairs)
    hooks = [
        SteeringHook(
            model,
            layer_idx=li,
            v_steer=vi,
            alpha=alpha,
            mode=mode,
            last_prompt_pos=last_prompt_pos,
            k_assist=k_assist,
            steer_start_offset=steer_start_offset,
            also_inject_last_prompt=also_inject_last_prompt,
            steer_target=steer_target,
            steer_scale=steer_scale,
        )
        for li, vi in pairs
    ]
    bundle = MultiSteeringHook(hooks)
    bundle.register()
    try:
        do_sample_eff = temperature > 0
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample_eff,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )
    finally:
        bundle.remove()

    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def steered_generate_batch(
    loader: Any,
    batch_msgs: list[list[dict]],
    layer_idx: int | None = None,
    v_steer: torch.Tensor | None = None,
    alpha: float = 1.0,
    mode: str = "all_new_tokens",
    max_new_tokens: int = 384,
    do_sample: bool = False,
    temperature: float = 1.0,
    k_assist: int = 16,
    steer_start_offset: int = 0,
    also_inject_last_prompt: bool = True,
    steer_target: str = STEER_TARGET_RESIDUAL,
    steer_scale: str = STEER_SCALE_FIXED,
    layer_vector_pairs: list[tuple[int, torch.Tensor]] | None = None,
    **generate_kwargs: Any,
) -> list[str]:
    """Batched steered generation using left-padding."""
    if len(batch_msgs) == 1:
        return [steered_generate(
            loader,
            batch_msgs[0],
            layer_idx,
            v_steer,
            alpha=alpha,
            mode=mode,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            k_assist=k_assist,
            steer_start_offset=steer_start_offset,
            also_inject_last_prompt=also_inject_last_prompt,
            steer_target=steer_target,
            steer_scale=steer_scale,
            layer_vector_pairs=layer_vector_pairs,
            **generate_kwargs,
        )]

    model, tokenizer = loader.load()

    texts = [
        apply_chat_template_safe(tokenizer, msgs, add_generation_prompt=True)
        for msgs in batch_msgs
    ]

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    finally:
        tokenizer.padding_side = orig_side

    prompt_len = inputs["input_ids"].shape[1]
    last_prompt_pos = prompt_len - 1

    pairs = _normalize_layer_vector_pairs(layer_idx, v_steer, layer_vector_pairs)
    hooks = [
        SteeringHook(
            model,
            layer_idx=li,
            v_steer=vi,
            alpha=alpha,
            mode=mode,
            last_prompt_pos=last_prompt_pos,
            k_assist=k_assist,
            steer_start_offset=steer_start_offset,
            also_inject_last_prompt=also_inject_last_prompt,
            steer_target=steer_target,
            steer_scale=steer_scale,
        )
        for li, vi in pairs
    ]
    bundle = MultiSteeringHook(hooks)
    bundle.register()
    try:
        do_sample_eff = temperature > 0
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample_eff,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )
    finally:
        bundle.remove()

    results: list[str] = []
    for seq in out:
        generated = seq[prompt_len:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True))
    return results
