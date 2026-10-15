"""
AUT inference methods: pluggable ways to produce model outputs.

Each method has a ``slug`` (used for output dirs / column names) and a
``generate(loader, messages, ...)`` that returns raw text.

Available methods
-----------------
* ``BaselineMethod``   – plain generation, no steering or extra prompting.
* ``SteeredMethod``    – activation-steered generation (requires steering_vectors.pt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch

from bridge_steering.steering import steered_generate, steered_generate_batch
from bridge_steering.steering.hook_targets import (
    STEER_SCALE_FIXED,
    STEER_TARGET_RESIDUAL,
    validate_steer_scale,
    validate_steer_target,
)


# ── Protocol ──────────────────────────────────────────────────────────────────


class AUTInferenceMethod(Protocol):
    """Protocol for an AUT inference method: slug + generate(loader, messages) -> str."""

    slug: str
    supports_batch: bool

    def generate(
        self,
        loader: Any,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str: ...

    def generate_batch(
        self,
        loader: Any,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> list[str]: ...


# ── Methods ───────────────────────────────────────────────────────────────────


class BaselineMethod:
    """Plain generation with no steering or extra prompting."""

    slug = "baseline"
    supports_batch = True

    def generate(
        self,
        loader: Any,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        return loader.generate(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )

    def generate_batch(
        self,
        loader: Any,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        return loader.generate_batch(
            batch_messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )


class SteeredMethod:
    """Generation with activation steering from a saved steering_vectors.pt.

    Args:
        vectors_path: Path to the saved .pt checkpoint.
        alpha: Steering strength multiplier (applied to every hooked layer).
        layer_idx: Override which layer to steer at. If None, uses the
                   auto-selected ``steer_layer`` from the checkpoint.
                   The checkpoint's ``v_bridge`` dict holds vectors for all
                   probed layers, so you can pick any of them.
        layer_indices: Steer at multiple layers in the **same** forward pass.
                       Each index must exist in ``v_bridge``. Mutually exclusive
                       with ``layer_idx``.
        steer_mode: Hook mode — "all_new_tokens" (default), "first_k_assistant_tokens",
                    "last_prompt_only", or "all".
        k_assist: Number of initial assistant tokens to steer when
                  steer_mode="first_k_assistant_tokens".
        steer_target: If set, override checkpoint ``steer_target`` ("residual" or
                      "attn_out"). If None, use the value saved in the checkpoint
                      (default ``residual`` for legacy files).
        steer_scale: If set, override checkpoint ``steer_scale`` ("fixed" or
                     "relative_norm"). If None, use checkpoint (default ``fixed``).
    """

    slug = "steered"
    supports_batch = True

    def __init__(
        self,
        vectors_path: str | Path,
        alpha: float = 1.0,
        layer_idx: int | None = None,
        layer_indices: list[int] | None = None,
        steer_mode: str = "all_new_tokens",
        k_assist: int = 16,
        steer_start_offset: int = 0,
        steer_target: str | None = None,
        steer_scale: str | None = None,
    ) -> None:
        if layer_idx is not None and layer_indices is not None:
            raise ValueError("Pass only one of layer_idx or layer_indices")
        self.vectors_path = Path(vectors_path)
        self.alpha = alpha
        self._layer_override = layer_idx
        self._layer_indices = layer_indices
        self._pairs: list[tuple[int, torch.Tensor]] | None = None
        self._steer_target_resolved: str | None = None
        self._steer_scale_resolved: str | None = None
        self.steer_mode = steer_mode
        self.k_assist = k_assist
        self.steer_start_offset = steer_start_offset
        self._steer_target_override = steer_target
        self._steer_scale_override = steer_scale

    def _ensure_state(self) -> list[tuple[int, torch.Tensor]]:
        if self._pairs is None:
            state = torch.load(
                self.vectors_path, map_location="cpu", weights_only=False
            )
            v_bridge = state.get("v_bridge", {})
            if self._layer_indices is not None:
                if not self._layer_indices:
                    raise ValueError("layer_indices must be a non-empty list")
                pairs: list[tuple[int, torch.Tensor]] = []
                for L in self._layer_indices:
                    if L not in v_bridge:
                        available = sorted(v_bridge.keys()) if v_bridge else []
                        raise ValueError(
                            f"Layer {L} not in checkpoint v_bridge. "
                            f"Available layers: {available}"
                        )
                    pairs.append((L, v_bridge[L]))
                self._pairs = pairs
            elif self._layer_override is not None:
                if self._layer_override not in v_bridge:
                    available = sorted(v_bridge.keys()) if v_bridge else []
                    raise ValueError(
                        f"Layer {self._layer_override} not in checkpoint. "
                        f"Available layers: {available}"
                    )
                self._pairs = [(self._layer_override, v_bridge[self._layer_override])]
            else:
                self._pairs = [(int(state["steer_layer"]), state["v_steer"])]
            raw_tgt = (
                self._steer_target_override
                if self._steer_target_override is not None
                else state.get("steer_target", STEER_TARGET_RESIDUAL)
            )
            self._steer_target_resolved = validate_steer_target(str(raw_tgt))
            raw_scale = (
                self._steer_scale_override
                if self._steer_scale_override is not None
                else state.get("steer_scale", STEER_SCALE_FIXED)
            )
            self._steer_scale_resolved = validate_steer_scale(str(raw_scale))
        return self._pairs

    def generate(
        self,
        loader: Any,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        pairs = self._ensure_state()
        assert self._steer_target_resolved is not None
        assert self._steer_scale_resolved is not None
        if len(pairs) == 1:
            li, v = pairs[0]
            return steered_generate(
                loader,
                messages,
                layer_idx=li,
                v_steer=v,
                alpha=self.alpha,
                mode=self.steer_mode,
                k_assist=self.k_assist,
                steer_start_offset=self.steer_start_offset,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                steer_target=self._steer_target_resolved,
                steer_scale=self._steer_scale_resolved,
                **kwargs,
            )
        return steered_generate(
            loader,
            messages,
            layer_vector_pairs=pairs,
            alpha=self.alpha,
            mode=self.steer_mode,
            k_assist=self.k_assist,
            steer_start_offset=self.steer_start_offset,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            steer_target=self._steer_target_resolved,
            steer_scale=self._steer_scale_resolved,
            **kwargs,
        )

    def generate_batch(
        self,
        loader: Any,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        pairs = self._ensure_state()
        assert self._steer_target_resolved is not None
        assert self._steer_scale_resolved is not None
        if len(pairs) == 1:
            li, v = pairs[0]
            return steered_generate_batch(
                loader,
                batch_messages,
                layer_idx=li,
                v_steer=v,
                alpha=self.alpha,
                mode=self.steer_mode,
                k_assist=self.k_assist,
                steer_start_offset=self.steer_start_offset,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                steer_target=self._steer_target_resolved,
                steer_scale=self._steer_scale_resolved,
                **kwargs,
            )
        return steered_generate_batch(
            loader,
            batch_messages,
            layer_vector_pairs=pairs,
            alpha=self.alpha,
            mode=self.steer_mode,
            k_assist=self.k_assist,
            steer_start_offset=self.steer_start_offset,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            steer_target=self._steer_target_resolved,
            steer_scale=self._steer_scale_resolved,
            **kwargs,
        )


class TauSteeredMethod(SteeredMethod):
    """Steering with a FRESH direction sampled per generate call.

    Mirrors scripts/serve_steered_openai.py::sample_pairs exactly: one shared
    eps ~ N(0, tau^2 I_k) across layers (layer diffs are correlated within a
    contrast pair), applied to each layer's rank-k PCA basis from
    fit_steering_distribution.py, then renormalized to ||mu|| so the effective
    dose alpha*||v|| matches the fixed-vector setup. tau=0 reproduces the
    fixed vector exactly.

    Sampling granularity: one draw per ``generate``/``generate_batch`` CALL —
    a batch shares a draw (directions vary across batches/items, which is what
    per-item quality metrics need; there is no within-item pair here as in
    LiveIdeaBench --separate-passes). Use a small batch_size for many draws.
    """

    slug = "tau_steered"

    def __init__(
        self,
        distribution_path: str | Path,
        tau: float,
        rank: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.distribution_path = Path(distribution_path)
        self.tau = float(tau)
        self.rank = int(rank)
        self._dist: dict | None = None

    def _ensure_state(self) -> list[tuple[int, torch.Tensor]]:
        base = super()._ensure_state()
        if self._dist is None:
            dckpt = torch.load(
                self.distribution_path, map_location="cpu", weights_only=False
            )
            self._dist = dckpt["distribution"]
            missing = [L for L, _ in base if L not in self._dist]
            if missing:
                raise ValueError(
                    f"Layers {missing} not in distribution file "
                    f"{self.distribution_path} (has {sorted(self._dist)})"
                )
        if self.tau == 0:
            return base
        eps = torch.randn(self.rank) * self.tau
        pairs: list[tuple[int, torch.Tensor]] = []
        for L, _ in base:
            d = self._dist[L]
            k = min(self.rank, d["scales"].shape[0])
            v = d["mu"] + (eps[:k] * d["scales"][:k]) @ d["components"][:k]
            v = v * (d["mu"].norm() / v.norm())
            pairs.append((L, v))
        return pairs


# ── Forced-choice Yes/No log-prob probe ───────────────────────────────────────


def forced_choice_logprobs(
    loader: Any,
    messages: list[dict[str, str]],
    choices: tuple[str, str] = (" Yes", " No"),
) -> tuple[float, float]:
    """Log-probabilities of two single-token continuations after ``messages``.

    Renders ``messages`` through the loader's chat template (with the
    generation prompt appended), then in **one** forward pass takes
    ``log_softmax`` of the final-position logits and looks up the two
    candidate tokens. Returns ``(logprob_choice_0, logprob_choice_1)``.

    Used by the Phase-2 utility/originality probes in
    ``stages.aut_brainstorm_select``. The paper's prompts end with ``A:``
    so the natural continuation is `` Yes`` / `` No`` — note the **leading
    space**, which is part of the token in BPE-style tokenizers.

    Each ``choice`` must encode to exactly one token under the loader's
    tokenizer; otherwise a ``ValueError`` is raised so the caller knows the
    probe is not well-defined.
    """
    from bridge_steering.chat_template import apply_chat_template_safe

    model, tokenizer = loader.load()

    text = apply_chat_template_safe(
        tokenizer, messages, add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    choice_ids: list[int] = []
    for c in choices:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"forced_choice_logprobs: choice {c!r} encodes to {len(ids)} "
                f"tokens ({ids}) under {type(tokenizer).__name__}; expected 1. "
                "Try a different surface form (e.g. ' Yes' vs ' No' vs 'Yes')."
            )
        choice_ids.append(ids[0])

    with torch.no_grad():
        out = model(**inputs)
    final_logits = out.logits[0, -1]                # [vocab]
    logprobs = torch.log_softmax(final_logits, dim=-1)
    a = float(logprobs[choice_ids[0]].item())
    b = float(logprobs[choice_ids[1]].item())
    return a, b


def forced_choice_logprobs_batch(
    loader: Any,
    batch_messages: list[list[dict[str, str]]],
    choices: tuple[str, str] = (" Yes", " No"),
    batch_size: int = 32,
) -> list[tuple[float, float]]:
    """Batched ``forced_choice_logprobs`` — same contract per item, parallelized.

    Renders each message list through the chat template, tokenizes them as a
    right-padded batch, runs ONE forward pass per chunk of ``batch_size``,
    and gathers ``log_softmax`` at each sequence's last non-pad position.
    Returns one ``(logprob_choice_0, logprob_choice_1)`` tuple per input,
    in input order.

    Per item this is bit-equivalent to ``forced_choice_logprobs`` up to
    floating-point noise from batched matmul (typically ~1e-5).
    """
    from bridge_steering.chat_template import apply_chat_template_safe

    model, tokenizer = loader.load()

    # Validate choices tokenize to single tokens (same contract as serial)
    choice_ids: list[int] = []
    for c in choices:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"forced_choice_logprobs_batch: choice {c!r} encodes to "
                f"{len(ids)} tokens ({ids}) under {type(tokenizer).__name__}; "
                "expected 1."
            )
        choice_ids.append(ids[0])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Render all chat templates once
    texts = [
        apply_chat_template_safe(tokenizer, msgs, add_generation_prompt=True)
        for msgs in batch_messages
    ]

    results: list[tuple[float, float]] = []
    # Right-pad so the last non-pad position = attention_mask.sum() - 1.
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            inputs = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
            attn = inputs["attention_mask"]
            last_pos = attn.sum(dim=1) - 1                       # [batch]
            with torch.no_grad():
                out = model(**inputs)
            # out.logits: [batch, seq_len, vocab]
            batch_idx = torch.arange(out.logits.shape[0], device=out.logits.device)
            final_logits = out.logits[batch_idx, last_pos]       # [batch, vocab]
            logprobs = torch.log_softmax(final_logits, dim=-1)
            for i in range(len(chunk)):
                a = float(logprobs[i, choice_ids[0]].item())
                b = float(logprobs[i, choice_ids[1]].item())
                results.append((a, b))
    finally:
        tokenizer.padding_side = old_side

    return results


# ── Registry: name → factory ──────────────────────────────────────────────────


def available_methods() -> list[str]:
    """Return the list of method names that can be built via ``build_method``."""
    return ["baseline", "steered"]


def build_method(
    name: str,
    vectors_path: str | Path | None = None,
    alpha: float = 1.0,
    layer_idx: int | None = None,
    layer_indices: list[int] | None = None,
    steer_mode: str = "all_new_tokens",
    k_assist: int = 16,
    steer_start_offset: int = 0,
    steer_target: str | None = None,
    steer_scale: str | None = None,
) -> AUTInferenceMethod:
    """
    Build a method by name.

    * ``steered``: ``vectors_path`` required; ``alpha``, ``layer_idx`` or ``layer_indices``,
      ``steer_mode``, ``k_assist`` optional.
    """
    if name == "baseline":
        return BaselineMethod()
    if name == "steered":
        if not vectors_path:
            raise ValueError("--method steered requires --vectors")
        return SteeredMethod(
            vectors_path=vectors_path,
            alpha=alpha,
            layer_idx=layer_idx,
            layer_indices=layer_indices,
            steer_mode=steer_mode,
            k_assist=k_assist,
            steer_start_offset=steer_start_offset,
            steer_target=steer_target,
            steer_scale=steer_scale,
        )
    raise ValueError(
        f"Unknown method {name!r}. Available: {available_methods()}"
    )
