"""HuggingFace model loader (Qwen, Llama, and general causal-LM models)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base import BaseModelLoader
from bridge_steering.chat_template import apply_chat_template_safe


class HFLoader(BaseModelLoader):
    """
    Load any HuggingFace CausalLM (Qwen2/Qwen2.5, Llama, Mistral, …) from the hub
    or a local checkpoint, with optional 8-bit quantisation.
    """

    def __init__(
        self,
        model_name_or_path: str | Path = "Qwen/Qwen2-7B-Instruct",
        torch_dtype: torch.dtype | str = "auto",
        device_map: str | dict[str, Any] = "auto",
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        load_in_8bit: bool = False,
        **model_kwargs: Any,
    ) -> None:
        """
        Args:
            model_name_or_path: HF model id or path to local directory.
            torch_dtype: Model dtype; "auto" lets HF pick. Ignored when load_in_8bit=True.
            device_map: Passed to from_pretrained; "auto" uses accelerate.
            trust_remote_code: Required for Qwen models.
            local_files_only: Skip hub lookups; model must already be cached.
            cache_dir: Override HF_HUB_CACHE / HF_HOME.
            load_in_8bit: BitsAndBytes 8-bit quantisation. Requires bitsandbytes.
            **model_kwargs: Extra kwargs forwarded to from_pretrained.
        """
        self.model_name_or_path = str(model_name_or_path)
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.load_in_8bit = load_in_8bit
        self.model_kwargs = model_kwargs

        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer (cached after first call)."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        tokenizer_kw: dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "use_fast": True,
        }
        if self.cache_dir is not None:
            tokenizer_kw["cache_dir"] = self.cache_dir

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            **tokenizer_kw,
        )

        # accelerate's "auto" device_map doesn't place on MPS — on Apple
        # Silicon it falls back to CPU. Resolve to an explicit device when
        # CUDA isn't available so MPS gets picked up.
        resolved_device_map: str | dict[str, Any] = self.device_map
        if self.device_map == "auto" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                resolved_device_map = "mps"

        if self.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kw: dict[str, Any] = {
                "quantization_config": bnb_config,
                "device_map": resolved_device_map,
                "low_cpu_mem_usage": True,
                "local_files_only": self.local_files_only,
                **self.model_kwargs,
            }
        else:
            dtype: torch.dtype | str = (
                self.torch_dtype
                if isinstance(self.torch_dtype, torch.dtype) or self.torch_dtype == "auto"
                else getattr(torch, str(self.torch_dtype), torch.float32)
            )
            model_kw = {
                "torch_dtype": dtype,
                "device_map": resolved_device_map,
                "local_files_only": self.local_files_only,
                **self.model_kwargs,
            }
        if self.cache_dir is not None:
            model_kw["cache_dir"] = self.cache_dir

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kw,
        )
        return self._model, self._tokenizer

    def unload(self) -> None:
        """Release the model's GPU memory; the next ``load()`` reloads it.

        Exists for the denial stages' judge time-multiplexing: on a 45 GiB
        L40S a 24B half-shard cannot coexist with the Qwen-32B-AWQ reviewer
        (18.1 GiB weights alone — KV starvation, confirmed 2026-07-18), so the
        generator steps aside for the reviewer round and reloads afterwards.
        Reload is cheap (~7 s from page cache). The tokenizer is kept; the
        global torch RNG is untouched, so generation streams are identical to
        an uninterrupted run.
        """
        if self._model is None:
            return
        import gc

        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def model(self) -> AutoModelForCausalLM:
        if self._model is None:
            self.load()
        assert self._model is not None
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self.load()
        assert self._tokenizer is not None
        return self._tokenizer

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **generate_kwargs: Any,
    ) -> str:
        """Generate a reply for chat messages.

        Sampling follows ``temperature``: ``temperature <= 0`` uses greedy decoding;
        ``temperature > 0`` uses sampling. The ``do_sample`` argument is ignored.
        """
        model, tokenizer = self.load()
        do_sample_eff = temperature > 0

        text = apply_chat_template_safe(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample_eff,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **generate_kwargs: Any,
    ) -> list[str]:
        """Generate replies for a batch of chat-message lists using left-padding."""
        do_sample_eff = temperature > 0
        if len(batch_messages) == 1:
            return [self.generate(
                batch_messages[0],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **generate_kwargs,
            )]

        model, tokenizer = self.load()

        texts = [
            apply_chat_template_safe(
                tokenizer, msgs, add_generation_prompt=True
            )
            for msgs in batch_messages
        ]

        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        try:
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=False,
            ).to(model.device)
        finally:
            tokenizer.padding_side = orig_side

        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample_eff,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )

        results: list[str] = []
        for seq in out:
            generated = seq[prompt_len:]
            results.append(tokenizer.decode(generated, skip_special_tokens=True))
        return results

    def prompt(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        **generate_kwargs: Any,
    ) -> str:
        """Single prompt → reply."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **generate_kwargs,
        )

    # ── Log-prob scoring (Brainstorm-then-Select) ─────────────────────────────

    def score_next_token(
        self,
        messages: list[dict[str, str]],
        target_tokens: list[str],
    ) -> dict[str, float]:
        """Single forward pass → log-probs for *target_tokens* at the next position.

        Useful for Yes/No evaluation prompts where we need log P(Yes) vs log P(No)
        to compute a confidence score (Summers-Stay et al., "Brainstorm then Select").
        """
        model, tokenizer = self.load()
        text = apply_chat_template_safe(
            tokenizer, messages, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        result: dict[str, float] = {}
        for tok in target_tokens:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            result[tok] = log_probs[ids[0]].item()
        return result

    def score_next_token_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        target_tokens: list[str],
    ) -> list[dict[str, float]]:
        """Batched forward pass → per-example log-probs for *target_tokens*."""
        if len(batch_messages) == 1:
            return [self.score_next_token(batch_messages[0], target_tokens)]

        model, tokenizer = self.load()
        texts = [
            apply_chat_template_safe(
                tokenizer, msgs, add_generation_prompt=True
            )
            for msgs in batch_messages
        ]

        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        try:
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=False,
            ).to(model.device)
        finally:
            tokenizer.padding_side = orig_side

        with torch.no_grad():
            all_logits = model(**inputs).logits  # (B, seq_len, vocab)

        # For left-padded inputs the last real token is always at position -1
        last_logits = all_logits[:, -1, :]  # (B, vocab)
        last_log_probs = torch.log_softmax(last_logits, dim=-1)

        tok_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]

        results: list[dict[str, float]] = []
        for b in range(last_log_probs.shape[0]):
            results.append({
                tok: last_log_probs[b, tid].item()
                for tok, tid in zip(target_tokens, tok_ids)
            })
        return results


# Backwards-compatible alias used by existing scripts and notebooks
QwenModelLoader = HFLoader
