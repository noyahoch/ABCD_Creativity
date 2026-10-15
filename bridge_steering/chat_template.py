"""Utilities for robust chat-template rendering across model families."""

from __future__ import annotations

from typing import Any


def merge_system_into_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold system messages into the first user message for templates without system-role support."""
    system_parts = [str(m.get("content", "")) for m in messages if m.get("role") == "system"]
    if not system_parts:
        return messages

    merged = [dict(m) for m in messages if m.get("role") != "system"]
    sys_text = "\n\n".join([s for s in system_parts if s.strip()])
    if not sys_text:
        return merged

    for m in merged:
        if m.get("role") == "user":
            m["content"] = f"{sys_text}\n\n{m.get('content', '')}"
            return merged

    return [{"role": "user", "content": sys_text}] + merged


def apply_chat_template_safe(tokenizer: Any, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    """Apply tokenizer chat template with fallback for templates that reject system role."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except Exception as e:
        if "System role not supported" not in str(e):
            raise
        return tokenizer.apply_chat_template(
            merge_system_into_user(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
