"""Backend-agnostic judge client construction.

Both the AUT and MacGyver judge stages call ``make_judge_client(backend)``
to get an async OpenAI-compatible client. vLLM, OpenAI-compat, and Azure
backends all share the same chat-completions protocol, so stage code is
identical from here on.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import (
    ApiJudgeBackend,
    AzureApiJudgeBackend,
    JudgeBackend,
    VllmJudgeBackend,
)


def make_judge_client(backend: JudgeBackend) -> Any:
    """Return an async OpenAI-compatible client configured for the chosen backend.

    * vLLM:  dummy api key, points at the local server (``AsyncOpenAI``).
    * API:   reads the key from the env var named in ``backend.api_key_env``
             (``AsyncOpenAI``).
    * Azure: reads key/endpoint/api-version from the env vars named on the
             backend; the ``model=`` field on call sites is the **deployment
             name**, not the underlying model id (``AsyncAzureOpenAI``).
    """
    if isinstance(backend, VllmJudgeBackend):
        from openai import AsyncOpenAI
        return AsyncOpenAI(base_url=backend.base_url, api_key="vllm")

    if isinstance(backend, ApiJudgeBackend):
        from openai import AsyncOpenAI
        key = os.environ.get(backend.api_key_env)
        if not key:
            raise RuntimeError(
                f"env var {backend.api_key_env!r} not set "
                "(validated at config-load, but recheck here for safety)"
            )
        kwargs: dict[str, Any] = {"base_url": backend.base_url, "api_key": key}
        if backend.organization:
            kwargs["organization"] = backend.organization
        if backend.project:
            kwargs["project"] = backend.project
        return AsyncOpenAI(**kwargs)

    if isinstance(backend, AzureApiJudgeBackend):
        from openai import AsyncAzureOpenAI
        try:
            return AsyncAzureOpenAI(
                api_key=os.environ[backend.api_key_env],
                azure_endpoint=os.environ[backend.endpoint_env],
                api_version=os.environ[backend.api_version_env],
            )
        except KeyError as exc:
            raise RuntimeError(
                f"env var {exc.args[0]!r} not set "
                "(validated at config-load, but recheck here for safety)"
            ) from exc

    raise TypeError(f"Unknown judge backend type: {type(backend).__name__}")


def judge_model_name(backend: JudgeBackend) -> str:
    """Return the model identifier the client should pass in ``model=``."""
    return backend.model


def completion_kwargs(
    backend: JudgeBackend, max_tokens: int, temperature: float, seed: int,
) -> dict[str, Any]:
    """Return per-backend completion kwargs for ``chat.completions.create``.

    Azure's newer chat models (gpt-5 family, o-series; api-version ≥
    2024-09) have two quirks the public-OpenAI / vLLM paths don't:

      * They reject ``max_tokens``; the parameter was renamed to
        ``max_completion_tokens``.
      * They reject any ``temperature`` other than the default (1.0).
        Passing 0.0 raises ``BadRequestError(code='unsupported_value')``.

    For those backends we send ``max_completion_tokens`` and omit
    ``temperature`` entirely (so the server applies its default). For
    every other backend we send the original names and values, since
    judge-time determinism (``temperature=0``) still matters there.

    ``seed`` is sent on every backend — it's the OpenAI/Azure best-effort
    determinism knob (server returns the same output for same prompt+seed
    *usually*; floating-point batching may still break ties differently).
    """
    if isinstance(backend, AzureApiJudgeBackend):
        # ``reasoning_effort`` is gpt-5/o-series specific (Azure passes it
        # through extra_body). "low" cuts hidden reasoning tokens ~5-10x,
        # which is what you want when the model is being used as a
        # structured-output judge (the rubric does the heavy lifting, the
        # model shouldn't burn the token budget on chain-of-thought).
        return {
            "max_completion_tokens": max_tokens,
            "seed": seed,
            "extra_body": {"reasoning_effort": "low"},
        }
    return {
        "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
    }
