"""Abstract base class for model loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModelLoader(ABC):
    """
    Common interface shared by HFLoader, OpenAILoader, VLLMLoader.

    Subclasses must implement load(), generate(), and prompt().
    """

    @abstractmethod
    def load(self) -> Any:
        """Load model/client. May be a no-op for API-backed loaders."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """
        Generate a reply for a list of chat messages.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling (False = greedy).
            temperature: Sampling temperature.
            **kwargs: Backend-specific overrides.

        Returns:
            Generated text string.
        """

    @abstractmethod
    def prompt(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """
        Single user prompt → reply. Convenience wrapper around generate().

        Args:
            user_prompt: User message text.
            system_prompt: Optional system instruction.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling.
            temperature: Sampling temperature.
            **kwargs: Passed through to generate().

        Returns:
            Generated text string.
        """
