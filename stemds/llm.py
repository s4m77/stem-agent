"""Minimal LLM client abstraction."""

from __future__ import annotations

import os
from typing import Protocol

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    load_dotenv = None  # type: ignore[assignment]


class LLMClient(Protocol):
    def generate_text(self, prompt: str, model: str, temperature: float = 0.2) -> str:
        ...


class OpenAIResponsesClient:
    def __init__(self, api_key: str | None = None) -> None:
        if load_dotenv is not None:
            load_dotenv()
        from openai import OpenAI

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIResponsesClient")
        self._client = OpenAI(api_key=resolved_api_key)

    def generate_text(self, prompt: str, model: str, temperature: float = 0.2) -> str:
        response = self._client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
        )
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        return str(response)

