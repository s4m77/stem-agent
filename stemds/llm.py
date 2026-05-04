"""Minimal LLM client abstractions."""

from __future__ import annotations

import os
import warnings
from typing import Protocol

from stemds.config import PROJECT_ROOT

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    load_dotenv = None  # type: ignore[assignment]


class BaseLLMClient(Protocol):
    def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        ...


class LLMClientError(RuntimeError):
    """Raised when an LLM provider request fails."""


class OpenAIClient:
    def __init__(self, api_key: str | None = None) -> None:
        if load_dotenv is not None:
            load_dotenv(PROJECT_ROOT / ".env")
        from openai import OpenAI

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required when using --agent openai")
        self._client = OpenAI(api_key=resolved_api_key)
        self.last_api_path: str | None = None
        self.last_seed_ignored: bool = False
        self._warned_seed_ignored_paths: set[str] = set()

    def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        responses_error: Exception | None = None
        self.last_api_path = None
        self.last_seed_ignored = False
        try:
            response = self._create_response(model=model, prompt=prompt, temperature=temperature, seed=seed)
            self.last_api_path = "responses"
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str):
                return output_text
            return str(response)
        except Exception as exc:
            responses_error = exc

        try:
            completion = self._create_chat_completion(model=model, prompt=prompt, temperature=temperature, seed=seed)
            self.last_api_path = "chat_completions"
            content = completion.choices[0].message.content
            if content is None:
                raise LLMClientError("OpenAI chat completion returned no content")
            return content
        except Exception as exc:
            raise LLMClientError(
                f"OpenAI request failed. Responses API error: {responses_error}. "
                f"Chat Completions fallback error: {exc}."
            ) from exc

    def _create_response(self, model: str, prompt: str, temperature: float, seed: int | None):
        kwargs = {"model": model, "input": prompt, "temperature": temperature}
        if seed is None:
            return self._client.responses.create(**kwargs)
        try:
            return self._client.responses.create(**kwargs, seed=seed)
        except TypeError:
            self.last_seed_ignored = True
            self._warn_seed_ignored_once("responses")
            return self._client.responses.create(**kwargs)

    def _create_chat_completion(self, model: str, prompt: str, temperature: float, seed: int | None):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if seed is None:
            return self._client.chat.completions.create(**kwargs)
        try:
            return self._client.chat.completions.create(**kwargs, seed=seed)
        except TypeError:
            self.last_seed_ignored = True
            self._warn_seed_ignored_once("chat_completions")
            return self._client.chat.completions.create(**kwargs)

    def _warn_seed_ignored_once(self, api_path: str) -> None:
        if api_path in self._warned_seed_ignored_paths:
            return
        self._warned_seed_ignored_paths.add(api_path)
        warnings.warn(f"OpenAI {api_path} client did not accept seed; retrying without seed.", RuntimeWarning)


class MockLLMClient:
    def __init__(self, responses: list[str] | str) -> None:
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        self.prompts: list[str] = []

    def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("MockLLMClient has no responses left")
        return self._responses.pop(0)


LLMClient = BaseLLMClient
OpenAIResponsesClient = OpenAIClient
