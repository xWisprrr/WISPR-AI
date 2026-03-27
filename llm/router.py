"""Multi-LLM Intelligence Layer — routes tasks to the most suitable model."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Type

import litellm
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Silence LiteLLM's verbose output unless debug is on
litellm.set_verbose = settings.debug


class TaskType(str, Enum):
    REASONING = "reasoning"
    CODING = "coding"
    SEARCH = "search"
    GENERAL = "general"
    REACT = "react"


@dataclass
class LLMUsage:
    """Token usage statistics returned alongside LLM completions."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_response(cls, response: Any) -> "LLMUsage":
        usage = getattr(response, "usage", None)
        if usage is None:
            return cls()
        return cls(
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


_TASK_MODEL_MAP: Dict[TaskType, str] = {
    TaskType.REASONING: settings.reasoning_model,
    TaskType.CODING: settings.coding_model,
    TaskType.SEARCH: settings.search_model,
    TaskType.GENERAL: settings.general_model,
    TaskType.REACT: settings.reasoning_model,
}


class LLMRouter:
    """Routes LLM requests to the best available model for the given task type.

    Uses LiteLLM under the hood so any supported provider works transparently.
    Implements exponential-back-off retry + automatic fallback.
    """

    def __init__(self) -> None:
        self._configure_keys()
        # Accumulated token usage across the lifetime of this router instance
        self._total_usage = LLMUsage()

    # ── public API ────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: List[Dict[str, str]],
        task_type: TaskType = TaskType.GENERAL,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Return the assistant text for the given messages."""
        text, _ = await self.complete_with_usage(
            messages, task_type=task_type, model=model,
            max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        return text

    async def complete_with_usage(
        self,
        messages: List[Dict[str, str]],
        task_type: TaskType = TaskType.GENERAL,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[str, LLMUsage]:
        """Return (assistant_text, token_usage) for the given messages."""
        chosen_model = model or _TASK_MODEL_MAP.get(task_type, settings.general_model)
        max_tokens = max_tokens or settings.llm_max_tokens
        temperature = temperature if temperature is not None else settings.llm_temperature

        try:
            return await self._call_llm_with_usage(
                chosen_model, messages, max_tokens, temperature, **kwargs
            )
        except Exception as primary_err:
            logger.warning(
                "Primary model %s failed (%s). Falling back to %s.",
                chosen_model,
                primary_err,
                settings.fallback_model,
            )
            if chosen_model == settings.fallback_model:
                raise
            return await self._call_llm_with_usage(
                settings.fallback_model, messages, max_tokens, temperature, **kwargs
            )

    async def structured_complete(
        self,
        messages: List[Dict[str, str]],
        response_schema: Type[BaseModel],
        task_type: TaskType = TaskType.GENERAL,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return a structured JSON object validated against *response_schema*.

        Forces the model into JSON-output mode (response_format=json_object).
        Falls back to best-effort JSON parsing from plain text if the provider
        does not support the JSON response format natively.
        """
        chosen_model = model or _TASK_MODEL_MAP.get(task_type, settings.general_model)
        max_tokens = settings.llm_max_tokens
        temperature = 0.0  # deterministic for structured outputs

        schema_hint = (
            f"Respond ONLY with valid JSON matching this schema:\n"
            f"{json.dumps(response_schema.model_json_schema(), indent=2)}"
        )
        # Inject schema guidance into the system message
        augmented = list(messages)
        if augmented and augmented[0]["role"] == "system":
            augmented[0] = {
                "role": "system",
                "content": augmented[0]["content"] + "\n\n" + schema_hint,
            }
        else:
            augmented.insert(0, {"role": "system", "content": schema_hint})

        try:
            response = await litellm.acompletion(
                model=chosen_model,
                messages=augmented,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=settings.llm_timeout,
                **kwargs,
            )
        except Exception:
            # Provider may not support response_format — retry without it
            response = await litellm.acompletion(
                model=chosen_model,
                messages=augmented,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=settings.llm_timeout,
                **kwargs,
            )

        raw = response.choices[0].message.content or "{}"
        # Strip accidental markdown fences
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        return response_schema(**data).model_dump()

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        task_type: TaskType = TaskType.GENERAL,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield text chunks from a streaming LLM response."""
        chosen_model = model or _TASK_MODEL_MAP.get(task_type, settings.general_model)
        response = await litellm.acompletion(
            model=chosen_model,
            messages=messages,
            stream=True,
            timeout=settings.llm_timeout,
            **kwargs,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content

    def get_total_usage(self) -> Dict[str, int]:
        """Return accumulated token usage since this router was created."""
        return self._total_usage.to_dict()

    # ── internals ─────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((litellm.exceptions.APIError, asyncio.TimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_llm_with_usage(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> Tuple[str, LLMUsage]:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=settings.llm_timeout,
            **kwargs,
        )
        usage = LLMUsage.from_response(response)
        # Accumulate global usage stats
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens
        self._total_usage.total_tokens += usage.total_tokens
        return response.choices[0].message.content or "", usage

    @staticmethod
    def _configure_keys() -> None:
        if settings.openai_api_key:
            os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
        if settings.anthropic_api_key:
            os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
        if settings.gemini_api_key:
            os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
        if settings.groq_api_key:
            os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
