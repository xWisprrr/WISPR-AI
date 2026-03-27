"""Multi-LLM Intelligence Layer — routes tasks to the most suitable model."""

from __future__ import annotations

import asyncio
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

import litellm
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


_TASK_MODEL_MAP: Dict[TaskType, str] = {
    TaskType.REASONING: settings.reasoning_model,
    TaskType.CODING: settings.coding_model,
    TaskType.SEARCH: settings.search_model,
    TaskType.GENERAL: settings.general_model,
}


class LLMRouter:
    """Routes LLM requests to the best available model for the given task type.

    Uses LiteLLM under the hood so any supported provider works transparently.
    Implements exponential-back-off retry + automatic fallback.
    """

    def __init__(self) -> None:
        self._configure_keys()

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
        chosen_model = model or _TASK_MODEL_MAP.get(task_type, settings.general_model)
        max_tokens = max_tokens or settings.llm_max_tokens
        temperature = temperature if temperature is not None else settings.llm_temperature

        try:
            return await self._call_llm(
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
            return await self._call_llm(
                settings.fallback_model, messages, max_tokens, temperature, **kwargs
            )

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        task_type: TaskType = TaskType.GENERAL,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
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

    # ── internals ─────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((litellm.exceptions.APIError, asyncio.TimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_llm(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=settings.llm_timeout,
            **kwargs,
        )
        return response.choices[0].message.content or ""

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
