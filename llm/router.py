"""
LLM Router — selects the best model per task type using LiteLLM.

Task types:
  - reasoning  → highest-quality model
  - coding     → best coding model
  - search     → fast/cheap model for result summarisation
  - fast       → fastest/cheapest for simple tasks
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional

from loguru import logger

from config import settings

try:
    import litellm
    from litellm import acompletion, completion

    litellm.drop_params = True          # ignore unknown provider params
    LITELLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    LITELLM_AVAILABLE = False
    logger.warning("litellm not installed — LLM calls will be simulated")


class TaskType(str, Enum):
    REASONING = "reasoning"
    CODING = "coding"
    SEARCH = "search"
    FAST = "fast"
    GENERAL = "general"


# Map task types → model setting
_TASK_MODEL_MAP: dict[TaskType, str] = {
    TaskType.REASONING: settings.reasoning_model,
    TaskType.CODING: settings.coding_model,
    TaskType.SEARCH: settings.search_model,
    TaskType.FAST: settings.fast_model,
    TaskType.GENERAL: settings.fast_model,
}


def _build_env() -> None:
    """Push API keys from settings into environment so LiteLLM can pick them up."""
    key_map = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "GROQ_API_KEY": settings.groq_api_key,
        "TOGETHERAI_API_KEY": settings.together_api_key,
        "COHERE_API_KEY": settings.cohere_api_key,
    }
    for env_var, value in key_map.items():
        if value and not os.environ.get(env_var):
            os.environ[env_var] = value


_build_env()


# ── Public API ────────────────────────────────────────────────────────────────

def select_model(task: TaskType) -> str:
    """Return the model name configured for the given task type."""
    return _TASK_MODEL_MAP.get(task, settings.fast_model)


async def chat_async(
    messages: list[dict[str, str]],
    task: TaskType = TaskType.GENERAL,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **kwargs: Any,
) -> str:
    """Async LLM call.  Returns the assistant content string."""
    chosen_model = model or select_model(task)
    logger.debug(f"[LLM] task={task.value} model={chosen_model}")

    if not LITELLM_AVAILABLE:
        return f"[SIMULATED RESPONSE — litellm not installed] model={chosen_model}"

    try:
        response = await acompletion(
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error(f"[LLM] Error with {chosen_model}: {exc}")
        # Fallback to fast model
        if chosen_model != settings.fast_model:
            logger.info(f"[LLM] Falling back to {settings.fast_model}")
            return await chat_async(
                messages=messages,
                task=TaskType.FAST,
                model=settings.fast_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raise


def chat_sync(
    messages: list[dict[str, str]],
    task: TaskType = TaskType.GENERAL,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **kwargs: Any,
) -> str:
    """Synchronous LLM call (convenience wrapper)."""
    chosen_model = model or select_model(task)
    logger.debug(f"[LLM-sync] task={task.value} model={chosen_model}")

    if not LITELLM_AVAILABLE:
        return f"[SIMULATED RESPONSE — litellm not installed] model={chosen_model}"

    try:
        response = completion(
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error(f"[LLM-sync] Error with {chosen_model}: {exc}")
        raise
