"""Base agent — shared interface for all WISPR agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from llm.router import LLMRouter, TaskType
from memory.manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    success: bool
    output: Any
    agent_name: str
    task_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base for all WISPR agents.

    Subclasses must implement :meth:`run`.
    """

    name: str = "BaseAgent"
    task_type: TaskType = TaskType.GENERAL

    def __init__(
        self,
        llm_router: Optional[LLMRouter] = None,
        memory: Optional[MemoryManager] = None,
    ) -> None:
        self.llm = llm_router or LLMRouter()
        self.memory = memory or MemoryManager()

    @abstractmethod
    async def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the agent's primary task and return a structured result."""

    async def _llm_complete(self, system_prompt: str, user_prompt: str) -> str:
        """Convenience wrapper: single-turn LLM call."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.llm.complete(messages, task_type=self.task_type)

    def _ok(self, output: Any, **meta: Any) -> AgentResult:
        return AgentResult(
            success=True,
            output=output,
            agent_name=self.name,
            task_type=self.task_type.value,
            metadata=meta,
        )

    def _err(self, error: str, **meta: Any) -> AgentResult:
        logger.error("[%s] error: %s", self.name, error)
        return AgentResult(
            success=False,
            output=None,
            agent_name=self.name,
            task_type=self.task_type.value,
            error=error,
            metadata=meta,
        )
