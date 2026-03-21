"""
Base agent class that all WISPR agents inherit from.
"""
from __future__ import annotations

import abc
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


@dataclass
class AgentResult:
    agent_id: str
    agent_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    elapsed: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "elapsed": round(self.elapsed, 3),
            "metadata": self.metadata,
        }


class BaseAgent(abc.ABC):
    """Abstract base class for all WISPR agents."""

    name: str = "base"
    description: str = ""

    def __init__(self) -> None:
        self._id = str(uuid.uuid4())

    @property
    def agent_id(self) -> str:
        return self._id

    @abc.abstractmethod
    async def run(self, task: str, context: Optional[dict[str, Any]] = None) -> AgentResult:
        """Execute the agent on the given task and return a structured result."""

    async def _timed_run(self, task: str, context: Optional[dict[str, Any]] = None) -> AgentResult:
        """Wrapper that measures execution time and catches exceptions."""
        start = time.monotonic()
        try:
            result = await self.run(task, context)
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(f"[{self.name}] unhandled error: {exc}")
            return AgentResult(
                agent_id=self._id,
                agent_name=self.name,
                success=False,
                output=None,
                error=str(exc),
                elapsed=elapsed,
            )
        result.elapsed = time.monotonic() - start
        return result
