"""
WISPR Core Agent — general intelligence, conversation, and reasoning tasks.
"""
from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from agents.base_agent import AgentResult, BaseAgent
from llm.router import TaskType, chat_async
from memory.store import ShortTermMemory


class CoreAgent(BaseAgent):
    """General-purpose conversational and reasoning agent."""

    name = "core"
    description = "General intelligence: conversation, Q&A, reasoning tasks."

    def __init__(self, memory: Optional[ShortTermMemory] = None) -> None:
        super().__init__()
        self._memory = memory or ShortTermMemory()

    async def run(self, task: str, context: Optional[dict[str, Any]] = None) -> AgentResult:
        logger.info(f"[CoreAgent] task={task[:80]!r}")

        system_prompt = (
            "You are WISPR Core, a highly capable AI assistant. "
            "You reason carefully, provide accurate and concise answers, "
            "and acknowledge uncertainty when you are unsure."
        )

        # Build messages from history + current task
        history = self._memory.as_messages(limit=10)
        messages = [{"role": "system", "content": system_prompt}] + history + [
            {"role": "user", "content": task}
        ]

        response = await chat_async(messages=messages, task=TaskType.REASONING)

        # Persist to short-term memory
        self._memory.add("user", task)
        self._memory.add("assistant", response)

        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.name,
            success=True,
            output=response,
        )
