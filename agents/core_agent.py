"""WISPR Core Agent — general intelligence, conversation, and reasoning."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agents.base_agent import AgentResult, BaseAgent
from llm.router import TaskType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are WISPR Core — a highly capable general-purpose AI assistant.
You excel at reasoning, analysis, explanation, and conversational tasks.
Always structure your responses clearly with well-organized reasoning.
When uncertain, say so explicitly rather than hallucinating facts.
"""


class CoreAgent(BaseAgent):
    """General-purpose conversational and reasoning agent."""

    name = "CoreAgent"
    task_type = TaskType.REASONING

    async def run(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        ctx = context or {}
        history = self.memory.short_term.as_messages(n=8)

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task})

        if ctx.get("extra_context"):
            messages[-1]["content"] += f"\n\nAdditional context:\n{ctx['extra_context']}"

        try:
            response = await self.llm.complete(messages, task_type=self.task_type)
            self.memory.short_term.add("user", task)
            self.memory.short_term.add("assistant", response)
            return self._ok(response)
        except Exception as exc:
            return self._err(str(exc))
