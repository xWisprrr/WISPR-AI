"""
WISPR Coder Agent — multi-language code generation, debugging, optimisation.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from loguru import logger

from agents.base_agent import AgentResult, BaseAgent
from coding.engine import CodingEngine
from llm.router import TaskType, chat_async


class CoderAgent(BaseAgent):
    """Code generation, debugging, optimisation, and cross-language translation."""

    name = "coder"
    description = "Advanced coding agent: generate, debug, optimize, and translate code."

    def __init__(self) -> None:
        super().__init__()
        self._engine = CodingEngine()

    async def run(self, task: str, context: Optional[dict[str, Any]] = None) -> AgentResult:
        logger.info(f"[CoderAgent] task={task[:80]!r}")
        ctx = context or {}

        language = ctx.get("language") or self._engine.detect_language(task)
        action = ctx.get("action", "generate")  # generate | debug | optimize | translate

        system_prompt = (
            "You are WISPR Coder, an expert software engineer. "
            "Produce clean, well-commented, production-ready code. "
            "Always wrap code in markdown fenced code blocks with the language label."
        )

        user_prompt = self._engine.build_prompt(task, language=language, action=action)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await chat_async(messages=messages, task=TaskType.CODING)

        # Extract code blocks from the response
        code_blocks = self._engine.extract_code_blocks(response)

        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.name,
            success=True,
            output=response,
            metadata={
                "language": language,
                "action": action,
                "code_blocks": code_blocks,
            },
        )
