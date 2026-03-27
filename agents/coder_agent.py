"""WISPR Coder Agent — code generation, debugging, and optimisation."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentResult, BaseAgent
from llm.router import TaskType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are WISPR Coder — an elite software engineer AI.
You write production-quality, well-commented code in any language requested.
When given a coding task you MUST:
1. Identify the programming language (default to Python if unspecified).
2. Write complete, runnable code inside a fenced code block.
3. Briefly explain the key design decisions after the code block.
4. If debugging, identify the root cause before proposing a fix.

Supported languages: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++,
Ruby, Bash, SQL, HTML/CSS, and more.
"""


class CoderAgent(BaseAgent):
    """Code generation, debugging, and multi-language translation agent."""

    name = "CoderAgent"
    task_type = TaskType.CODING

    async def run(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        ctx = context or {}
        language = ctx.get("language", "")
        existing_code = ctx.get("code", "")

        user_content = task
        if language:
            user_content += f"\n\nLanguage: {language}"
        if existing_code:
            user_content += f"\n\nExisting code:\n```\n{existing_code}\n```"

        try:
            response = await self._llm_complete(_SYSTEM_PROMPT, user_content)
            code_blocks = self._extract_code_blocks(response)
            return self._ok(
                response,
                code_blocks=code_blocks,
                language=language,
                num_blocks=len(code_blocks),
            )
        except Exception as exc:
            return self._err(str(exc))

    @staticmethod
    def _extract_code_blocks(text: str) -> List[str]:
        """Extract fenced code block contents from a markdown response."""
        pattern = r"```(?:\w+)?\n?(.*?)```"
        return re.findall(pattern, text, flags=re.DOTALL)
