"""Code Engine — Ask mode agent.

Ask mode is the intelligent Q&A companion. It provides expert technical
answers, explanations, and guidance. It MUST NOT modify any files.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llm.router import LLMRouter, TaskType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert software engineering assistant in Ask mode.
Your role is to answer technical questions, explain concepts, and provide guidance.

CRITICAL RULES:
- You MUST NOT create, edit, delete, or modify any files.
- You MUST NOT generate commands that write to the filesystem.
- You may show code examples inline as illustrations, but never instruct file writes.
- Be concise but thorough. Use markdown for clarity.
- Cite file paths and line numbers from the codebase when relevant.
- If a task requires file changes, tell the user to switch to Code mode.
"""


class AskAgent:
    """Q&A and guidance agent — read-only, no filesystem writes."""

    name = "AskAgent"
    mode = "ask"

    def __init__(self, llm_router: Optional[LLMRouter] = None) -> None:
        self.llm = llm_router or LLMRouter()

    async def run(
        self,
        task: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Answer *task* without touching the filesystem."""
        messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if history:
            messages.extend(history)

        if context:
            ctx_lines = []
            if context.get("workspace"):
                ctx_lines.append(f"Workspace root: {context['workspace']}")
            if context.get("file_contents"):
                ctx_lines.append(f"Relevant file contents:\n{context['file_contents']}")
            if ctx_lines:
                messages.append({"role": "system", "content": "\n".join(ctx_lines)})

        messages.append({"role": "user", "content": task})

        try:
            response = await self.llm.complete(messages, task_type=TaskType.CODING)
            return {"success": True, "mode": self.mode, "response": response, "files_changed": []}
        except Exception as exc:
            logger.exception("AskAgent error")
            return {"success": False, "mode": self.mode, "error": str(exc), "files_changed": []}
