"""WISPR Studio Agent — browser-style IDE that writes, runs, and deploys code."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agents.base_agent import AgentResult, BaseAgent
from llm.router import TaskType
from studio.ide import StudioIDE

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are WISPR Studio — an expert full-stack developer and deployment engineer.
Given a project description or user request you:
1. Design the complete file structure.
2. Write all required code files (frontend + backend + config).
3. Provide step-by-step deployment instructions for GitHub, Vercel, and Netlify.
4. Highlight any environment variables or secrets required.
Always produce complete, production-ready code — no placeholders.
"""


class StudioAgent(BaseAgent):
    """Writes, executes, and deploys complete applications."""

    name = "StudioAgent"
    task_type = TaskType.CODING

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ide = StudioIDE()

    async def run(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        ctx = context or {}
        language = ctx.get("language", "python")
        run_code = ctx.get("run", False)
        deploy_target = ctx.get("deploy_target")

        try:
            response = await self._llm_complete(_SYSTEM_PROMPT, task)
        except Exception as exc:
            return self._err(f"LLM call failed: {exc}")

        result_meta: Dict[str, Any] = {"plan": response}

        if run_code and ctx.get("code"):
            run_result = await self._ide.execute(
                code=ctx["code"], language=language
            )
            result_meta["execution"] = run_result

        if deploy_target and ctx.get("project_path"):
            deploy_result = await self._ide.deploy(
                project_path=ctx["project_path"],
                target=deploy_target,
            )
            result_meta["deployment"] = deploy_result

        return self._ok(response, **result_meta)
