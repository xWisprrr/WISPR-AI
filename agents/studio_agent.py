"""
WISPR AI Studio Agent — browser IDE simulation with deployment capabilities.
"""
from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from agents.base_agent import AgentResult, BaseAgent
from llm.router import TaskType, chat_async
from studio.ide import StudioIDE


class StudioAgent(BaseAgent):
    """
    Studio IDE agent: writes, runs (simulated), and deploys code.
    Comparable to Replit + Cursor + Devin combined.
    """

    name = "studio"
    description = "Browser IDE agent: write, run, and deploy full applications."

    def __init__(self) -> None:
        super().__init__()
        self._ide = StudioIDE()

    async def run(self, task: str, context: Optional[dict[str, Any]] = None) -> AgentResult:
        logger.info(f"[StudioAgent] task={task[:80]!r}")
        ctx = context or {}

        deploy_target = ctx.get("deploy_target", "github")
        project_name = ctx.get("project_name", "wispr-project")

        # Step 1: Generate full project structure via LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "You are WISPR Studio, an expert full-stack developer. "
                    "Given a project description, generate a complete project structure "
                    "with all necessary files. Format each file as:\n\n"
                    "### FILE: <relative/path/to/file>\n```<lang>\n<content>\n```\n\n"
                    "Include a README.md, appropriate config files, and deployment instructions."
                ),
            },
            {"role": "user", "content": f"Create a project: {task}"},
        ]

        project_output = await chat_async(messages=messages, task=TaskType.CODING, max_tokens=4096)

        # Step 2: Parse files and set up the project in the studio
        files = self._ide.parse_project_files(project_output)
        self._ide.create_project(project_name, files)

        # Step 3: Simulate running the project
        run_result = self._ide.simulate_run(project_name)

        # Step 4: Generate deployment instructions
        deploy_instructions = self._ide.deployment_instructions(deploy_target, project_name)

        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.name,
            success=True,
            output=project_output,
            metadata={
                "project_name": project_name,
                "files_created": list(files.keys()),
                "run_result": run_result,
                "deploy_target": deploy_target,
                "deploy_instructions": deploy_instructions,
            },
        )
