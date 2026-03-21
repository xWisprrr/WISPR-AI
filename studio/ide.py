"""
WISPR AI Studio — browser IDE simulation with project management and deployment.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from config import settings


class StudioIDE:
    """
    Simulates a WebContainer-style browser IDE.

    In production this would integrate with a real sandboxed execution
    environment (e.g. StackBlitz WebContainer, Pyodide, or a remote sandbox).
    """

    def __init__(self) -> None:
        self._projects: dict[str, dict[str, str]] = {}  # project_name -> {path: content}

    # ── File parsing ──────────────────────────────────────────────────────────

    def parse_project_files(self, llm_output: str) -> dict[str, str]:
        """
        Parse LLM output that uses the ### FILE: <path> convention.

        Expected format:
            ### FILE: src/main.py
            ```python
            print("hello")
            ```
        """
        files: dict[str, str] = {}
        # Match "### FILE: path\n```lang\ncontent\n```"
        pattern = r"###\s*FILE:\s*(.+?)\n```\w*\n(.*?)```"
        for match in re.finditer(pattern, llm_output, re.DOTALL):
            path = match.group(1).strip()
            content = match.group(2)
            files[path] = content

        if not files:
            # Fallback: treat entire output as a single main.py
            files["main.py"] = llm_output

        return files

    # ── Project management ────────────────────────────────────────────────────

    def create_project(self, name: str, files: dict[str, str]) -> None:
        """Register a project in the in-memory workspace."""
        self._projects[name] = files
        logger.info(f"[Studio] Project '{name}' created with {len(files)} file(s).")

    def get_project(self, name: str) -> Optional[dict[str, str]]:
        return self._projects.get(name)

    def list_projects(self) -> list[str]:
        return list(self._projects.keys())

    # ── Execution simulation ──────────────────────────────────────────────────

    def simulate_run(self, project_name: str) -> dict[str, Any]:
        """
        Simulate running the project.  In production, this would spin up
        a sandboxed environment and return actual stdout/stderr.
        """
        project = self._projects.get(project_name)
        if not project:
            return {"status": "error", "message": f"Project '{project_name}' not found."}

        file_list = list(project.keys())
        has_main = any("main" in f or "index" in f or "app" in f for f in file_list)

        return {
            "status": "success",
            "message": "Project executed successfully (simulated).",
            "files": file_list,
            "entry_point_detected": has_main,
            "note": "Connect a WebContainer runtime for real execution.",
        }

    # ── Deployment ────────────────────────────────────────────────────────────

    def deployment_instructions(self, target: str, project_name: str) -> str:
        if target not in settings.allowed_deployment_targets:
            return f"Deployment target '{target}' is not supported."

        instructions = {
            "github": (
                f"# Deploy '{project_name}' to GitHub\n"
                "1. `git init && git add . && git commit -m 'initial'`\n"
                "2. Create a new repo on github.com\n"
                "3. `git remote add origin <repo-url>`\n"
                "4. `git push -u origin main`"
            ),
            "vercel": (
                f"# Deploy '{project_name}' to Vercel\n"
                "1. Install Vercel CLI: `npm i -g vercel`\n"
                "2. Run: `vercel`\n"
                "3. Follow the prompts to link or create a project."
            ),
            "netlify": (
                f"# Deploy '{project_name}' to Netlify\n"
                "1. Install Netlify CLI: `npm i -g netlify-cli`\n"
                "2. Run: `netlify deploy --prod`\n"
                "3. Follow the prompts."
            ),
        }
        return instructions.get(target, "No instructions available.")
