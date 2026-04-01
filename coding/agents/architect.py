"""Code Engine — Architect mode agent.

Architect mode thoroughly analyses the codebase and produces robust system
architectures and detailed, step-by-step implementation plans.
It does NOT write code or touch files — it produces plans only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.router import LLMRouter, TaskType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a world-class software architect in Architect mode.
Your role is to analyse codebases and produce thorough architectural designs
and detailed step-by-step implementation plans.

Guidelines:
- First describe the current state of the codebase (modules, patterns, dependencies).
- Then propose the target architecture with clear component boundaries.
- Break the implementation into numbered steps, each with: goal, files to touch,
  key decisions, and potential risks.
- Identify reuse opportunities and warn about breaking changes.
- Output in clean Markdown with headers, bullet points, and code fences where helpful.
- Do NOT write implementation code. Produce plans, not code.
"""

_INDEX_SYSTEM = """\
You are a codebase indexer. Given a file tree and file contents, produce a concise
structural summary: modules, classes, functions, patterns, and dependencies.
Be precise and terse. Output in Markdown.
"""


class ArchitectAgent:
    """Codebase analysis + architecture + planning agent."""

    name = "ArchitectAgent"
    mode = "architect"

    def __init__(self, llm_router: Optional[LLMRouter] = None) -> None:
        self.llm = llm_router or LLMRouter()

    async def run(
        self,
        task: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        workspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyse the codebase and produce an architectural plan for *task*."""
        # 1. Build codebase index if workspace is available
        codebase_summary = ""
        if workspace:
            codebase_summary = await self._index_workspace(workspace)

        # 2. Compose messages
        messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if codebase_summary:
            messages.append({
                "role": "system",
                "content": f"## Codebase Index\n\n{codebase_summary}",
            })

        if history:
            messages.extend(history)

        if context:
            ctx_parts = []
            if context.get("extra_context"):
                ctx_parts.append(context["extra_context"])
            if ctx_parts:
                messages.append({"role": "system", "content": "\n".join(ctx_parts)})

        messages.append({"role": "user", "content": f"## Architecture Request\n\n{task}"})

        try:
            response = await self.llm.complete(messages, task_type=TaskType.REASONING)
            return {
                "success": True,
                "mode": self.mode,
                "response": response,
                "codebase_summary": codebase_summary,
                "files_changed": [],
            }
        except Exception as exc:
            logger.exception("ArchitectAgent error")
            return {"success": False, "mode": self.mode, "error": str(exc), "files_changed": []}

    async def _index_workspace(self, workspace: str, max_files: int = 60) -> str:
        """Build a lightweight text index of *workspace* to feed to the LLM."""
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            return f"(workspace '{workspace}' not found)"

        tree_lines: List[str] = []
        file_samples: List[str] = []
        count = 0

        _skip = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".mypy_cache"}

        for path in sorted(root.rglob("*")):
            if any(part in _skip for part in path.parts):
                continue
            rel = path.relative_to(root)
            indent = "  " * (len(rel.parts) - 1)
            if path.is_dir():
                tree_lines.append(f"{indent}{rel.name}/")
            else:
                tree_lines.append(f"{indent}{rel.name}")
                if count < max_files and path.suffix in {
                    ".py", ".js", ".ts", ".go", ".rs", ".java", ".rb",
                    ".md", ".yaml", ".yml", ".toml", ".json",
                }:
                    try:
                        snippet = path.read_text(encoding="utf-8", errors="replace")[:800]
                        file_samples.append(f"### {rel}\n```\n{snippet}\n```")
                        count += 1
                    except OSError:
                        pass

        tree_text = "\n".join(tree_lines[:300])
        samples_text = "\n\n".join(file_samples[:20])

        # Ask LLM to summarise
        index_messages = [
            {"role": "system", "content": _INDEX_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"File tree:\n```\n{tree_text}\n```\n\n"
                    f"File samples:\n{samples_text}"
                ),
            },
        ]
        try:
            return await self.llm.complete(index_messages, task_type=TaskType.REASONING)
        except Exception:
            return tree_text
