"""Code Engine — Code mode agent.

Code mode is the primary coding partner. It transforms natural language
requests (and optional architect plans) into clean, production-ready code,
creating and editing local files via FileTools.

Supports automatic failure recovery: after writing code it can run tests and,
on failure, delegate to the Debug agent for up to N retry rounds.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.router import LLMRouter, TaskType
from coding.tools.file_tools import FileTools
from coding.tools.audit_log import AuditLog

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an elite software engineer in Code mode.
Your task is to implement exactly what the user requests by producing clean,
production-ready code.

When creating or editing files you MUST respond with a structured JSON action
plan followed by the file contents. Use this exact format for each file operation:

<action>
{
  "op": "write_file",       // write_file | create_file | edit_file | delete_file | mkdir
  "path": "<relative or absolute path>",
  "content": "<full file content>",  // for write_file / create_file
  "old_str": "<exact string>",       // for edit_file only
  "new_str": "<replacement string>"  // for edit_file only
}
</action>

Rules:
- Use relative paths anchored to the workspace root when possible.
- Always include complete file content (never truncate with '...').
- After all <action> blocks, write a brief plain-English summary of what was done.
- If the request is unclear, ask for clarification before writing any files.
- Prefer editing existing files over rewriting them wholesale when the change is small.
"""

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


class CodeAgent:
    """File-writing code implementation agent."""

    name = "CodeAgent"
    mode = "code"

    def __init__(
        self,
        llm_router: Optional[LLMRouter] = None,
        audit: Optional[AuditLog] = None,
        workspace: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.llm = llm_router or LLMRouter()
        self.audit = audit or AuditLog()
        self.workspace = workspace
        self.session_id = session_id

    def _file_tools(self) -> FileTools:
        return FileTools(audit=self.audit, session_id=self.session_id, agent_name=self.name)

    async def run(
        self,
        task: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        plan: Optional[str] = None,
        auto_test: bool = False,
        test_command: Optional[str] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Implement *task*, writing files as needed.

        Args:
            task:         Natural-language description of what to implement.
            history:      Prior conversation turns.
            context:      Extra context dict (workspace, file_contents, etc.).
            plan:         Optional architect plan to follow.
            auto_test:    If True, run *test_command* after writing and retry on failure.
            test_command: Shell command to run tests (e.g. "pytest tests/").
            max_retries:  Max Debug-agent retry rounds on test failure.
        """
        workspace = (context or {}).get("workspace") or self.workspace
        ft = self._file_tools()

        for attempt in range(max_retries + 1):
            result = await self._implement(task, history=history, context=context, plan=plan, workspace=workspace)
            if not result["success"]:
                return result

            # Apply file operations
            ops_result = await self._apply_actions(result["actions"], ft, workspace)
            result["files_changed"] = ops_result["files_changed"]
            result["file_errors"] = ops_result["errors"]

            if not auto_test or not test_command:
                break

            # Run tests
            test_result = await self._run_tests(test_command, workspace)
            result["test_result"] = test_result
            if test_result["success"]:
                result["recovery_attempts"] = attempt
                break
            if attempt < max_retries:
                logger.info("CodeAgent: tests failed on attempt %d, invoking debug recovery…", attempt + 1)
                task = (
                    f"The following tests failed after the last implementation:\n\n"
                    f"```\n{test_result.get('output', '')}\n```\n\n"
                    f"Original task: {task}\n\n"
                    "Fix the code so all tests pass."
                )
            else:
                result["recovery_exhausted"] = True

        return result

    # ── internals ─────────────────────────────────────────────────────────

    async def _implement(
        self,
        task: str,
        *,
        history: Optional[List[Dict[str, str]]],
        context: Optional[Dict[str, Any]],
        plan: Optional[str],
        workspace: Optional[str],
    ) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if workspace:
            messages.append({"role": "system", "content": f"Workspace root: {workspace}"})

        if plan:
            messages.append({"role": "system", "content": f"## Architect Plan\n\n{plan}"})

        if context:
            ctx_parts = []
            if context.get("file_contents"):
                ctx_parts.append(f"## Relevant Files\n{context['file_contents']}")
            if context.get("extra_context"):
                ctx_parts.append(context["extra_context"])
            if ctx_parts:
                messages.append({"role": "system", "content": "\n\n".join(ctx_parts)})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": task})

        try:
            response = await self.llm.complete(messages, task_type=TaskType.CODING)
            actions = self._parse_actions(response)
            return {"success": True, "mode": self.mode, "response": response, "actions": actions}
        except Exception as exc:
            logger.exception("CodeAgent LLM error")
            return {"success": False, "mode": self.mode, "error": str(exc), "files_changed": [], "actions": []}

    def _parse_actions(self, response: str) -> List[Dict[str, Any]]:
        actions = []
        for match in _ACTION_RE.finditer(response):
            try:
                actions.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError as exc:
                logger.warning("Could not parse action JSON: %s", exc)
        return actions

    async def _apply_actions(
        self,
        actions: List[Dict[str, Any]],
        ft: FileTools,
        workspace: Optional[str],
    ) -> Dict[str, Any]:
        files_changed: List[str] = []
        errors: List[str] = []

        for action in actions:
            op = action.get("op", "")
            raw_path = action.get("path", "")

            # Resolve relative paths against workspace
            if workspace and raw_path and not Path(raw_path).is_absolute():
                path = str(Path(workspace) / raw_path)
            else:
                path = raw_path

            if op in ("write_file", "create_file"):
                content = action.get("content", "")
                fn = ft.write_file if op == "write_file" else ft.create_file
                r = await fn(path, content)
                if r["success"]:
                    files_changed.append(path)
                else:
                    errors.append(f"{op} '{path}': {r['error']}")
            elif op == "edit_file":
                r = await ft.edit_file(path, action.get("old_str", ""), action.get("new_str", ""))
                if r["success"]:
                    files_changed.append(path)
                else:
                    errors.append(f"edit_file '{path}': {r['error']}")
            elif op == "delete_file":
                r = await ft.delete_file(path)
                if r["success"]:
                    files_changed.append(path)
                else:
                    errors.append(f"delete_file '{path}': {r['error']}")
            elif op == "mkdir":
                r = await ft.mkdir(path)
                if not r["success"]:
                    errors.append(f"mkdir '{path}': {r['error']}")
            else:
                logger.warning("CodeAgent: unknown op '%s'", op)

        return {"files_changed": files_changed, "errors": errors}

    async def _run_tests(self, command: str, workspace: Optional[str]) -> Dict[str, Any]:
        """Run *command* in a subprocess, return success + output.

        Uses shell=False with shlex.split() to prevent command injection.
        """
        import shlex
        try:
            args = shlex.split(command)
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                cwd=workspace,
                timeout=120,
            )
            success = proc.returncode == 0
            return {
                "success": success,
                "returncode": proc.returncode,
                "output": (proc.stdout + proc.stderr)[:4000],
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "Test command timed out after 120 s."}
        except Exception as exc:
            return {"success": False, "output": str(exc)}
