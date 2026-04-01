"""Code Engine — Debug mode agent.

Debug mode is a specialized troubleshooting expert. It systematically
diagnoses issues (test failures, stack traces, runtime errors), identifies
root causes, and provides targeted solutions — including applying fixes
directly to local files when given access.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from llm.router import LLMRouter, TaskType
from coding.tools.file_tools import FileTools
from coding.tools.audit_log import AuditLog

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert software debugger in Debug mode.
Your job is to systematically diagnose bugs, failures, and runtime errors.

Approach:
1. IDENTIFY: State the exact nature of the failure (error type, line, traceback).
2. HYPOTHESIZE: List the most likely root causes (ranked by probability).
3. DIAGNOSE: Inspect relevant code and evidence to confirm the root cause.
4. FIX: Provide targeted, minimal fixes.

When you need to apply a file fix, emit one or more <action> blocks:

<action>
{
  "op": "write_file",       // write_file | edit_file | create_file
  "path": "<path>",
  "content": "<full new content>",  // for write_file/create_file
  "old_str": "<exact original>",    // for edit_file
  "new_str": "<replacement>"        // for edit_file
}
</action>

After fixing, explain what caused the bug and why the fix resolves it.
Keep changes minimal — fix only what is broken.
"""

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)

import json


class DebugAgent:
    """Diagnosis and targeted-fix agent."""

    name = "DebugAgent"
    mode = "debug"

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
        error_output: Optional[str] = None,
        code_snippet: Optional[str] = None,
        apply_fixes: bool = True,
    ) -> Dict[str, Any]:
        """Diagnose *task* and optionally apply fixes.

        Args:
            task:         Description of the problem or the failing test command.
            history:      Prior conversation turns.
            context:      Extra context (workspace, file_contents, etc.).
            error_output: Raw error/stack trace/test output.
            code_snippet: Relevant code to inspect.
            apply_fixes:  Whether to execute <action> blocks found in the response.
        """
        workspace = (context or {}).get("workspace") or self.workspace
        ft = self._file_tools()

        messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if workspace:
            messages.append({"role": "system", "content": f"Workspace root: {workspace}"})

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

        user_content = task
        if code_snippet:
            user_content += f"\n\n## Code\n```\n{code_snippet}\n```"
        if error_output:
            user_content += f"\n\n## Error / Stack Trace\n```\n{error_output}\n```"

        messages.append({"role": "user", "content": user_content})

        try:
            response = await self.llm.complete(messages, task_type=TaskType.CODING)
        except Exception as exc:
            logger.exception("DebugAgent LLM error")
            return {"success": False, "mode": self.mode, "error": str(exc), "files_changed": []}

        files_changed: List[str] = []
        file_errors: List[str] = []

        if apply_fixes:
            actions = self._parse_actions(response)
            for action in actions:
                op = action.get("op", "")
                from pathlib import Path
                raw_path = action.get("path", "")
                if workspace and raw_path and not Path(raw_path).is_absolute():
                    path = str(Path(workspace) / raw_path)
                else:
                    path = raw_path

                if op in ("write_file", "create_file"):
                    fn = ft.write_file if op == "write_file" else ft.create_file
                    r = await fn(path, action.get("content", ""))
                    if r["success"]:
                        files_changed.append(path)
                    else:
                        file_errors.append(f"{op} '{path}': {r['error']}")
                elif op == "edit_file":
                    r = await ft.edit_file(path, action.get("old_str", ""), action.get("new_str", ""))
                    if r["success"]:
                        files_changed.append(path)
                    else:
                        file_errors.append(f"edit_file '{path}': {r['error']}")

        return {
            "success": True,
            "mode": self.mode,
            "response": response,
            "files_changed": files_changed,
            "file_errors": file_errors,
        }

    def _parse_actions(self, response: str) -> List[Dict[str, Any]]:
        actions = []
        for match in _ACTION_RE.finditer(response):
            try:
                actions.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError as exc:
                logger.warning("DebugAgent: could not parse action: %s", exc)
        return actions
