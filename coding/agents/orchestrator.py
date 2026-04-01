"""Code Engine — Orchestrator mode agent.

Orchestrator mode intelligently breaks complex, multi-step projects into
manageable tasks, delegates work to the specialist agents (Ask / Architect /
Code / Debug), and coordinates their results into a cohesive deliverable.

Parallel execution is supported via asyncio.gather() with per-task isolation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from llm.router import LLMRouter, TaskType
from coding.tools.audit_log import AuditLog

logger = logging.getLogger(__name__)

_PLAN_SYSTEM = """\
You are an expert software engineering orchestrator.
Given a complex task, decompose it into a numbered list of sub-tasks.
Each sub-task must specify:
  - "id": integer
  - "mode": one of ask | architect | code | debug
  - "description": what this sub-task should accomplish
  - "depends_on": list of sub-task IDs that must complete first ([] if none)

Respond ONLY with valid JSON in this exact structure:
{
  "tasks": [
    {"id": 1, "mode": "architect", "description": "...", "depends_on": []},
    {"id": 2, "mode": "code",      "description": "...", "depends_on": [1]},
    ...
  ]
}
"""

_SYNTH_SYSTEM = """\
You are a senior engineering lead. Given the outputs from several specialist agents,
produce a unified final summary:
  1. What was accomplished (bullet points per task)
  2. Files created or modified
  3. Outstanding issues or next steps
Write in clear Markdown.
"""


class OrchestratorCodeAgent:
    """Decomposes complex requests and delegates to specialist agents."""

    name = "OrchestratorCodeAgent"
    mode = "orchestrator"

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

    async def run(
        self,
        task: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """Orchestrate *task* end-to-end.

        Args:
            task:     High-level project description.
            history:  Prior conversation turns.
            context:  Extra context dict (workspace, etc.).
            parallel: Run independent tasks in parallel when True.
        """
        workspace = (context or {}).get("workspace") or self.workspace

        # 1. Build task plan
        plan = await self._build_plan(task, history=history, context=context)
        if not plan["success"]:
            return {**plan, "mode": self.mode, "files_changed": []}

        tasks = plan["tasks"]
        logger.info("OrchestratorCodeAgent: executing %d sub-tasks", len(tasks))

        # 2. Execute tasks respecting dependencies
        completed: Dict[int, Dict[str, Any]] = {}
        all_files_changed: List[str] = []

        if parallel:
            completed, all_files_changed = await self._run_parallel(
                tasks, completed, workspace, context
            )
        else:
            completed, all_files_changed = await self._run_sequential(
                tasks, completed, workspace, context
            )

        # 3. Synthesise results
        summary = await self._synthesise(task, tasks, completed)

        return {
            "success": True,
            "mode": self.mode,
            "response": summary,
            "plan": tasks,
            "task_results": completed,
            "files_changed": all_files_changed,
        }

    # ── plan builder ──────────────────────────────────────────────────────

    async def _build_plan(
        self,
        task: str,
        *,
        history: Optional[List[Dict[str, str]]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": _PLAN_SYSTEM}]
        if context and context.get("workspace"):
            messages.append({"role": "system", "content": f"Workspace: {context['workspace']}"})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": f"Decompose this task:\n\n{task}"})

        try:
            raw = await self.llm.complete(messages, task_type=TaskType.REASONING)
            # Extract JSON (LLM sometimes wraps it in a code fence)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in plan response")
            plan_data = json.loads(match.group(0))
            tasks = plan_data.get("tasks", [])
            return {"success": True, "tasks": tasks}
        except Exception as exc:
            logger.exception("OrchestratorCodeAgent: plan building failed")
            return {"success": False, "error": str(exc)}

    # ── execution ─────────────────────────────────────────────────────────

    async def _run_parallel(
        self,
        tasks: List[Dict[str, Any]],
        completed: Dict[int, Dict[str, Any]],
        workspace: Optional[str],
        context: Optional[Dict[str, Any]],
    ):
        all_files: List[str] = []
        # Group tasks by dependency wave
        remaining = list(tasks)
        while remaining:
            # Tasks whose dependencies are all done
            ready = [t for t in remaining if all(dep in completed for dep in t.get("depends_on", []))]
            if not ready:
                # Dependency cycle or unsatisfiable — run all remaining sequentially
                logger.warning("OrchestratorCodeAgent: dependency stall, running remaining sequentially")
                ready = remaining

            results = await asyncio.gather(
                *[self._run_task(t, workspace, context, completed) for t in ready],
                return_exceptions=True,
            )
            for task_def, result in zip(ready, results):
                tid = task_def["id"]
                if isinstance(result, Exception):
                    completed[tid] = {"success": False, "error": str(result)}
                else:
                    completed[tid] = result
                    all_files.extend(result.get("files_changed", []))
                remaining.remove(task_def)

        return completed, all_files

    async def _run_sequential(
        self,
        tasks: List[Dict[str, Any]],
        completed: Dict[int, Dict[str, Any]],
        workspace: Optional[str],
        context: Optional[Dict[str, Any]],
    ):
        all_files: List[str] = []
        for task_def in tasks:
            result = await self._run_task(task_def, workspace, context, completed)
            completed[task_def["id"]] = result
            all_files.extend(result.get("files_changed", []))
        return completed, all_files

    async def _run_task(
        self,
        task_def: Dict[str, Any],
        workspace: Optional[str],
        context: Optional[Dict[str, Any]],
        completed: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Dispatch one sub-task to the appropriate specialist agent."""
        from coding.agents.ask import AskAgent
        from coding.agents.architect import ArchitectAgent
        from coding.agents.code import CodeAgent
        from coding.agents.debug import DebugAgent

        mode = task_def.get("mode", "ask")
        description = task_def.get("description", "")

        # Inject prior outputs as context
        prior_outputs = "\n\n".join(
            f"[Task {dep} output]\n{completed.get(dep, {}).get('response', '')}"
            for dep in task_def.get("depends_on", [])
            if dep in completed
        )
        ctx = dict(context or {})
        if workspace:
            ctx["workspace"] = workspace
        if prior_outputs:
            ctx["extra_context"] = prior_outputs

        try:
            if mode == "ask":
                agent = AskAgent(llm_router=self.llm)
                return await agent.run(description, context=ctx)
            elif mode == "architect":
                agent = ArchitectAgent(llm_router=self.llm)
                return await agent.run(description, context=ctx, workspace=workspace)
            elif mode == "code":
                agent = CodeAgent(
                    llm_router=self.llm,
                    audit=self.audit,
                    workspace=workspace,
                    session_id=self.session_id,
                )
                return await agent.run(description, context=ctx)
            elif mode == "debug":
                agent = DebugAgent(
                    llm_router=self.llm,
                    audit=self.audit,
                    workspace=workspace,
                    session_id=self.session_id,
                )
                return await agent.run(description, context=ctx)
            else:
                return {"success": False, "error": f"Unknown mode '{mode}'", "files_changed": []}
        except Exception as exc:
            logger.exception("OrchestratorCodeAgent task %s failed", task_def.get("id"))
            return {"success": False, "error": str(exc), "files_changed": []}

    # ── synthesis ─────────────────────────────────────────────────────────

    async def _synthesise(
        self,
        original_task: str,
        tasks: List[Dict[str, Any]],
        completed: Dict[int, Dict[str, Any]],
    ) -> str:
        task_summaries = []
        for t in tasks:
            tid = t["id"]
            res = completed.get(tid, {})
            status = "✅" if res.get("success") else "❌"
            output = res.get("response", res.get("error", "(no output)"))[:600]
            task_summaries.append(f"{status} Task {tid} [{t['mode']}]: {t['description']}\n{output}")

        messages = [
            {"role": "system", "content": _SYNTH_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original request: {original_task}\n\n"
                    + "\n\n---\n\n".join(task_summaries)
                ),
            },
        ]
        try:
            return await self.llm.complete(messages, task_type=TaskType.REASONING)
        except Exception as exc:
            logger.warning("OrchestratorCodeAgent synthesis failed: %s", exc)
            return "\n\n".join(task_summaries)
