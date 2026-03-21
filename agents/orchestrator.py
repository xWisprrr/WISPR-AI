"""
Orchestrator Agent — controls all agents, runs them in parallel, combines outputs.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from loguru import logger

from agents.base_agent import AgentResult, BaseAgent
from agents.core_agent import CoreAgent
from agents.coder_agent import CoderAgent
from agents.search_agent import SearchAgent
from agents.studio_agent import StudioAgent
from hallucination.reducer import HallucinationReducer
from llm.router import TaskType, chat_async
from memory.store import ShortTermMemory, TaskMemory


class OrchestratorAgent(BaseAgent):
    """
    Top-level agent that:
    1. Decomposes the task into sub-tasks
    2. Routes sub-tasks to appropriate agents
    3. Runs agents in parallel where possible
    4. Combines and verifies outputs
    """

    name = "orchestrator"
    description = "Orchestrates all agents in parallel and synthesises results."

    def __init__(self) -> None:
        super().__init__()
        self._memory = ShortTermMemory()
        self._core = CoreAgent(memory=self._memory)
        self._coder = CoderAgent()
        self._search = SearchAgent()
        self._studio = StudioAgent()
        self._reducer = HallucinationReducer()

    # ── Task decomposition ────────────────────────────────────────────────────

    async def _decompose(self, task: str) -> dict[str, Any]:
        """Use an LLM to break the task into typed sub-tasks."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task decomposition expert. Given a user request, "
                    "output a JSON object with keys:\n"
                    '  "needs_search": bool,\n'
                    '  "needs_coding": bool,\n'
                    '  "needs_studio": bool,\n'
                    '  "sub_tasks": [{"agent": "core|coder|search|studio", "task": "..."}]\n'
                    "Output ONLY valid JSON, no prose."
                ),
            },
            {"role": "user", "content": task},
        ]
        import json, re

        raw = await chat_async(messages=messages, task=TaskType.FAST, max_tokens=512)
        # Extract JSON from the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Default: route to core agent only
        return {
            "needs_search": False,
            "needs_coding": False,
            "needs_studio": False,
            "sub_tasks": [{"agent": "core", "task": task}],
        }

    # ── Sub-task routing ──────────────────────────────────────────────────────

    async def _run_sub_task(
        self, agent_name: str, sub_task: str, context: Optional[dict[str, Any]] = None
    ) -> AgentResult:
        agent_map: dict[str, BaseAgent] = {
            "core": self._core,
            "coder": self._coder,
            "search": self._search,
            "studio": self._studio,
        }
        agent = agent_map.get(agent_name, self._core)
        return await agent._timed_run(sub_task, context)

    # ── Main orchestration ────────────────────────────────────────────────────

    async def run(self, task: str, context: Optional[dict[str, Any]] = None) -> AgentResult:
        logger.info(f"[Orchestrator] task={task[:80]!r}")
        task_mem = TaskMemory()

        # 1. Decompose
        decomposition = await self._decompose(task)
        task_mem.record_step("orchestrator", "decompose", decomposition)
        logger.debug(f"[Orchestrator] decomposition={decomposition}")

        sub_tasks = decomposition.get("sub_tasks", [{"agent": "core", "task": task}])

        # 2. Run sub-tasks in parallel
        coroutines = [
            self._run_sub_task(st["agent"], st.get("task", task))
            for st in sub_tasks
        ]
        results: list[AgentResult] = await asyncio.gather(*coroutines)

        for res in results:
            task_mem.record_step(res.agent_name, "run", res.output)

        # 3. Combine outputs
        combined = self._combine_results(task, results)

        # 4. Hallucination reduction / confidence check
        verified = await self._reducer.verify(task, combined, results)
        task_mem.record_step("orchestrator", "verify", verified)

        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.name,
            success=True,
            output=verified["answer"],
            metadata={
                "decomposition": decomposition,
                "agent_results": [r.to_dict() for r in results],
                "confidence": verified["confidence"],
                "task_summary": task_mem.summary(),
            },
        )

    def _combine_results(self, task: str, results: list[AgentResult]) -> str:
        """Naively concatenate successful agent outputs."""
        parts = []
        for r in results:
            if r.success and r.output:
                parts.append(f"[{r.agent_name.upper()}]\n{r.output}")
        return "\n\n".join(parts) if parts else "No output produced."
