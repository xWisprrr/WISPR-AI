"""WISPR Orchestrator — coordinates all agents and runs them in parallel."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentResult, BaseAgent
from agents.core_agent import CoreAgent
from agents.coder_agent import CoderAgent
from agents.react_agent import ReActAgent
from agents.search_agent import SearchAgent
from agents.studio_agent import StudioAgent
from llm.router import LLMRouter, TaskType
from memory.manager import MemoryManager
from reasoning.engine import ReasoningEngine

logger = logging.getLogger(__name__)

_PLAN_PROMPT = """\
You are the WISPR Orchestrator — a meta-AI that plans how to fulfil complex
user requests using specialised sub-agents.

Available agents:
- CoreAgent  : reasoning, analysis, conversation
- CoderAgent : writing, debugging, optimising code
- SearchAgent: real-time web search and research
- StudioAgent: building and deploying full applications
- ReActAgent : autonomous iterative Reason-Act-Observe loops for complex,
               multi-step tasks that require tool use (search + code execution)

Given the user task, output a JSON array of sub-tasks. Each element must have:
  {
    "agent": "<agent name>",
    "task": "<specific sub-task description>",
    "context": { <optional key-value pairs> }
  }

Output ONLY valid JSON — no markdown, no prose.
"""

_CLASSIFY_PROMPT = """\
Classify the following user request as SIMPLE or COMPLEX.

SIMPLE: a single direct question or conversational exchange answerable in one or two sentences.
Examples: greetings, factual lookups, quick definitions, basic calculations, yes/no questions.

COMPLEX: tasks that need multiple steps, in-depth research, code generation, deployment,
or detailed multi-part reasoning.
Examples: build an app, research and compare topics, debug a program, write a comprehensive plan.

Output only one word: SIMPLE or COMPLEX.
"""


class OrchestratorAgent:
    """Routes work to the right agent(s) and runs them concurrently."""

    name = "OrchestratorAgent"

    def __init__(
        self,
        llm_router: Optional[LLMRouter] = None,
        memory: Optional[MemoryManager] = None,
    ) -> None:
        self.llm = llm_router or LLMRouter()
        self.memory = memory or MemoryManager()
        self.reasoning = ReasoningEngine(llm_router=self.llm, memory=self.memory)
        self._agents: Dict[str, BaseAgent] = {
            "CoreAgent": CoreAgent(self.llm, self.memory),
            "CoderAgent": CoderAgent(self.llm, self.memory),
            "SearchAgent": SearchAgent(llm_router=self.llm, memory=self.memory),
            "StudioAgent": StudioAgent(llm_router=self.llm, memory=self.memory),
            "ReActAgent": ReActAgent(llm_router=self.llm, memory=self.memory),
        }

    async def run(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Classify *task* complexity and either answer directly (simple) or
        decompose into sub-tasks and run agents in parallel (complex)."""
        task_id = self.memory.tasks.new_task(task)
        logger.info("Orchestrator starting task %s: %s", task_id, task[:80])

        # ── Classify complexity ────────────────────────────────────────────
        complexity = await self._classify_complexity(task)
        logger.info("Task classified as: %s", complexity)

        if complexity == "simple":
            return await self._handle_simple(task, task_id, context or {})

        # ── Plan ──────────────────────────────────────────────────────────
        plan = await self._plan(task, context or {})
        self.memory.tasks.add_step(task_id, self.name, "plan", plan)

        if not plan:
            # Fall back to CoreAgent for unstructured tasks
            plan = [{"agent": "CoreAgent", "task": task, "context": {}}]

        # ── Execute in parallel ────────────────────────────────────────────
        results = await self._execute_parallel(plan, task_id)

        # ── Combine ───────────────────────────────────────────────────────
        combined = await self._combine(task, results)
        self.memory.tasks.complete_task(task_id, combined)

        return {
            "task_id": task_id,
            "task": task,
            "plan": plan,
            "agent_results": results,
            "final_answer": combined,
        }

    # ── internals ─────────────────────────────────────────────────────────

    async def _classify_complexity(self, task: str) -> str:
        """Return 'simple' or 'complex' for the given task."""
        messages = [
            {"role": "system", "content": _CLASSIFY_PROMPT},
            {"role": "user", "content": f"Request: {task}"},
        ]
        try:
            raw = await self.llm.complete(messages, task_type=TaskType.GENERAL)
            if "SIMPLE" in raw.strip().upper():
                return "simple"
            return "complex"
        except Exception as exc:
            logger.warning(
                "Complexity classification failed (%s); defaulting to complex.", exc
            )
            return "complex"

    async def _handle_simple(
        self, task: str, task_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Answer a simple task directly via CoreAgent using concise response style."""
        agent = self._agents["CoreAgent"]
        concise_context = {**context, "response_style": "concise"}
        result: AgentResult = await agent.run(task, concise_context)
        output = (
            str(result.output)
            if result.success
            else "I couldn't answer that. Please try again."
        )
        self.memory.tasks.add_step(
            task_id, agent.name, task, output, success=result.success
        )
        self.memory.tasks.complete_task(task_id, output)
        return {
            "task_id": task_id,
            "task": task,
            "plan": [],
            "agent_results": [
                {
                    "agent": agent.name,
                    "task": task,
                    "success": result.success,
                    "output": result.output,
                    "metadata": result.metadata,
                    "error": result.error,
                }
            ],
            "final_answer": output,
        }

    async def _plan(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": _PLAN_PROMPT},
            {"role": "user", "content": f"Task: {task}\nContext: {context}"},
        ]
        try:
            raw = await self.llm.complete(messages, task_type=TaskType.REASONING)
            # Strip markdown code fences (```json ... ``` or ``` ... ```)
            raw = raw.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Plan generation failed (%s); using single CoreAgent.", exc)
            return []

    async def _execute_parallel(
        self, plan: List[Dict[str, Any]], task_id: str
    ) -> List[Dict[str, Any]]:
        coros = []
        for step in plan:
            agent_name = step.get("agent", "CoreAgent")
            agent = self._agents.get(agent_name, self._agents["CoreAgent"])
            coros.append(
                self._run_step(agent, step["task"], step.get("context", {}), task_id)
            )
        results = await asyncio.gather(*coros, return_exceptions=True)
        output = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                output.append({"agent": plan[i]["agent"], "error": str(res)})
            else:
                output.append(res)
        return output

    async def _run_step(
        self,
        agent: BaseAgent,
        task: str,
        context: Dict[str, Any],
        task_id: str,
    ) -> Dict[str, Any]:
        result: AgentResult = await agent.run(task, context)
        self.memory.tasks.add_step(
            task_id,
            agent.name,
            task,
            result.output,
            success=result.success,
        )
        return {
            "agent": agent.name,
            "task": task,
            "success": result.success,
            "output": result.output,
            "metadata": result.metadata,
            "error": result.error,
        }

    async def _combine(self, original_task: str, results: List[Dict[str, Any]]) -> str:
        successful = [r for r in results if r.get("success") and r.get("output")]
        if not successful:
            return "All agents failed to produce results. Please try again."

        if len(successful) == 1:
            return str(successful[0]["output"])

        outputs_text = "\n\n".join(
            f"--- {r['agent']} ---\n{r['output']}" for r in successful
        )
        combine_prompt = (
            f"Original request: {original_task}\n\n"
            f"Outputs from multiple specialised agents:\n{outputs_text}\n\n"
            "Synthesise a single, coherent, high-quality final answer."
        )
        messages = [
            {
                "role": "system",
                "content": "You are a master synthesiser. Combine agent outputs into the best possible answer.",
            },
            {"role": "user", "content": combine_prompt},
        ]
        try:
            return await self.llm.complete(messages, task_type=TaskType.REASONING)
        except Exception as exc:
            logger.warning("Combine step failed: %s", exc)
            return str(successful[0]["output"])
