"""WISPR ReAct Agent — Reasoning + Acting loop with tool calling."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentResult, BaseAgent
from llm.router import TaskType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are WISPR ReAct — an autonomous AI agent that solves tasks by interleaving
Reasoning and Acting in an iterative loop until you arrive at a Final Answer.

Available tools:
  search(query: str)           — Search the web for real-time information.
  python(code: str)            — Execute Python code and return stdout/stderr.
  memory(key: str)             — Look up a value stored in long-term memory.

Output format — you MUST follow this pattern exactly:

Thought: <your step-by-step reasoning about what to do next>
Action: <tool_name>(<argument>)
Observation: <you do NOT write this — it will be filled in automatically>

Repeat Thought/Action/Observation until you are ready to answer, then:

Thought: I now have enough information to answer.
Final Answer: <your complete, well-structured answer>

Rules:
- Each Action must call exactly ONE tool.
- Never fabricate Observations — only use real tool results.
- If a tool fails, reason about the failure and try a different approach.
- Be concise in Thoughts; be thorough in the Final Answer.
"""


class ReActAgent(BaseAgent):
    """Implements the ReAct (Reasoning + Acting) pattern.

    Iterates through Thought→Action→Observation cycles using real tools
    (search, code execution, memory lookup) before producing a Final Answer.
    """

    name = "ReActAgent"
    task_type = TaskType.REACT

    def __init__(self, max_iterations: int = 8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    async def run(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        ctx = context or {}
        scratchpad: List[Dict[str, str]] = []

        for iteration in range(self.max_iterations):
            # Build the full prompt with accumulated scratchpad
            messages = self._build_messages(task, scratchpad, ctx)

            try:
                llm_output = await self.llm.complete(
                    messages, task_type=self.task_type, temperature=0.0
                )
            except Exception as exc:
                return self._err(f"LLM call failed at iteration {iteration}: {exc}")

            logger.debug("[ReAct iter=%d] LLM output: %s", iteration, llm_output[:200])

            # Check for a Final Answer
            final = self._extract_final_answer(llm_output)
            if final is not None:
                return self._ok(
                    final,
                    iterations=iteration + 1,
                    scratchpad=scratchpad,
                )

            # Parse the next Thought + Action
            thought, action_str = self._parse_thought_action(llm_output)
            if action_str is None:
                # No recognisable action — treat the entire response as the answer
                return self._ok(
                    llm_output.strip(),
                    iterations=iteration + 1,
                    scratchpad=scratchpad,
                )

            # Execute the tool
            observation = await self._execute_action(action_str, ctx)
            logger.debug("[ReAct iter=%d] action=%s → obs=%s", iteration, action_str, observation[:120])

            scratchpad.append(
                {
                    "thought": thought,
                    "action": action_str,
                    "observation": observation,
                }
            )

        # Max iterations reached — return the last observation as best answer
        last_obs = scratchpad[-1]["observation"] if scratchpad else "No result."
        return self._ok(
            last_obs,
            iterations=self.max_iterations,
            scratchpad=scratchpad,
            warning="max_iterations_reached",
        )

    # ── prompt construction ───────────────────────────────────────────────

    def _build_messages(
        self,
        task: str,
        scratchpad: List[Dict[str, str]],
        ctx: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        user_parts = [f"Task: {task}"]
        if ctx.get("extra_context"):
            user_parts.append(f"Extra context: {ctx['extra_context']}")

        # Append the accumulated Thought/Action/Observation history
        for step in scratchpad:
            user_parts.append(f"Thought: {step['thought']}")
            user_parts.append(f"Action: {step['action']}")
            user_parts.append(f"Observation: {step['observation']}")

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

    # ── output parsing ────────────────────────────────────────────────────

    @staticmethod
    def _extract_final_answer(text: str) -> Optional[str]:
        match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_thought_action(text: str):
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)(?=\nObservation:|$)", text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else text.strip()
        action_str = action_match.group(1).strip() if action_match else None
        return thought, action_str

    # ── tool execution ────────────────────────────────────────────────────

    async def _execute_action(self, action_str: str, ctx: Dict[str, Any]) -> str:
        """Dispatch the parsed action to the appropriate tool."""
        tool_name, arg = self._parse_tool_call(action_str)
        if tool_name is None:
            return f"Could not parse tool call: {action_str!r}"

        try:
            if tool_name == "search":
                return await self._tool_search(arg)
            elif tool_name == "python":
                return await self._tool_python(arg)
            elif tool_name == "memory":
                return self._tool_memory(arg)
            else:
                return f"Unknown tool '{tool_name}'. Available: search, python, memory."
        except Exception as exc:
            return f"Tool '{tool_name}' raised an error: {exc}"

    @staticmethod
    def _parse_tool_call(action_str: str):
        """Parse 'tool_name(argument)' into (tool_name, argument)."""
        match = re.match(r"(\w+)\((.+)\)$", action_str.strip(), re.DOTALL)
        if not match:
            return None, None
        tool_name = match.group(1).lower()
        arg = match.group(2).strip().strip("\"'")
        return tool_name, arg

    async def _tool_search(self, query: str) -> str:
        from search.mega_search import MegaSearch
        try:
            results = await MegaSearch().search(query, max_results=5)
        except Exception as exc:
            return f"Search failed: {exc}"
        if not results:
            return "No results found."
        lines = []
        for r in results[:5]:
            lines.append(f"- [{r.get('source','?')}] {r.get('title','')}: {r.get('snippet','')}")
        return "\n".join(lines)

    async def _tool_python(self, code: str) -> str:
        from studio.ide import StudioIDE
        result = await StudioIDE().execute(code, language="python", timeout=15)
        if result["success"]:
            out = result.get("stdout", "").strip()
            return out if out else "(no output)"
        err = result.get("stderr", "").strip()
        return f"Error: {err}" if err else "Execution failed."

    def _tool_memory(self, key: str) -> str:
        value = self.memory.long_term.retrieve(key)
        if value is None:
            return f"No entry found for key '{key}'."
        return str(value)
