"""Autonomous Reasoning Engine — multi-step reasoning with self-reflection."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from llm.router import LLMRouter, TaskType
from memory.manager import MemoryManager
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_DECOMPOSE_PROMPT = """\
You are a strategic planner. Break the following task into clear, ordered steps.
Output a numbered list of steps only — no preamble, no explanations.
"""

_REFLECT_PROMPT = """\
You are a critical reviewer. Evaluate the following reasoning output:
- Is the answer complete?
- Are there logical gaps or errors?
- What is your confidence score (0.0–1.0)?

Respond in this exact format:
COMPLETE: yes/no
ISSUES: <list any issues or "none">
CONFIDENCE: <float 0.0-1.0>
IMPROVED_ANSWER: <improved answer or "same">
"""


class ReasoningStep:
    def __init__(self, step_num: int, description: str) -> None:
        self.step_num = step_num
        self.description = description
        self.result: Optional[str] = None
        self.confidence: float = 0.0


class ReasoningEngine:
    """Implements multi-step reasoning with self-reflection and retry logic."""

    def __init__(
        self,
        llm_router: Optional[LLMRouter] = None,
        memory: Optional[MemoryManager] = None,
    ) -> None:
        self.llm = llm_router or LLMRouter()
        self.memory = memory or MemoryManager()

    async def reason(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute multi-step reasoning on *task* and return structured result."""
        max_steps = max_steps or settings.reasoning_max_steps
        ctx_text = str(context) if context else ""

        # Step 1: Decompose
        steps = await self._decompose(task, ctx_text)
        logger.info("Reasoning decomposed into %d steps", len(steps))

        # Step 2: Execute each step
        accumulated = ""
        executed_steps: List[Dict[str, Any]] = []
        for step in steps[:max_steps]:
            step_result = await self._execute_step(task, step.description, accumulated)
            step.result = step_result
            accumulated += f"\nStep {step.step_num}: {step.description}\nResult: {step_result}\n"
            executed_steps.append(
                {
                    "step": step.step_num,
                    "description": step.description,
                    "result": step_result,
                }
            )

        # Step 3: Self-reflect and refine
        final_answer, confidence, issues = await self._reflect(task, accumulated)

        # Step 4: Retry if low confidence
        retry_count = 0
        while (
            confidence < settings.reasoning_confidence_threshold
            and retry_count < settings.agent_max_retries
        ):
            retry_count += 1
            logger.info("Low confidence (%.2f), retrying reasoning (%d)…", confidence, retry_count)
            improved = await self._improve(task, final_answer, issues, accumulated)
            _, new_confidence, _ = await self._reflect(task, improved)
            if new_confidence > confidence:
                final_answer = improved
                confidence = new_confidence

        return {
            "task": task,
            "steps": executed_steps,
            "final_answer": final_answer,
            "confidence": confidence,
            "retries": retry_count,
        }

    # ── internals ─────────────────────────────────────────────────────────

    async def _decompose(self, task: str, context: str) -> List[ReasoningStep]:
        user_prompt = f"Task: {task}"
        if context:
            user_prompt += f"\nContext: {context}"
        messages = [
            {"role": "system", "content": _DECOMPOSE_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw = await self.llm.complete(messages, task_type=TaskType.REASONING)
        except Exception as exc:
            logger.warning("Decomposition failed: %s", exc)
            return [ReasoningStep(1, task)]

        steps = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Accept lines like "1. Do something" or "1) Do something"
            match = re.match(r"^(\d+)[.)]\s+(.*)", line)
            if match:
                num = int(match.group(1))
                desc = match.group(2)
            else:
                num = len(steps) + 1
                desc = line
            steps.append(ReasoningStep(num, desc))
        return steps or [ReasoningStep(1, task)]

    async def _execute_step(
        self, original_task: str, step_description: str, accumulated: str
    ) -> str:
        prompt = (
            f"Overall task: {original_task}\n"
            f"Progress so far:\n{accumulated}\n\n"
            f"Now complete this step: {step_description}"
        )
        messages = [
            {"role": "system", "content": "You are a careful, step-by-step problem solver."},
            {"role": "user", "content": prompt},
        ]
        try:
            return await self.llm.complete(messages, task_type=TaskType.REASONING)
        except Exception as exc:
            return f"[Step failed: {exc}]"

    async def _reflect(self, task: str, accumulated_result: str) -> tuple:
        """Return (answer, confidence, issues)."""
        prompt = (
            f"Original task: {task}\n\n"
            f"Reasoning output:\n{accumulated_result}"
        )
        messages = [
            {"role": "system", "content": _REFLECT_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await self.llm.complete(messages, task_type=TaskType.REASONING)
        except Exception as exc:
            logger.warning("Reflection failed: %s", exc)
            return accumulated_result, 0.5, []

        # Parse structured response
        confidence = 0.5
        issues: List[str] = []
        improved = accumulated_result

        for line in raw.splitlines():
            if line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    logger.debug("Could not parse CONFIDENCE value from reflection response")
            elif line.startswith("ISSUES:"):
                issues_text = line.split(":", 1)[1].strip()
                if issues_text.lower() != "none":
                    issues = [i.strip() for i in issues_text.split(",")]
            elif line.startswith("IMPROVED_ANSWER:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate.lower() != "same":
                    improved = candidate

        return improved, min(max(confidence, 0.0), 1.0), issues

    async def _improve(
        self, task: str, current_answer: str, issues: List[str], context: str
    ) -> str:
        issues_text = "\n".join(f"- {i}" for i in issues) if issues else "general quality"
        prompt = (
            f"Task: {task}\n\n"
            f"Current answer:\n{current_answer}\n\n"
            f"Issues identified:\n{issues_text}\n\n"
            f"Context:\n{context}\n\n"
            "Please provide an improved, more complete and accurate answer."
        )
        messages = [
            {"role": "system", "content": "You are an expert at refining and improving AI-generated answers."},
            {"role": "user", "content": prompt},
        ]
        try:
            return await self.llm.complete(messages, task_type=TaskType.REASONING)
        except Exception as exc:
            logger.warning("Improvement step failed: %s", exc)
            return current_answer
