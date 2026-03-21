"""
Autonomous Reasoning Engine — multi-step reasoning with self-reflection.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from loguru import logger

from config import settings
from llm.router import TaskType, chat_async


class ReasoningStep:
    def __init__(self, step_num: int, thought: str, action: str, observation: str) -> None:
        self.step_num = step_num
        self.thought = thought
        self.action = action
        self.observation = observation

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
        }


class ReasoningEngine:
    """
    Implements a ReAct-style (Reason + Act) loop:
    1. Think about the current state
    2. Decide on an action
    3. Observe the result
    4. Repeat until a satisfactory answer is reached or max steps hit
    """

    def __init__(self, max_steps: int = settings.max_reasoning_steps) -> None:
        self._max_steps = max_steps

    async def reason(
        self,
        task: str,
        context: Optional[str] = None,
        available_tools: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Execute the reasoning loop.

        Returns:
            {
                "answer": str,
                "steps": list[dict],
                "confidence": float,
                "iterations": int,
            }
        """
        tools = available_tools or ["search", "code", "analyze", "reflect"]
        steps: list[ReasoningStep] = []

        system_prompt = (
            "You are an autonomous reasoning agent. "
            "For each step output EXACTLY this JSON:\n"
            '{"thought": "...", "action": "search|code|analyze|reflect|answer", '
            '"observation": "...", "confidence": 0.0-1.0, "final": false}\n'
            'When you have a final answer set "final": true and put the answer in "observation".\n'
            "Be concise. Output ONLY JSON."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Task: {task}\n"
                    + (f"Context: {context}\n" if context else "")
                    + f"Available actions: {', '.join(tools)}"
                ),
            },
        ]

        confidence = 0.0
        answer = ""

        for i in range(self._max_steps):
            raw = await chat_async(messages=messages, task=TaskType.REASONING, max_tokens=512)

            parsed = self._parse_step(raw)
            step = ReasoningStep(
                step_num=i + 1,
                thought=parsed.get("thought", ""),
                action=parsed.get("action", "reflect"),
                observation=parsed.get("observation", ""),
            )
            steps.append(step)
            confidence = float(parsed.get("confidence", 0.5))

            logger.debug(
                f"[Reasoning] step={i+1} action={step.action} conf={confidence:.2f}"
            )

            if parsed.get("final") or confidence >= settings.min_confidence_threshold:
                answer = step.observation
                break

            # Feed observation back as assistant message for next iteration
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Continue reasoning. Output next step JSON.",
                }
            )
        else:
            # Max steps reached — use last observation
            answer = steps[-1].observation if steps else "Unable to determine answer."

        # Self-reflection pass
        if answer:
            answer = await self._reflect(task, answer)

        return {
            "answer": answer,
            "steps": [s.to_dict() for s in steps],
            "confidence": confidence,
            "iterations": len(steps),
        }

    async def _reflect(self, task: str, answer: str) -> str:
        """Ask the model to critique and improve its own answer."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a critical reviewer. Review the answer below for accuracy, "
                    "completeness, and clarity. If it is good, return it as-is. "
                    "If it has issues, return a corrected version. Output ONLY the answer."
                ),
            },
            {
                "role": "user",
                "content": f"Original task: {task}\n\nAnswer to review:\n{answer}",
            },
        ]
        try:
            return await chat_async(messages=messages, task=TaskType.REASONING, max_tokens=1024)
        except Exception:
            return answer

    @staticmethod
    def _parse_step(raw: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"thought": raw, "action": "reflect", "observation": raw, "confidence": 0.3}
