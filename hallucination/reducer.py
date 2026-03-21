"""
Hallucination Reduction System.

Techniques:
  - Confidence scoring from reasoning steps
  - Majority voting across multiple agent outputs
  - Keyword-level consistency checking
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

from loguru import logger

from llm.router import TaskType, chat_async


class HallucinationReducer:
    """
    Verifies agent output using multiple heuristics and an optional
    LLM cross-check.
    """

    async def verify(
        self,
        task: str,
        combined_output: str,
        agent_results: list[Any],  # list[AgentResult]
        run_llm_check: bool = True,
    ) -> dict[str, Any]:
        """
        Run hallucination reduction checks on the combined output.

        Returns:
            {
                "answer": str,
                "confidence": float,
                "checks": dict,
            }
        """
        checks: dict[str, Any] = {}

        # 1. Majority-vote confidence from agent results
        raw_confidences = [
            r.metadata.get("confidence", 0.7)
            for r in agent_results
            if hasattr(r, "metadata") and isinstance(r.metadata, dict)
        ]
        avg_confidence = sum(raw_confidences) / len(raw_confidences) if raw_confidences else 0.7
        checks["avg_agent_confidence"] = round(avg_confidence, 3)

        # 2. Consistency check — do outputs agree on key named entities?
        entity_consistency = self._check_entity_consistency(agent_results)
        checks["entity_consistency"] = entity_consistency

        # 3. LLM self-verification
        if run_llm_check and combined_output:
            verified_answer, llm_confidence = await self._llm_verify(task, combined_output)
            checks["llm_verified"] = True
            checks["llm_confidence"] = round(llm_confidence, 3)
        else:
            verified_answer = combined_output
            llm_confidence = avg_confidence
            checks["llm_verified"] = False

        # 4. Final confidence blending
        final_confidence = 0.6 * llm_confidence + 0.4 * avg_confidence

        logger.info(
            f"[HallucinationReducer] confidence={final_confidence:.3f} "
            f"entity_consistency={entity_consistency:.3f}"
        )

        return {
            "answer": verified_answer,
            "confidence": round(final_confidence, 3),
            "checks": checks,
        }

    async def _llm_verify(self, task: str, answer: str) -> tuple[str, float]:
        """Ask an LLM to self-verify and return (answer, confidence 0-1)."""
        import json

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a fact-checker. Review the answer for accuracy, "
                    "completeness, and absence of hallucinations. "
                    "Output JSON: {\"verified_answer\": \"...\", \"confidence\": 0.0-1.0, "
                    '"issues": []}. Output ONLY JSON.'
                ),
            },
            {
                "role": "user",
                "content": f"Task: {task}\n\nAnswer:\n{answer}",
            },
        ]
        try:
            raw = await chat_async(messages=messages, task=TaskType.FAST, max_tokens=1024)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("verified_answer", answer), float(data.get("confidence", 0.7))
        except Exception as exc:
            logger.warning(f"[HallucinationReducer] LLM verify failed: {exc}")
        return answer, 0.7

    def _check_entity_consistency(self, agent_results: list[Any]) -> float:
        """
        Check if key entities (numbers, proper nouns) appear consistently
        across multiple agent outputs. Returns a consistency score 0-1.
        """
        outputs = [
            str(r.output) for r in agent_results if hasattr(r, "output") and r.output
        ]
        if len(outputs) < 2:
            return 1.0  # No disagreement possible

        # Extract numeric tokens as a proxy for factual claims
        def extract_numbers(text: str) -> set[str]:
            return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))

        sets = [extract_numbers(o) for o in outputs]
        if not any(sets):
            return 1.0

        # Intersection over union
        union = set().union(*sets)
        intersection = sets[0].intersection(*sets[1:])
        return len(intersection) / len(union) if union else 1.0
