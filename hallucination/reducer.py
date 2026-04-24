"""Hallucination Reduction System — cross-checking, voting, and verification."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from llm.router import LLMRouter, TaskType
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_VERIFY_PROMPT = """\
You are a fact-checking AI. Given a claim and optional search evidence, assess:
1. Is the claim factually supported?
2. Assign a confidence score (0.0–1.0).

Respond ONLY in this format:
SUPPORTED: yes/no/partial
CONFIDENCE: <float>
EXPLANATION: <one sentence>
"""

_COMPARE_PROMPT = """\
You are an expert at comparing multiple AI outputs for the same question.
Your job: identify the most accurate and complete answer, combining the best
elements from all responses.

Output the synthesised best answer as plain prose — no headers or meta-commentary.
"""


class _VerificationCache:
    """Simple TTL cache for claim verification results.

    Keyed by a SHA-256 hash of (claim, truncated evidence) so identical
    verification requests are served without an extra LLM call.
    """

    def __init__(self, maxsize: int = 256, ttl: int = 1800) -> None:
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._maxsize = maxsize
        self._ttl = ttl

    def _key(self, claim: str, evidence_summary: str) -> str:
        payload = json.dumps({"c": claim, "e": evidence_summary}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, claim: str, evidence_summary: str) -> Optional[Dict[str, Any]]:
        key = self._key(claim, evidence_summary)
        entry = self._cache.get(key)
        if entry is None:
            return None
        result, expires_at = entry
        if time.monotonic() < expires_at:
            return result
        del self._cache[key]
        return None

    def set(self, claim: str, evidence_summary: str, result: Dict[str, Any]) -> None:
        if len(self._cache) >= self._maxsize:
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        key = self._key(claim, evidence_summary)
        self._cache[key] = (result, time.monotonic() + self._ttl)


_verification_cache = _VerificationCache()


class HallucinationReducer:
    """Reduces hallucinations via multi-agent voting and search verification."""

    def __init__(self, llm_router: Optional[LLMRouter] = None) -> None:
        self.llm = llm_router or LLMRouter()

    # -- public API -----------------------------------------------------------

    async def vote(self, question: str, num_votes: int = 3) -> Dict[str, Any]:
        """Sample the LLM multiple times and return the majority-voted answer."""
        tasks = [self._single_sample(question) for _ in range(num_votes)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in responses if isinstance(r, str) and r.strip()]

        if not valid:
            return {"answer": "", "confidence": 0.0, "method": "vote", "votes": 0}

        best = await self._find_consensus(question, valid)
        confidence = await self._score_confidence(question, best)

        return {
            "answer": best,
            "confidence": confidence,
            "method": "majority_vote",
            "votes": len(valid),
            "candidates": valid,
        }

    async def verify_with_search(
        self,
        claim: str,
        search_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify a claim against actual search results.

        Results are cached so identical (claim, evidence) pairs are not
        re-verified with a fresh LLM call within the TTL window.
        """
        evidence_lines = [
            f"- [{r.get('source','?')}] {r.get('title','')}: {r.get('snippet','')}"
            for r in search_results[:10]
        ]
        evidence = "\n".join(evidence_lines)
        # Use a truncated summary as the cache key to avoid huge keys
        evidence_summary = evidence[:500]

        cached = _verification_cache.get(claim, evidence_summary)
        if cached is not None:
            logger.debug("Verification cache hit for claim (len=%d)", len(claim))
            return cached

        prompt = f"Claim: {claim}\n\nEvidence:\n{evidence}"
        messages = [
            {"role": "system", "content": _VERIFY_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await self.llm.complete(messages, task_type=TaskType.GENERAL)
        except Exception as exc:
            return {"supported": "unknown", "confidence": 0.0, "explanation": str(exc)}

        result = self._parse_verify_response(raw)
        _verification_cache.set(claim, evidence_summary, result)
        return result

    async def cross_check(
        self,
        answer: str,
        alternative_answers: List[str],
    ) -> Tuple[str, float]:
        """Compare *answer* against alternatives and return the best with a confidence score."""
        if not alternative_answers:
            conf = await self._score_confidence("", answer)
            return answer, conf

        all_answers = [answer] + alternative_answers
        best = await self._find_consensus("", all_answers)
        confidence = await self._score_confidence("", best)
        return best, confidence

    # -- internals ------------------------------------------------------------

    async def _single_sample(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "Answer the question accurately and concisely."},
            {"role": "user", "content": question},
        ]
        return await self.llm.complete(messages, task_type=TaskType.GENERAL, temperature=0.9)

    async def _find_consensus(self, question: str, answers: List[str]) -> str:
        if len(answers) == 1:
            return answers[0]

        answers_text = "\n\n---\n\n".join(
            f"Response {i+1}:\n{a}" for i, a in enumerate(answers)
        )
        prompt = (
            f"Question: {question}\n\nMultiple AI responses:\n{answers_text}"
            if question
            else f"Multiple AI responses:\n{answers_text}"
        )
        messages = [
            {"role": "system", "content": _COMPARE_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            return await self.llm.complete(messages, task_type=TaskType.REASONING)
        except Exception as exc:
            logger.warning("Consensus finding failed: %s", exc)
            return answers[0]

    async def _score_confidence(self, question: str, answer: str) -> float:
        prompt = (
            f"Question: {question}\nAnswer: {answer}\n\n"
            "Rate how confident you are that this answer is accurate (0.0–1.0). "
            "Output ONLY a float."
        )
        messages = [
            {"role": "system", "content": "You are an objective evaluator."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await self.llm.complete(messages, task_type=TaskType.GENERAL, temperature=0.0)
            return min(max(float(raw.strip()), 0.0), 1.0)
        except Exception as exc:
            logger.debug("Confidence scoring failed: %s", exc)
            return 0.5

    @staticmethod
    def _parse_verify_response(raw: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "supported": "unknown",
            "confidence": 0.5,
            "explanation": "",
        }
        for line in raw.splitlines():
            if line.startswith("SUPPORTED:"):
                result["supported"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    logger.debug("Could not parse CONFIDENCE value from verification response")
            elif line.startswith("EXPLANATION:"):
                result["explanation"] = line.split(":", 1)[1].strip()
        return result
