"""
Tests for the Hallucination Reducer (non-LLM paths).
"""
from __future__ import annotations

import pytest

from hallucination.reducer import HallucinationReducer
from agents.base_agent import AgentResult


class TestHallucinationReducer:
    def setup_method(self):
        self._reducer = HallucinationReducer()

    def _make_result(self, output: str, confidence: float = 0.8) -> AgentResult:
        return AgentResult(
            agent_id="test",
            agent_name="test",
            success=True,
            output=output,
            metadata={"confidence": confidence},
        )

    def test_entity_consistency_single_output(self):
        results = [self._make_result("The answer is 42")]
        score = self._reducer._check_entity_consistency(results)
        assert score == 1.0  # single output → no disagreement

    def test_entity_consistency_matching_numbers(self):
        results = [
            self._make_result("Population is 8000 people"),
            self._make_result("There are 8000 individuals"),
        ]
        score = self._reducer._check_entity_consistency(results)
        assert score == 1.0  # both contain 8000

    def test_entity_consistency_disagreement(self):
        results = [
            self._make_result("The count is 100"),
            self._make_result("The count is 200"),
        ]
        score = self._reducer._check_entity_consistency(results)
        assert score < 1.0

    def test_entity_consistency_no_numbers(self):
        results = [
            self._make_result("Python is great"),
            self._make_result("Python is wonderful"),
        ]
        score = self._reducer._check_entity_consistency(results)
        assert score == 1.0  # no numeric entities to disagree on

    def test_entity_consistency_empty_results(self):
        score = self._reducer._check_entity_consistency([])
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_verify_no_llm(self):
        """verify() without LLM check should still return a dict with required keys."""
        results = [self._make_result("The sky is blue", confidence=0.9)]
        output = await self._reducer.verify(
            task="What colour is the sky?",
            combined_output="The sky is blue",
            agent_results=results,
            run_llm_check=False,
        )
        assert "answer" in output
        assert "confidence" in output
        assert "checks" in output
        assert output["answer"] == "The sky is blue"
        assert 0.0 <= output["confidence"] <= 1.0
