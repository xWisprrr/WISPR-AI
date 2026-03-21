"""
Tests for the LLM Router (model selection, no actual LLM calls needed).
"""
from __future__ import annotations

import pytest

from llm.router import TaskType, select_model
from config import settings


class TestLLMRouter:
    def test_select_reasoning_model(self):
        model = select_model(TaskType.REASONING)
        assert model == settings.reasoning_model

    def test_select_coding_model(self):
        model = select_model(TaskType.CODING)
        assert model == settings.coding_model

    def test_select_search_model(self):
        model = select_model(TaskType.SEARCH)
        assert model == settings.search_model

    def test_select_fast_model(self):
        model = select_model(TaskType.FAST)
        assert model == settings.fast_model

    def test_select_general_model(self):
        model = select_model(TaskType.GENERAL)
        assert model == settings.fast_model

    def test_task_types_exist(self):
        expected = {"reasoning", "coding", "search", "fast", "general"}
        actual = {t.value for t in TaskType}
        assert expected == actual
