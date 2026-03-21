"""
Tests for the Coding Engine.
"""
from __future__ import annotations

import pytest

from coding.engine import CodingEngine


class TestCodingEngine:
    def setup_method(self):
        self._engine = CodingEngine()

    # ── Language detection ─────────────────────────────────────────────────────

    def test_detect_python(self):
        assert self._engine.detect_language("Write a Flask app in Python") == "python"

    def test_detect_javascript(self):
        assert self._engine.detect_language("Build a React component with JavaScript") == "javascript"

    def test_detect_rust(self):
        assert self._engine.detect_language("Implement binary search in Rust using Cargo") == "rust"

    def test_detect_go(self):
        assert self._engine.detect_language("Write a Golang HTTP server") == "go"

    def test_detect_sql(self):
        assert self._engine.detect_language("Write a SQL query joining two PostgreSQL tables") == "sql"

    def test_detect_default(self):
        # Unrecognised → default python
        assert self._engine.detect_language("do something") == "python"

    # ── Prompt building ────────────────────────────────────────────────────────

    def test_build_generate_prompt(self):
        prompt = self._engine.build_prompt("sort a list", "python", "generate")
        assert "python" in prompt.lower()
        assert "sort a list" in prompt

    def test_build_debug_prompt(self):
        prompt = self._engine.build_prompt("buggy code here", "rust", "debug")
        assert "debug" in prompt.lower() or "fix" in prompt.lower()

    def test_build_unknown_action_defaults_to_generate(self):
        prompt = self._engine.build_prompt("task", "python", "unknown_action")
        assert "task" in prompt

    # ── Code extraction ────────────────────────────────────────────────────────

    def test_extract_code_blocks_single(self):
        text = "Here is the code:\n```python\nprint('hello')\n```"
        blocks = self._engine.extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"
        assert "print" in blocks[0]["code"]

    def test_extract_code_blocks_multiple(self):
        text = (
            "```python\nx = 1\n```\n"
            "Some text\n"
            "```javascript\nconsole.log(1)\n```"
        )
        blocks = self._engine.extract_code_blocks(text)
        assert len(blocks) == 2

    def test_extract_code_blocks_no_lang_label(self):
        text = "```\nsome code\n```"
        blocks = self._engine.extract_code_blocks(text)
        assert blocks[0]["language"] == "text"

    def test_extract_code_blocks_empty(self):
        assert self._engine.extract_code_blocks("no fences here") == []

    # ── Complexity estimation ─────────────────────────────────────────────────

    def test_complexity_low(self):
        code = "x = 1\ny = 2\nprint(x + y)"
        assert self._engine.estimate_complexity(code) == "low"

    def test_complexity_high(self):
        # Simulate a deeply-nested long file
        lines = []
        for i in range(200):
            indent = "    " * (i % 6)
            lines.append(f"{indent}x = {i}")
        code = "\n".join(lines)
        assert self._engine.estimate_complexity(code) == "high"

    # ── Supported languages ───────────────────────────────────────────────────

    def test_supported_languages_non_empty(self):
        assert len(CodingEngine.SUPPORTED_LANGUAGES) >= 10
