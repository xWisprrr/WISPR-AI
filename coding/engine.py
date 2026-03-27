"""Multi-Language Coding Engine — detection, generation, debug, and translation."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from llm.router import LLMRouter, TaskType
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Language detection heuristics ─────────────────────────────────────────────
_LANG_PATTERNS: Dict[str, List[str]] = {
    "python":     [r"\bdef\b", r"\bimport\b", r"\bprint\(", r"\.py\b"],
    "javascript": [r"\bconst\b", r"\blet\b", r"\bconsole\.log\b", r"\.js\b", r"=>\s*{"],
    "typescript": [r"\binterface\b", r"\btype\s+\w+\s*=", r":\s*string", r"\.ts\b"],
    "go":         [r"\bpackage main\b", r"\bfunc\b", r"\bfmt\.Print", r"\.go\b"],
    "rust":       [r"\bfn main\b", r"\blet mut\b", r"\bprintln!", r"\.rs\b"],
    "java":       [r"\bpublic class\b", r"\bSystem\.out\.print", r"\.java\b"],
    "c":          [r"#include\s*<stdio\.h>", r"\bprintf\b", r"\.c\b"],
    "cpp":        [r"#include\s*<iostream>", r"\bstd::", r"\bcout\b", r"\.cpp\b"],
    "ruby":       [r"\bputs\b", r"\bend\b", r"\bdo\s*\|", r"\.rb\b"],
    "bash":       [r"^#!/bin/bash", r"\becho\b", r"\b\$\{", r"\.sh\b"],
    "sql":        [r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b"],
    "html":       [r"<html", r"<div", r"<!DOCTYPE"],
    "css":        [r"\{.*:.*;\}", r"^@media", r"\.css\b"],
}

SUPPORTED_LANGUAGES = list(_LANG_PATTERNS.keys())

_GENERATE_SYSTEM = """\
You are an elite software engineer. Generate complete, production-quality code.
Include type hints, error handling, and brief inline comments.
Always wrap code in a single fenced code block with the language identifier.
"""

_DEBUG_SYSTEM = """\
You are an expert debugger. Analyse the provided code and error, identify the
root cause, then provide a fixed version in a fenced code block.
Explain the fix briefly after the code block.
"""

_OPTIMISE_SYSTEM = """\
You are a performance engineer. Optimise the provided code for:
- Speed
- Memory efficiency
- Readability
Return the optimised code in a fenced code block, then list the key changes made.
"""

_TRANSLATE_SYSTEM = """\
You are a polyglot software engineer. Translate the code from the source language
to the target language, preserving all logic and behaviour.
Return the translated code in a single fenced code block.
"""


class CodingEngine:
    """High-level coding API: generate, debug, optimise, translate."""

    def __init__(self, llm_router: Optional[LLMRouter] = None) -> None:
        self.llm = llm_router or LLMRouter()

    # ── Public API ────────────────────────────────────────────────────────

    def detect_language(self, code: str) -> str:
        """Detect the programming language of *code* via heuristics."""
        scores: Dict[str, int] = {}
        for lang, patterns in _LANG_PATTERNS.items():
            score = sum(
                1 for p in patterns if re.search(p, code, flags=re.IGNORECASE | re.MULTILINE)
            )
            if score:
                scores[lang] = score
        if not scores:
            return "unknown"
        return max(scores, key=lambda k: scores[k])

    async def generate(
        self,
        description: str,
        language: str = "python",
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate code for *description* in *language*."""
        user_prompt = f"Language: {language}\nTask: {description}"
        if context:
            user_prompt += f"\nContext/constraints: {context}"

        messages = [
            {"role": "system", "content": _GENERATE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = await self.llm.complete(messages, task_type=TaskType.CODING)
            blocks = self._extract_blocks(response)
            return {
                "success": True,
                "code": blocks[0] if blocks else response,
                "full_response": response,
                "language": language,
                "num_blocks": len(blocks),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "language": language}

    async def debug(
        self, code: str, error: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Identify and fix the bug in *code* that causes *error*."""
        lang = language or self.detect_language(code)
        user_prompt = (
            f"Language: {lang}\n\nCode:\n```{lang}\n{code}\n```\n\nError:\n{error}"
        )
        messages = [
            {"role": "system", "content": _DEBUG_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = await self.llm.complete(messages, task_type=TaskType.CODING)
            blocks = self._extract_blocks(response)
            return {
                "success": True,
                "fixed_code": blocks[0] if blocks else "",
                "explanation": response,
                "language": lang,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "language": lang}

    async def optimise(
        self, code: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return an optimised version of *code*."""
        lang = language or self.detect_language(code)
        user_prompt = f"Language: {lang}\n\nCode:\n```{lang}\n{code}\n```"
        messages = [
            {"role": "system", "content": _OPTIMISE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = await self.llm.complete(messages, task_type=TaskType.CODING)
            blocks = self._extract_blocks(response)
            return {
                "success": True,
                "optimised_code": blocks[0] if blocks else "",
                "explanation": response,
                "language": lang,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "language": lang}

    async def translate(
        self, code: str, target_language: str, source_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Translate *code* to *target_language*."""
        src_lang = source_language or self.detect_language(code)
        user_prompt = (
            f"Source language: {src_lang}\n"
            f"Target language: {target_language}\n\n"
            f"Code:\n```{src_lang}\n{code}\n```"
        )
        messages = [
            {"role": "system", "content": _TRANSLATE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = await self.llm.complete(messages, task_type=TaskType.CODING)
            blocks = self._extract_blocks(response)
            return {
                "success": True,
                "translated_code": blocks[0] if blocks else "",
                "source_language": src_lang,
                "target_language": target_language,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_blocks(text: str) -> List[str]:
        # Match fenced blocks with or without a language identifier, with or without
        # a newline immediately after the opening fence (e.g. ```python\n or ```python ).
        return re.findall(r"```(?:\w+)?\n?(.*?)```", text, flags=re.DOTALL)
