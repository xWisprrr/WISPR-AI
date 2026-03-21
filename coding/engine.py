"""
Multi-Language Coding Engine — language detection, prompt building, code extraction.
"""
from __future__ import annotations

import re
from typing import Optional


# ── Language patterns ──────────────────────────────────────────────────────────

_LANGUAGE_PATTERNS: dict[str, list[str]] = {
    "python": [r"\bpython\b", r"\bpip\b", r"\.py\b", r"\bdjango\b", r"\bflask\b", r"\bfastapi\b"],
    "javascript": [r"\bjavascript\b", r"\bjs\b", r"\bnode\.?js\b", r"\breact\b", r"\bvue\b"],
    "typescript": [r"\btypescript\b", r"\b\.tsx?\b", r"\bangular\b", r"\bnext\.?js\b"],
    "rust": [r"\brust\b", r"\bcargo\b", r"\bferris\b"],
    "go": [r"\bgolang\b", r"\bgo lang\b", r"\b\.go\b"],
    "java": [r"\bjava\b", r"\bspring\b", r"\bmaven\b", r"\bgradle\b"],
    "cpp": [r"\bc\+\+\b", r"\bcpp\b", r"\bstd::\b"],
    "c": [r"\blanguage c\b", r"\.c\b", r"\blibc\b"],
    "ruby": [r"\bruby\b", r"\brails\b", r"\bgem\b"],
    "php": [r"\bphp\b", r"\blaravel\b", r"\bwordpress\b"],
    "swift": [r"\bswift\b", r"\bxcode\b", r"\bios\b"],
    "kotlin": [r"\bkotlin\b", r"\bandroid\b"],
    "r": [r"\brlang\b", r"\bggplot\b", r"\bdplyr\b"],
    "sql": [r"\bsql\b", r"\bmysql\b", r"\bpostgres\b", r"\bsqlite\b"],
    "bash": [r"\bbash\b", r"\bshell script\b", r"\bsh\b"],
    "html": [r"\bhtml\b", r"\bhtml5\b"],
    "css": [r"\bcss\b", r"\bsass\b", r"\bscss\b"],
}


class CodingEngine:
    """Handles language detection, prompt construction, and code block extraction."""

    SUPPORTED_LANGUAGES: list[str] = list(_LANGUAGE_PATTERNS.keys())

    def detect_language(self, text: str) -> str:
        """Heuristically detect the programming language from free-form text."""
        lower = text.lower()
        scores: dict[str, int] = {}
        for lang, patterns in _LANGUAGE_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, lower))
            if score:
                scores[lang] = score
        if not scores:
            return "python"  # sensible default
        return max(scores, key=lambda k: scores[k])

    def build_prompt(self, task: str, language: str, action: str = "generate") -> str:
        """Construct an LLM prompt for the given coding task."""
        action_instructions = {
            "generate": f"Write complete, production-ready {language} code for the following:\n",
            "debug": f"Debug and fix the following {language} code. Explain each fix:\n",
            "optimize": f"Optimize the following {language} code for performance and readability:\n",
            "translate": f"Translate the following code to {language}:\n",
            "explain": f"Explain the following {language} code in detail:\n",
        }
        prefix = action_instructions.get(action, action_instructions["generate"])
        return prefix + task

    def extract_code_blocks(self, text: str) -> list[dict[str, str]]:
        """Extract all fenced code blocks from markdown text."""
        pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [
            {"language": lang or "text", "code": code.strip()}
            for lang, code in matches
        ]

    def estimate_complexity(self, code: str) -> str:
        """Rough complexity estimate based on line count and nesting."""
        lines = [l for l in code.splitlines() if l.strip()]
        loc = len(lines)
        max_indent = max((len(l) - len(l.lstrip()) for l in lines), default=0) // 4

        if loc < 30 and max_indent < 3:
            return "low"
        elif loc < 150 and max_indent < 5:
            return "medium"
        return "high"
