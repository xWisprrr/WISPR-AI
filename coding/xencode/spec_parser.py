"""Parses a README/prompt text into a structured XencodeSpec."""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from llm.router import LLMRouter, TaskType
from coding.xencode.schemas import XencodeSpec

logger = logging.getLogger(__name__)

_LANG_KEYWORDS: Dict[str, List[str]] = {
    "python": ["python", ".py", "pip", "flask", "django", "fastapi", "pytest"],
    "javascript": ["javascript", " js ", "node", "npm", "react", "vue", "express"],
    "typescript": ["typescript", " ts ", "angular", "tsx"],
    "go": ["golang", "go ", ".go", "goroutine"],
    "rust": ["rust", ".rs", "cargo", "crate"],
    "java": ["java", ".java", "maven", "gradle", "spring"],
    "cpp": ["c++", "cpp", ".cpp", "#include <iostream>"],
    "csharp": ["c#", "csharp", ".cs", ".net", "dotnet"],
    "ruby": ["ruby", ".rb", "rails", "bundler", "gem"],
    "bash": ["bash", "shell script", ".sh", "#!/bin/"],
}

_PARSE_SYSTEM = """\
You are a technical spec parser. Given a project description, extract a structured specification.
Return ONLY valid JSON with these exact fields:
{
  "project_name": "slug-format-no-spaces",
  "description": "1-2 sentence description",
  "language": "python|javascript|typescript|go|rust|java|cpp|csharp|ruby|bash",
  "requirements": ["lib1", "lib2"],
  "entry_point": "main.py or index.js etc",
  "build_command": "pip install -r requirements.txt OR null",
  "test_command": "pytest OR null"
}
"""


class SpecParser:
    """Parses a project prompt into a structured XencodeSpec."""

    def __init__(self, llm: Optional[LLMRouter] = None) -> None:
        self.llm = llm or LLMRouter()

    async def parse(
        self,
        prompt: str,
        project_name: str,
        language: str = "python",
    ) -> XencodeSpec:
        """Parse *prompt* into an XencodeSpec, falling back to heuristics on LLM failure."""
        try:
            return await self._parse_with_llm(prompt, project_name, language)
        except Exception as exc:
            logger.warning("LLM spec parse failed (%s), using heuristics.", exc)
            return self._parse_heuristic(prompt, project_name, language)

    async def _parse_with_llm(
        self, prompt: str, project_name: str, language: str
    ) -> XencodeSpec:
        messages = [
            {"role": "system", "content": _PARSE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Project name hint: {project_name}\n"
                    f"Language hint: {language}\n\n"
                    f"Description:\n{prompt}"
                ),
            },
        ]
        raw = await self.llm.complete(messages, task_type=TaskType.CODING, temperature=0.0)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        data = json.loads(raw)
        return XencodeSpec(
            project_name=data.get("project_name", project_name),
            description=data.get("description", prompt[:200]),
            language=data.get("language", language).lower(),
            requirements=data.get("requirements", []),
            entry_point=data.get("entry_point", _default_entry(language)),
            build_command=data.get("build_command"),
            test_command=data.get("test_command"),
        )

    def _parse_heuristic(
        self, prompt: str, project_name: str, language: str
    ) -> XencodeSpec:
        """Best-effort spec extraction without an LLM."""
        lang = _detect_language(prompt, language)
        return XencodeSpec(
            project_name=_slugify(project_name),
            description=prompt.split("\n")[0][:300],
            language=lang,
            requirements=_extract_requirements(prompt, lang),
            entry_point=_default_entry(lang),
            build_command=_default_build(lang),
            test_command=_default_test(lang),
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_language(text: str, default: str) -> str:
    text_lower = text.lower()
    for lang, keywords in _LANG_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return lang
    return default.lower()


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", name.lower()).strip("_") or "project"


def _default_entry(lang: str) -> str:
    return {
        "python": "main.py",
        "javascript": "index.js",
        "typescript": "index.ts",
        "go": "main.go",
        "rust": "src/main.rs",
        "java": "Main.java",
        "cpp": "main.cpp",
        "csharp": "Program.cs",
        "ruby": "main.rb",
        "bash": "main.sh",
    }.get(lang, "main.py")


def _default_build(lang: str) -> Optional[str]:
    return {
        "python": "pip install -r requirements.txt",
        "javascript": "npm install",
        "typescript": "npm install && npx tsc",
        "go": "go build ./...",
        "rust": "cargo build --release",
        "java": "javac Main.java",
        "cpp": "g++ -o app main.cpp",
        "csharp": "dotnet build",
        "ruby": "bundle install",
    }.get(lang)


def _default_test(lang: str) -> Optional[str]:
    return {
        "python": "pytest",
        "javascript": "npm test",
        "typescript": "npm test",
        "go": "go test ./...",
        "rust": "cargo test",
        "java": "mvn test",
        "ruby": "bundle exec rspec",
    }.get(lang)


def _extract_requirements(text: str, lang: str) -> List[str]:
    reqs: List[str] = []
    text_lower = text.lower()
    if lang == "python":
        candidates = [
            "flask", "fastapi", "django", "requests", "pandas", "numpy",
            "sqlalchemy", "pydantic", "aiohttp", "uvicorn", "pytest",
            "httpx", "click", "typer", "rich",
        ]
        reqs = [p for p in candidates if p in text_lower]
    elif lang in ("javascript", "typescript"):
        candidates = ["express", "react", "vue", "axios", "lodash", "jest", "next", "cors"]
        reqs = [p for p in candidates if p in text_lower]
    return reqs
