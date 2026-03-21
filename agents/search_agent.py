"""WISPR Search Agent — wraps MegaSearch and synthesises results via LLM."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agents.base_agent import AgentResult, BaseAgent
from llm.router import TaskType
from search.mega_search import MegaSearch

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are WISPR Search — an expert research analyst.
You receive raw search results from multiple engines and must:
1. Synthesise the most relevant information into a concise, factual answer.
2. Cite sources where possible.
3. Flag any contradictions or uncertainty you detect across sources.
4. Keep hallucinations to zero — if information is absent, say so.
"""


class SearchAgent(BaseAgent):
    """Queries 50+ search engines concurrently and synthesises results."""

    name = "SearchAgent"
    task_type = TaskType.SEARCH

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._search_engine = MegaSearch()

    async def run(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        ctx = context or {}
        num_results = ctx.get("num_results", 10)

        try:
            raw_results = await self._search_engine.search(task, max_results=num_results)
        except Exception as exc:
            return self._err(f"MegaSearch failed: {exc}")

        if not raw_results:
            return self._err("No search results found.")

        results_text = self._format_results(raw_results)
        user_prompt = (
            f"Query: {task}\n\nSearch results:\n{results_text}\n\n"
            "Please synthesise a comprehensive, accurate answer."
        )

        try:
            synthesis = await self._llm_complete(_SYSTEM_PROMPT, user_prompt)
            return self._ok(synthesis, raw_results=raw_results, sources=len(raw_results))
        except Exception as exc:
            return self._err(f"Synthesis failed: {exc}")

    @staticmethod
    def _format_results(results: list) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            lines.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}")
        return "\n\n".join(lines)
