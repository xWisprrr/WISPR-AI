"""
MegaSearch Agent — runs MegaSearch and uses an LLM to summarise the results.
"""
from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from agents.base_agent import AgentResult, BaseAgent
from llm.router import TaskType, chat_async
from search.mega_search import MegaSearch


class SearchAgent(BaseAgent):
    """Queries 50+ sources in parallel and returns a synthesised answer."""

    name = "search"
    description = "MegaSearch: queries 50+ engines in parallel, aggregates results."

    def __init__(self) -> None:
        super().__init__()
        self._mega = MegaSearch()

    async def run(self, task: str, context: Optional[dict[str, Any]] = None) -> AgentResult:
        logger.info(f"[SearchAgent] query={task[:80]!r}")

        raw_results = await self._mega.search(task, max_results=10)

        if not raw_results:
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.name,
                success=True,
                output="No results found.",
                metadata={"results": []},
            )

        # Format results for the LLM
        results_text = "\n\n".join(
            f"[{i+1}] **{r['title']}** ({r['source']})\n{r['snippet']}\nURL: {r['url']}"
            for i, r in enumerate(raw_results)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are WISPR Search, an expert at synthesising search results. "
                    "Given a query and search results, produce a concise, accurate summary "
                    "that directly answers the query. Cite sources by number."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {task}\n\nSearch Results:\n{results_text}",
            },
        ]

        summary = await chat_async(messages=messages, task=TaskType.SEARCH)

        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.name,
            success=True,
            output=summary,
            metadata={"results": raw_results, "engine_count": self._mega.engine_count},
        )
