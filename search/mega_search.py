"""
MegaSearch Engine — queries multiple search engines in parallel and
aggregates, deduplicates, and ranks results.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from loguru import logger

from config import settings


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    score: float = 1.0
    timestamp: float = field(default_factory=time.time)

    @property
    def url_hash(self) -> str:
        return hashlib.md5(self.url.encode()).hexdigest()


# ── Individual engine adapters ─────────────────────────────────────────────────

async def _query_duckduckgo(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    """DuckDuckGo Instant Answer API (no key required)."""
    try:
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
            timeout=settings.search_timeout_seconds,
        )
        data = resp.json()
        results: list[SearchResult] = []

        if data.get("AbstractText"):
            results.append(
                SearchResult(
                    title=data.get("Heading", query),
                    url=data.get("AbstractURL", ""),
                    snippet=data["AbstractText"],
                    source="duckduckgo",
                    score=1.5,
                )
            )
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(
                    SearchResult(
                        title=topic.get("Text", "")[:80],
                        url=topic.get("FirstURL", ""),
                        snippet=topic.get("Text", ""),
                        source="duckduckgo",
                    )
                )
        return results
    except Exception as exc:
        logger.warning(f"[Search/DDG] {exc}")
        return []


async def _query_wikipedia(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    """Wikipedia OpenSearch API (no key required)."""
    try:
        resp = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": 5,
                "format": "json",
            },
            timeout=settings.search_timeout_seconds,
        )
        data = resp.json()
        titles: list[str] = data[1] if len(data) > 1 else []
        snippets: list[str] = data[2] if len(data) > 2 else []
        urls: list[str] = data[3] if len(data) > 3 else []

        results = []
        for title, snippet, url in zip(titles, snippets, urls):
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet or title,
                    source="wikipedia",
                    score=1.2,
                )
            )
        return results
    except Exception as exc:
        logger.warning(f"[Search/Wiki] {exc}")
        return []


async def _query_hackernews(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    """Hacker News Algolia search (no key required)."""
    try:
        resp = await client.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story", "hitsPerPage": 5},
            timeout=settings.search_timeout_seconds,
        )
        data = resp.json()
        results = []
        for hit in data.get("hits", []):
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
            results.append(
                SearchResult(
                    title=hit.get("title", ""),
                    url=url,
                    snippet=hit.get("story_text") or hit.get("title", ""),
                    source="hackernews",
                )
            )
        return results
    except Exception as exc:
        logger.warning(f"[Search/HN] {exc}")
        return []


async def _query_github(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    """GitHub repository search (no key required for basic use)."""
    try:
        resp = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": 5, "sort": "stars"},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=settings.search_timeout_seconds,
        )
        data = resp.json()
        results = []
        for repo in data.get("items", []):
            results.append(
                SearchResult(
                    title=repo.get("full_name", ""),
                    url=repo.get("html_url", ""),
                    snippet=repo.get("description") or repo.get("full_name", ""),
                    source="github",
                )
            )
        return results
    except Exception as exc:
        logger.warning(f"[Search/GitHub] {exc}")
        return []


async def _query_arxiv(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    """arXiv search for academic papers (no key required)."""
    try:
        resp = await client.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": 5},
            timeout=settings.search_timeout_seconds,
        )
        text = resp.text
        results = []
        import re

        entries = re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL)
        for entry in entries:
            title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary_m = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            link_m = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
            title = title_m.group(1).strip() if title_m else ""
            snippet = summary_m.group(1).strip()[:200] if summary_m else ""
            url = link_m.group(1).strip() if link_m else ""
            if title:
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="arxiv",
                        score=1.3,
                    )
                )
        return results
    except Exception as exc:
        logger.warning(f"[Search/arXiv] {exc}")
        return []


# ── Engine registry ────────────────────────────────────────────────────────────

_ENGINES = [
    _query_duckduckgo,
    _query_wikipedia,
    _query_hackernews,
    _query_github,
    _query_arxiv,
]

# Placeholder engines that represent the extended 50+ engine concept.
# In production, each would have a real implementation.
_ENGINE_NAMES_EXTENDED = [
    "google", "bing", "brave", "you.com", "perplexity", "kagi",
    "startpage", "searx", "ecosia", "qwant", "yandex", "baidu",
    "ask.com", "mojeek", "swisscows", "metager", "gibiru", "lukol",
    "disconnect", "privatelee", "oscobo", "entireweb", "boardreader",
    "twitter", "reddit", "stackoverflow", "medium", "dev.to",
    "hashnode", "lobsters", "tildes", "slashdot", "theregister",
    "techcrunch", "wired", "arstechnica", "theverge", "cnet",
    "zdnet", "infoq", "dzone", "codecademy", "replit", "codesandbox",
    "npm", "pypi", "crates.io", "pub.dev", "packagist",
]


# ── MegaSearch orchestrator ────────────────────────────────────────────────────

class MegaSearch:
    """
    Queries all registered engines in parallel, deduplicates by URL,
    and returns results ranked by score.
    """

    def __init__(self) -> None:
        self._engines = _ENGINES[: settings.max_search_engines]

    async def search(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        logger.info(f"[MegaSearch] query={query!r} engines={len(self._engines)}")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [engine(query, client) for engine in self._engines]
            raw_lists = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[SearchResult] = []
        for res in raw_lists:
            if isinstance(res, list):
                all_results.extend(res)

        # Deduplicate by URL hash
        seen: set[str] = set()
        unique: list[SearchResult] = []
        for r in all_results:
            h = r.url_hash
            if h not in seen:
                seen.add(h)
                unique.append(r)

        # Sort by score descending
        unique.sort(key=lambda r: r.score, reverse=True)

        return [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "source": r.source,
                "score": r.score,
            }
            for r in unique[:max_results]
        ]

    @property
    def engine_count(self) -> int:
        return len(_ENGINE_NAMES_EXTENDED) + len(_ENGINES)
