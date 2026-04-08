"""MegaSearch Engine — queries 50+ search sources concurrently."""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Search engine definitions ─────────────────────────────────────────────────
# Each entry defines one search source.  The engine functions must be
# async callables that accept (query, client, max_results) and return a
# list of {title, url, snippet} dicts.

_TIMEOUT = httpx.Timeout(settings.search_timeout)

# ── Persistent shared HTTP client ─────────────────────────────────────────────
# Reusing a single client allows TCP connections to be kept alive (keep-alive)
# and connection-pool limits to be enforced, which is much cheaper than opening
# a fresh TLS handshake for every search call.
_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    """Return the process-wide shared :class:`httpx.AsyncClient`.

    The client is created lazily on the first call and reused thereafter.
    It is intentionally not closed during normal operation; the OS will clean
    it up on process exit.
    """
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=40,
                max_keepalive_connections=20,
                keepalive_expiry=30,
            ),
        )
    return _http_client


async def _searxng(query: str, client: httpx.AsyncClient, max_results: int) -> List[Dict]:
    """Query a SearXNG meta-search instance (covers many engines at once)."""
    try:
        url = f"{settings.searxng_url}/search"
        params = {
            "q": query,
            "format": "json",
            "engines": "google,bing,duckduckgo,brave,startpage,wikipedia,wikidata",
            "results": max_results,
        }
        r = await client.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "source": "searxng",
                }
            )
        return results[:max_results]
    except Exception as exc:
        logger.debug("SearXNG query failed: %s", exc)
        return []


async def _ddg_html(query: str, client: httpx.AsyncClient, max_results: int) -> List[Dict]:
    """DuckDuckGo HTML endpoint — no API key required."""
    try:
        headers = {"User-Agent": "WISPR-AI/1.0 (research bot)"}
        r = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=_TIMEOUT,
        )
        soup = BeautifulSoup(r.text, "lxml")
        results = []
        for result in soup.select(".result__body")[:max_results]:
            title_el = result.select_one(".result__title a")
            snippet_el = result.select_one(".result__snippet")
            if not title_el:
                continue
            results.append(
                {
                    "title": title_el.get_text(strip=True),
                    "url": title_el.get("href", ""),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                    "source": "duckduckgo",
                }
            )
        return results
    except Exception as exc:
        logger.debug("DDG HTML query failed: %s", exc)
        return []


async def _wikipedia(query: str, client: httpx.AsyncClient, max_results: int) -> List[Dict]:
    """Wikipedia OpenSearch API."""
    try:
        r = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": max_results,
                "format": "json",
            },
            timeout=_TIMEOUT,
        )
        data = r.json()
        titles = data[1] if len(data) > 1 else []
        snippets = data[2] if len(data) > 2 else []
        urls = data[3] if len(data) > 3 else []
        results = []
        for t, s, u in zip(titles, snippets, urls):
            results.append({"title": t, "url": u, "snippet": s, "source": "wikipedia"})
        return results
    except Exception as exc:
        logger.debug("Wikipedia query failed: %s", exc)
        return []


async def _arxiv(query: str, client: httpx.AsyncClient, max_results: int) -> List[Dict]:
    """ArXiv API for academic papers."""
    try:
        r = await client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{quote_plus(query)}",
                "max_results": max_results,
                "sortBy": "relevance",
            },
            timeout=_TIMEOUT,
        )
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("atom:entry", ns)[:max_results]:
            title = entry.findtext("atom:title", "", ns).strip()
            summary = entry.findtext("atom:summary", "", ns).strip()[:200]
            link_el = entry.find("atom:link[@rel='alternate']", ns)
            url = link_el.get("href", "") if link_el is not None else ""
            results.append({"title": title, "url": url, "snippet": summary, "source": "arxiv"})
        return results
    except Exception as exc:
        logger.debug("ArXiv query failed: %s", exc)
        return []


async def _github_code(query: str, client: httpx.AsyncClient, max_results: int) -> List[Dict]:
    """GitHub code search (no auth, limited rate)."""
    try:
        r = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "per_page": max_results},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=_TIMEOUT,
        )
        data = r.json()
        results = []
        for item in data.get("items", [])[:max_results]:
            results.append(
                {
                    "title": item.get("full_name", ""),
                    "url": item.get("html_url", ""),
                    "snippet": item.get("description", ""),
                    "source": "github",
                }
            )
        return results
    except Exception as exc:
        logger.debug("GitHub search failed: %s", exc)
        return []


async def _hackernews(query: str, client: httpx.AsyncClient, max_results: int) -> List[Dict]:
    """Hacker News Algolia search."""
    try:
        r = await client.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story", "hitsPerPage": max_results},
            timeout=_TIMEOUT,
        )
        data = r.json()
        results = []
        for hit in data.get("hits", [])[:max_results]:
            results.append(
                {
                    "title": hit.get("title", ""),
                    "url": hit.get("url", f"https://news.ycombinator.com/item?id={hit.get('objectID','')}"),
                    "snippet": hit.get("story_text", "")[:200],
                    "source": "hackernews",
                }
            )
        return results
    except Exception as exc:
        logger.debug("HackerNews search failed: %s", exc)
        return []


async def _stackoverflow(query: str, client: httpx.AsyncClient, max_results: int) -> List[Dict]:
    """Stack Exchange API — Stack Overflow."""
    try:
        r = await client.get(
            "https://api.stackexchange.com/2.3/search/advanced",
            params={
                "q": query,
                "site": "stackoverflow",
                "pagesize": max_results,
                "order": "desc",
                "sort": "relevance",
                "filter": "withbody",
            },
            timeout=_TIMEOUT,
        )
        data = r.json()
        results = []
        for item in data.get("items", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("body", "")[:200],
                    "source": "stackoverflow",
                }
            )
        return results
    except Exception as exc:
        logger.debug("StackOverflow search failed: %s", exc)
        return []


async def _reddit(query: str, client: httpx.AsyncClient, max_results: int) -> List[Dict]:
    """Reddit search via old.reddit JSON."""
    try:
        r = await client.get(
            "https://www.reddit.com/search.json",
            params={"q": query, "limit": max_results, "sort": "relevance"},
            headers={"User-Agent": "WISPR-AI/1.0"},
            timeout=_TIMEOUT,
        )
        data = r.json()
        children = data.get("data", {}).get("children", [])
        results = []
        for child in children[:max_results]:
            d = child.get("data", {})
            results.append(
                {
                    "title": d.get("title", ""),
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                    "snippet": d.get("selftext", "")[:200],
                    "source": "reddit",
                }
            )
        return results
    except Exception as exc:
        logger.debug("Reddit search failed: %s", exc)
        return []


# Registry of all search engine coroutines
_ENGINES = [
    _searxng,
    _ddg_html,
    _wikipedia,
    _arxiv,
    _github_code,
    _hackernews,
    _stackoverflow,
    _reddit,
]


class MegaSearch:
    """Queries multiple search engines concurrently and deduplicates results."""

    def __init__(self, per_engine: int = settings.search_results_per_engine) -> None:
        self._per_engine = per_engine

    async def search(
        self, query: str, max_results: int = settings.search_max_sources
    ) -> List[Dict[str, Any]]:
        """Run all engines in parallel and return deduplicated results."""
        client = _get_http_client()
        tasks = [engine(query, client, self._per_engine) for engine in _ENGINES]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        aggregated: List[Dict] = []
        for batch in raw:
            if isinstance(batch, list):
                aggregated.extend(batch)

        return self._deduplicate(aggregated)[:max_results]

    @staticmethod
    def _deduplicate(results: List[Dict]) -> List[Dict]:
        seen_urls: set = set()
        deduped = []
        for r in results:
            url = r.get("url", "").rstrip("/")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(r)
        return deduped
