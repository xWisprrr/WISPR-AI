"""Example WISPR plugin: simple word-frequency text statistics.

Returns word count, unique word count, average word length, and the top-N
most frequent words in the provided text.  No external dependencies required.

Usage
-----
    result = await plugin_manager.invoke(
        "TextStats",
        task="The quick brown fox jumps over the lazy dog",
        context={"top_n": 3},
    )
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict

from plugins.plugin_manager import Plugin


def _analyse(text: str, top_n: int = 5) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return "No words found in the provided text."

    total      = len(words)
    unique     = len(set(words))
    avg_len    = sum(len(w) for w in words) / total
    top_words  = Counter(words).most_common(top_n)

    lines = [
        f"Word count:      {total}",
        f"Unique words:    {unique}",
        f"Avg word length: {avg_len:.1f}",
        f"Top {top_n} words:   " + ", ".join(f"{w!r}({c})" for w, c in top_words),
    ]
    return "\n".join(lines)


async def _handler(task: str, context: Dict[str, Any]) -> str:
    top_n = int(context.get("top_n", 5))
    return _analyse(task, top_n=top_n)


def register() -> Plugin:
    return Plugin(
        name="TextStats",
        version="1.0.0",
        description=(
            "Analyses text and returns word count, unique words, "
            "average word length, and top-N most frequent words. "
            "Pass top_n in context to control how many top words are shown."
        ),
        handler=_handler,
        metadata={"category": "text", "requires_llm": False},
    )
