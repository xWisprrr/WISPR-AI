"""Long-term persistent memory backed by a JSON file."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _tokenize(text: str) -> List[str]:
    """Lowercase and split *text* into alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf_idf_score(query_tokens: List[str], doc_tokens: List[str], idf: Dict[str, float]) -> float:
    """Return a simple TF-IDF relevance score for a document given a query."""
    if not doc_tokens or not query_tokens:
        return 0.0
    total = len(doc_tokens)
    score = 0.0
    for token in query_tokens:
        tf = doc_tokens.count(token) / total
        score += tf * idf.get(token, 0.0)
    return score


class LongTermMemory:
    """Persists knowledge across sessions as JSON records.

    Internally uses an :class:`dict` keyed on the record's ``key`` field for
    O(1) lookups instead of an O(n) linear scan.

    Search uses TF-IDF scoring so results are ranked by relevance rather than
    returned in arbitrary insertion order.
    """

    def __init__(self, db_path: str = settings.long_term_db_path) -> None:
        self._path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._data: Dict[str, Dict[str, Any]] = self._load()

    # -- public API -----------------------------------------------------------

    def store(self, key: str, value: Any, tags: Optional[List[str]] = None) -> None:
        """Add or update a knowledge entry."""
        self._data[key] = {
            "key": key,
            "value": value,
            "tags": tags or [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def retrieve(self, key: str) -> Optional[Any]:
        entry = self._data.get(key)
        return entry["value"] if entry is not None else None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the *top_k* records most relevant to *query*.

        Uses TF-IDF ranking over key + value text so that records mentioning
        the query terms more frequently and distinctively rank higher.
        Falls back to substring match ordering when all TF-IDF scores are zero
        (e.g. very short queries with common tokens).
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        records = list(self._data.values())
        if not records:
            return []

        # Build a corpus of tokenised documents: key + stringified value
        docs: List[List[str]] = [
            _tokenize(f"{rec['key']} {rec['value']!s}") for rec in records
        ]
        num_docs = len(docs)

        # Compute IDF for each query token over the entire corpus
        idf: Dict[str, float] = {}
        for token in set(query_tokens):
            containing = sum(1 for doc in docs if token in doc)
            if containing:
                idf[token] = math.log((num_docs + 1) / (containing + 1)) + 1.0

        # Score every record
        scored = [
            (rec, _tf_idf_score(query_tokens, doc, idf))
            for rec, doc in zip(records, docs)
        ]

        # Filter out zero-score records; if all are zero fall back to substring match
        non_zero = [(rec, s) for rec, s in scored if s > 0.0]
        if not non_zero:
            # Substring fallback for very common terms
            q = query.lower()
            non_zero = [
                (rec, 1.0)
                for rec in records
                if q in rec["key"].lower() or q in str(rec["value"]).lower()
            ]

        non_zero.sort(key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in non_zero[:top_k]]

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def all_entries(self) -> List[Dict[str, Any]]:
        return list(self._data.values())

    # -- internals ------------------------------------------------------------

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self._path):
            try:
                with open(self._path, encoding="utf-8") as f:
                    records: List[Dict[str, Any]] = json.load(f)
                result: Dict[str, Dict[str, Any]] = {}
                for rec in records:
                    if "key" not in rec:
                        logger.warning("Skipping long-term memory record without 'key': %s", rec)
                        continue
                    result[rec["key"]] = rec
                return result
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load long-term memory: %s", exc)
        return {}

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(list(self._data.values()), f, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("Could not persist long-term memory: %s", exc)
