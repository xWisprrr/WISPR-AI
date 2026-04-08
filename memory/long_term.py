"""Long-term persistent memory backed by a JSON file."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LongTermMemory:
    """Persists knowledge across sessions as JSON records.

    Internally uses an :class:`dict` keyed on the record's ``key`` field for
    O(1) lookups instead of an O(n) linear scan.
    """

    def __init__(self, db_path: str = settings.long_term_db_path) -> None:
        self._path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._data: Dict[str, Dict[str, Any]] = self._load()

    # ── public API ────────────────────────────────────────────────────────

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
        """Simple substring search over keys and string values."""
        q = query.lower()
        matches = []
        for rec in self._data.values():
            if q in rec["key"].lower() or q in str(rec["value"]).lower():
                matches.append(rec)
        return matches[:top_k]

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def all_entries(self) -> List[Dict[str, Any]]:
        return list(self._data.values())

    # ── internals ─────────────────────────────────────────────────────────

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
