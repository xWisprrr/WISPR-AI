"""Long-term persistent memory backed by a JSON file."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LongTermMemory:
    """Persists knowledge across sessions as JSON records."""

    def __init__(self, db_path: str = settings.long_term_db_path) -> None:
        self._path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._data: List[Dict[str, Any]] = self._load()

    # ── public API ────────────────────────────────────────────────────────

    def store(self, key: str, value: Any, tags: Optional[List[str]] = None) -> None:
        """Add or update a knowledge entry."""
        entry = {
            "key": key,
            "value": value,
            "tags": tags or [],
            "updated_at": datetime.utcnow().isoformat(),
        }
        # Overwrite existing key
        for i, rec in enumerate(self._data):
            if rec["key"] == key:
                self._data[i] = entry
                self._save()
                return
        self._data.append(entry)
        self._save()

    def retrieve(self, key: str) -> Optional[Any]:
        for rec in self._data:
            if rec["key"] == key:
                return rec["value"]
        return None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simple substring search over keys and string values."""
        q = query.lower()
        matches = []
        for rec in self._data:
            if q in rec["key"].lower() or q in str(rec["value"]).lower():
                matches.append(rec)
        return matches[:top_k]

    def delete(self, key: str) -> bool:
        before = len(self._data)
        self._data = [r for r in self._data if r["key"] != key]
        if len(self._data) < before:
            self._save()
            return True
        return False

    def all_entries(self) -> List[Dict[str, Any]]:
        return list(self._data)

    # ── internals ─────────────────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self._path):
            try:
                with open(self._path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load long-term memory: %s", exc)
        return []

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("Could not persist long-term memory: %s", exc)
