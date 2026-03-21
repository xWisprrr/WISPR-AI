"""
Memory System — short-term (in-process) and long-term (file-based) stores.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from config import settings


# ── Short-term memory (in-process, capped deque) ──────────────────────────────

class ShortTermMemory:
    """Session-scoped key-value store with a fixed capacity."""

    def __init__(self, max_entries: int = settings.max_short_term_entries) -> None:
        self._max = max_entries
        self._store: deque[dict[str, Any]] = deque(maxlen=max_entries)

    def add(self, role: str, content: str, metadata: Optional[dict] = None) -> str:
        entry_id = str(uuid.uuid4())
        self._store.append(
            {
                "id": entry_id,
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
        )
        return entry_id

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        items = list(self._store)
        return items[-limit:]

    def as_messages(self, limit: int = 20) -> list[dict[str, str]]:
        """Return entries as OpenAI-style message dicts."""
        return [
            {"role": e["role"], "content": e["content"]}
            for e in self.get_history(limit)
        ]

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ── Long-term memory (JSON file-backed) ───────────────────────────────────────

class LongTermMemory:
    """Persistent file-backed memory that survives process restarts."""

    def __init__(self, memory_dir: Path = settings.memory_dir) -> None:
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._index: dict[str, dict[str, Any]] = self._load_index()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Long-term memory index corrupted — resetting.")
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    # ── Public API ────────────────────────────────────────────────────────────

    def store(self, key: str, value: Any, tags: Optional[list[str]] = None) -> None:
        entry = {
            "key": key,
            "value": value,
            "tags": tags or [],
            "timestamp": time.time(),
        }
        self._index[key] = entry
        self._save_index()
        logger.debug(f"[Memory/LT] stored key={key!r}")

    def retrieve(self, key: str) -> Optional[Any]:
        entry = self._index.get(key)
        return entry["value"] if entry else None

    def search_by_tag(self, tag: str) -> list[dict[str, Any]]:
        return [e for e in self._index.values() if tag in e.get("tags", [])]

    def all_keys(self) -> list[str]:
        return list(self._index.keys())

    def delete(self, key: str) -> bool:
        if key in self._index:
            del self._index[key]
            self._save_index()
            return True
        return False

    def __len__(self) -> int:
        return len(self._index)


# ── Task memory (transient per-task scratchpad) ───────────────────────────────

class TaskMemory:
    """Lightweight scratchpad for a single task execution."""

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []
        self._artifacts: dict[str, Any] = {}

    def record_step(self, agent: str, action: str, result: Any) -> None:
        self._steps.append(
            {
                "seq": len(self._steps) + 1,
                "agent": agent,
                "action": action,
                "result": result,
                "timestamp": time.time(),
            }
        )

    def set_artifact(self, name: str, value: Any) -> None:
        self._artifacts[name] = value

    def get_artifact(self, name: str) -> Optional[Any]:
        return self._artifacts.get(name)

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(self._steps)

    @property
    def artifacts(self) -> dict[str, Any]:
        return dict(self._artifacts)

    def summary(self) -> dict[str, Any]:
        return {
            "total_steps": len(self._steps),
            "agents_used": list({s["agent"] for s in self._steps}),
            "artifacts": list(self._artifacts.keys()),
        }
