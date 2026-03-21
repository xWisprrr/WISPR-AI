"""Short-term (in-process) session memory."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional

from config import get_settings

settings = get_settings()


@dataclass
class MemoryEntry:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict = field(default_factory=dict)


class ShortTermMemory:
    """Bounded in-memory ring-buffer for the active session."""

    def __init__(self, max_entries: int = settings.short_term_max_entries) -> None:
        self._entries: Deque[MemoryEntry] = deque(maxlen=max_entries)

    def add(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        self._entries.append(
            MemoryEntry(role=role, content=content, metadata=metadata or {})
        )

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        return list(self._entries)[-n:]

    def as_messages(self, n: int = 10) -> List[Dict[str, str]]:
        """Return the last *n* entries as LLM-compatible message dicts."""
        return [
            {"role": e.role, "content": e.content} for e in self.get_recent(n)
        ]

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
