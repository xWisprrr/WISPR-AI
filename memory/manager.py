"""Unified memory manager — single entry point for all memory subsystems."""

from __future__ import annotations

from memory.long_term import LongTermMemory
from memory.session import SessionManager
from memory.short_term import ShortTermMemory
from memory.task_memory import TaskMemory


class MemoryManager:
    """Aggregates short-term, long-term, task, and session memory in one place."""

    def __init__(self) -> None:
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.tasks = TaskMemory()
        self.sessions = SessionManager()
