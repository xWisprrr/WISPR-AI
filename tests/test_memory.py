"""
Tests for the Memory System.
"""
from __future__ import annotations

import time
import tempfile
from pathlib import Path

import pytest

from memory.store import LongTermMemory, ShortTermMemory, TaskMemory


# ── ShortTermMemory ────────────────────────────────────────────────────────────

class TestShortTermMemory:
    def test_add_and_retrieve(self):
        mem = ShortTermMemory(max_entries=10)
        mem.add("user", "Hello!")
        mem.add("assistant", "Hi there!")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"

    def test_max_entries_respected(self):
        mem = ShortTermMemory(max_entries=3)
        for i in range(5):
            mem.add("user", f"msg {i}")
        assert len(mem) == 3

    def test_as_messages(self):
        mem = ShortTermMemory()
        mem.add("user", "Question")
        mem.add("assistant", "Answer")
        msgs = mem.as_messages()
        assert len(msgs) == 2
        assert all("role" in m and "content" in m for m in msgs)

    def test_clear(self):
        mem = ShortTermMemory()
        mem.add("user", "Hello")
        mem.clear()
        assert len(mem) == 0

    def test_limit_parameter(self):
        mem = ShortTermMemory()
        for i in range(10):
            mem.add("user", f"msg {i}")
        assert len(mem.get_history(limit=5)) == 5


# ── LongTermMemory ────────────────────────────────────────────────────────────

class TestLongTermMemory:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._mem = LongTermMemory(memory_dir=Path(self._tmpdir))

    def test_store_and_retrieve(self):
        self._mem.store("fact1", "The sky is blue")
        assert self._mem.retrieve("fact1") == "The sky is blue"

    def test_missing_key_returns_none(self):
        assert self._mem.retrieve("nonexistent") is None

    def test_tags_and_search(self):
        self._mem.store("doc1", "content A", tags=["ai", "research"])
        self._mem.store("doc2", "content B", tags=["coding"])
        ai_docs = self._mem.search_by_tag("ai")
        assert len(ai_docs) == 1
        assert ai_docs[0]["key"] == "doc1"

    def test_delete(self):
        self._mem.store("temp", "value")
        assert self._mem.delete("temp") is True
        assert self._mem.retrieve("temp") is None

    def test_delete_missing(self):
        assert self._mem.delete("does-not-exist") is False

    def test_all_keys(self):
        self._mem.store("k1", "v1")
        self._mem.store("k2", "v2")
        keys = self._mem.all_keys()
        assert "k1" in keys and "k2" in keys

    def test_persistence(self):
        self._mem.store("persist", 42)
        # Create a new instance pointing to the same directory
        mem2 = LongTermMemory(memory_dir=Path(self._tmpdir))
        assert mem2.retrieve("persist") == 42


# ── TaskMemory ────────────────────────────────────────────────────────────────

class TestTaskMemory:
    def test_record_step(self):
        tm = TaskMemory()
        tm.record_step("core", "generate", "output text")
        assert len(tm.steps) == 1
        assert tm.steps[0]["agent"] == "core"

    def test_artifacts(self):
        tm = TaskMemory()
        tm.set_artifact("code", "print('hello')")
        assert tm.get_artifact("code") == "print('hello')"
        assert tm.get_artifact("missing") is None

    def test_summary(self):
        tm = TaskMemory()
        tm.record_step("core", "a", "x")
        tm.record_step("coder", "b", "y")
        tm.set_artifact("result", "done")
        summary = tm.summary()
        assert summary["total_steps"] == 2
        assert set(summary["agents_used"]) == {"core", "coder"}
        assert "result" in summary["artifacts"]
