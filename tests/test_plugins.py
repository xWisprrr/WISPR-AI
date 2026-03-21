"""
Tests for the Plugin Manager.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from plugins.manager import PluginManager


class TestPluginManager:
    def test_discover_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(plugins_dir=Path(tmpdir))
            loaded = pm.discover()
            assert loaded == []
            assert pm.plugin_count == 0

    def test_discover_nonexistent_directory(self):
        pm = PluginManager(plugins_dir=Path("/nonexistent/path"))
        loaded = pm.discover()
        assert loaded == []

    def test_discover_valid_plugin(self):
        plugin_code = '''
from agents.base_agent import BaseAgent, AgentResult
from typing import Any, Optional

class Plugin(BaseAgent):
    name = "test_plugin"
    description = "A test plugin"

    async def run(self, task: str, context: Optional[dict] = None) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.name,
            success=True,
            output="test",
        )
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = Path(tmpdir) / "my_plugin.py"
            plugin_path.write_text(plugin_code)
            pm = PluginManager(plugins_dir=Path(tmpdir))
            loaded = pm.discover()
            assert "test_plugin" in loaded
            assert pm.plugin_count == 1

    def test_list_plugins(self):
        plugin_code = '''
from agents.base_agent import BaseAgent, AgentResult
from typing import Any, Optional

class Plugin(BaseAgent):
    name = "listed_plugin"
    description = "listed"

    async def run(self, task: str, context: Optional[dict] = None) -> AgentResult:
        return AgentResult(agent_id=self.agent_id, agent_name=self.name, success=True, output="")
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "p.py").write_text(plugin_code)
            pm = PluginManager(plugins_dir=Path(tmpdir))
            pm.discover()
            listing = pm.list_plugins()
            assert any(p["name"] == "listed_plugin" for p in listing)

    def test_get_missing_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(plugins_dir=Path(tmpdir))
            assert pm.get("nothing") is None

    def test_instantiate_plugin(self):
        plugin_code = '''
from agents.base_agent import BaseAgent, AgentResult
from typing import Any, Optional

class Plugin(BaseAgent):
    name = "inst_plugin"
    description = "instantiate test"

    async def run(self, task: str, context: Optional[dict] = None) -> AgentResult:
        return AgentResult(agent_id=self.agent_id, agent_name=self.name, success=True, output="")
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "inst.py").write_text(plugin_code)
            pm = PluginManager(plugins_dir=Path(tmpdir))
            pm.discover()
            instance = pm.instantiate("inst_plugin")
            assert instance is not None
            assert instance.name == "inst_plugin"

    def test_skips_dunder_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "__init__.py").write_text("")
            (Path(tmpdir) / "_private.py").write_text("")
            pm = PluginManager(plugins_dir=Path(tmpdir))
            loaded = pm.discover()
            assert loaded == []
