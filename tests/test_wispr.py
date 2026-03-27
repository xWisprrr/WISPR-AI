"""Tests for the WISPR AI OS core components (no LLM API calls needed)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    def test_settings_load(self):
        from config import get_settings
        s = get_settings()
        self.assertEqual(s.app_name, "WISPR AI OS")
        self.assertGreater(s.llm_max_tokens, 0)
        self.assertGreater(s.search_max_sources, 0)
        self.assertIsInstance(s.studio_supported_languages, list)

    def test_settings_singleton(self):
        from config import get_settings
        self.assertIs(get_settings(), get_settings())


# ── Short-Term Memory ─────────────────────────────────────────────────────────

class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        from memory.short_term import ShortTermMemory
        self.mem = ShortTermMemory(max_entries=5)

    def test_add_and_retrieve(self):
        self.mem.add("user", "Hello")
        self.mem.add("assistant", "Hi there")
        self.assertEqual(len(self.mem), 2)
        recent = self.mem.get_recent(10)
        self.assertEqual(recent[0].content, "Hello")
        self.assertEqual(recent[1].content, "Hi there")

    def test_max_entries_respected(self):
        for i in range(10):
            self.mem.add("user", f"msg {i}")
        self.assertEqual(len(self.mem), 5)

    def test_as_messages(self):
        self.mem.add("user", "What is AI?")
        msgs = self.mem.as_messages()
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "What is AI?")

    def test_clear(self):
        self.mem.add("user", "test")
        self.mem.clear()
        self.assertEqual(len(self.mem), 0)


# ── Long-Term Memory ──────────────────────────────────────────────────────────

class TestLongTermMemory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        from memory.long_term import LongTermMemory
        self.mem = LongTermMemory(db_path=os.path.join(self._tmp, "lt.json"))

    def test_store_and_retrieve(self):
        self.mem.store("capital_france", "Paris")
        self.assertEqual(self.mem.retrieve("capital_france"), "Paris")

    def test_overwrite(self):
        self.mem.store("key", "v1")
        self.mem.store("key", "v2")
        self.assertEqual(self.mem.retrieve("key"), "v2")
        self.assertEqual(len(self.mem.all_entries()), 1)

    def test_delete(self):
        self.mem.store("to_delete", "bye")
        self.assertTrue(self.mem.delete("to_delete"))
        self.assertIsNone(self.mem.retrieve("to_delete"))

    def test_search(self):
        self.mem.store("ml_paper", "Attention is all you need")
        self.mem.store("unrelated", "nothing here")
        results = self.mem.search("attention")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["key"], "ml_paper")

    def test_persistence(self):
        self.mem.store("persist_key", "persist_value")
        db_path = self.mem._path
        from memory.long_term import LongTermMemory
        mem2 = LongTermMemory(db_path=db_path)
        self.assertEqual(mem2.retrieve("persist_key"), "persist_value")


# ── Task Memory ───────────────────────────────────────────────────────────────

class TestTaskMemory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        from memory.task_memory import TaskMemory
        self.mem = TaskMemory(db_path=os.path.join(self._tmp, "tasks.json"))

    def test_create_and_complete_task(self):
        task_id = self.mem.new_task("Build a chatbot")
        self.assertIsNotNone(task_id)
        self.mem.add_step(task_id, "CoreAgent", "Plan", "Steps 1-3")
        self.mem.complete_task(task_id, "Chatbot built!")
        task = self.mem.get_task(task_id)
        self.assertEqual(task["status"], "completed")
        self.assertEqual(len(task["steps"]), 1)

    def test_fail_task(self):
        task_id = self.mem.new_task("Impossible task")
        self.mem.fail_task(task_id, "Out of memory")
        task = self.mem.get_task(task_id)
        self.assertEqual(task["status"], "failed")

    def test_recent_tasks(self):
        for i in range(5):
            self.mem.new_task(f"Task {i}")
        recent = self.mem.recent_tasks(n=3)
        self.assertEqual(len(recent), 3)


# ── Language Detection ────────────────────────────────────────────────────────

class TestLanguageDetection(unittest.TestCase):
    def setUp(self):
        from coding.engine import CodingEngine
        self.engine = CodingEngine.__new__(CodingEngine)

    def test_python(self):
        code = "def hello():\n    print('Hello, world!')\n"
        self.assertEqual(self.engine.detect_language(code), "python")

    def test_javascript(self):
        code = "const greet = () => console.log('hi');"
        self.assertEqual(self.engine.detect_language(code), "javascript")

    def test_go(self):
        code = "package main\nfunc main() { fmt.Println('hi') }"
        self.assertEqual(self.engine.detect_language(code), "go")

    def test_unknown(self):
        code = "?????"
        self.assertEqual(self.engine.detect_language(code), "unknown")


# ── MegaSearch deduplication ──────────────────────────────────────────────────

class TestMegaSearch(unittest.TestCase):
    def test_deduplication(self):
        from search.mega_search import MegaSearch
        results = [
            {"url": "https://example.com/", "title": "A"},
            {"url": "https://example.com", "title": "A duplicate"},
            {"url": "https://other.com", "title": "B"},
        ]
        deduped = MegaSearch._deduplicate(results)
        # Both "https://example.com/" and "https://example.com" strip to "https://example.com"
        self.assertEqual(len(deduped), 2)

    def test_empty(self):
        from search.mega_search import MegaSearch
        self.assertEqual(MegaSearch._deduplicate([]), [])


# ── AgentResult ───────────────────────────────────────────────────────────────

class TestAgentResult(unittest.TestCase):
    def test_ok_result(self):
        from agents.base_agent import AgentResult
        r = AgentResult(
            success=True, output="hello", agent_name="TestAgent", task_type="general"
        )
        self.assertTrue(r.success)
        self.assertIsNone(r.error)

    def test_error_result(self):
        from agents.base_agent import AgentResult
        r = AgentResult(
            success=False, output=None, agent_name="TestAgent",
            task_type="general", error="boom"
        )
        self.assertFalse(r.success)
        self.assertEqual(r.error, "boom")


# ── Complexity Classification ─────────────────────────────────────────────────

class TestComplexityClassification(unittest.TestCase):
    """Tests for OrchestratorAgent._classify_complexity and simple/complex routing."""

    def _make_orchestrator(self, llm_response: str):
        """Build an OrchestratorAgent whose LLM always returns *llm_response*."""
        from agents.orchestrator import OrchestratorAgent
        from memory.manager import MemoryManager

        memory = MemoryManager()
        orch = OrchestratorAgent.__new__(OrchestratorAgent)
        orch.memory = memory

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=llm_response)
        orch.llm = mock_llm
        return orch

    def test_classify_simple(self):
        orch = self._make_orchestrator("SIMPLE")
        result = asyncio.get_event_loop().run_until_complete(
            orch._classify_complexity("What is 2 + 2?")
        )
        self.assertEqual(result, "simple")

    def test_classify_complex(self):
        orch = self._make_orchestrator("COMPLEX")
        result = asyncio.get_event_loop().run_until_complete(
            orch._classify_complexity("Build a full-stack web app with authentication.")
        )
        self.assertEqual(result, "complex")

    def test_classify_defaults_to_complex_on_llm_error(self):
        from agents.orchestrator import OrchestratorAgent
        from memory.manager import MemoryManager

        orch = OrchestratorAgent.__new__(OrchestratorAgent)
        orch.memory = MemoryManager()

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        orch.llm = mock_llm

        result = asyncio.get_event_loop().run_until_complete(
            orch._classify_complexity("Hello!")
        )
        self.assertEqual(result, "complex")

    def test_simple_task_returns_empty_plan(self):
        """Simple tasks should bypass orchestration (plan == [])."""
        from agents.orchestrator import OrchestratorAgent
        from agents.base_agent import AgentResult
        from memory.manager import MemoryManager

        memory = MemoryManager()
        orch = OrchestratorAgent.__new__(OrchestratorAgent)
        orch.memory = memory

        # LLM for classify returns SIMPLE; CoreAgent returns a short answer
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value="SIMPLE")
        orch.llm = mock_llm
        orch.reasoning = MagicMock()

        mock_core = MagicMock()
        mock_core.name = "CoreAgent"
        mock_core.run = AsyncMock(
            return_value=AgentResult(
                success=True, output="Paris.", agent_name="CoreAgent", task_type="reasoning"
            )
        )
        orch._agents = {"CoreAgent": mock_core}

        result = asyncio.get_event_loop().run_until_complete(
            orch.run("What is the capital of France?")
        )
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["final_answer"], "Paris.")

    def test_simple_task_passes_concise_context(self):
        """CoreAgent must receive response_style=concise for simple tasks."""
        from agents.orchestrator import OrchestratorAgent
        from agents.base_agent import AgentResult
        from memory.manager import MemoryManager

        memory = MemoryManager()
        orch = OrchestratorAgent.__new__(OrchestratorAgent)
        orch.memory = memory

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value="SIMPLE")
        orch.llm = mock_llm
        orch.reasoning = MagicMock()

        captured_context: list = []

        async def capture_run(task, context=None):
            captured_context.append(context)
            return AgentResult(
                success=True, output="42.", agent_name="CoreAgent", task_type="reasoning"
            )

        mock_core = MagicMock()
        mock_core.name = "CoreAgent"
        mock_core.run = capture_run
        orch._agents = {"CoreAgent": mock_core}

        asyncio.get_event_loop().run_until_complete(
            orch.run("What is 6 times 7?")
        )
        self.assertTrue(len(captured_context) > 0)
        self.assertEqual(captured_context[0].get("response_style"), "concise")


# ── CoreAgent concise mode ────────────────────────────────────────────────────

class TestCoreAgentConciseMode(unittest.TestCase):
    """CoreAgent should use the concise system prompt when response_style=concise."""

    def _make_core_agent(self, llm_response: str):
        from agents.core_agent import CoreAgent
        from memory.manager import MemoryManager

        memory = MemoryManager()
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=llm_response)
        return CoreAgent(llm_router=mock_llm, memory=memory)

    def test_concise_mode_uses_concise_prompt(self):
        agent = self._make_core_agent("Yes.")
        asyncio.get_event_loop().run_until_complete(
            agent.run("Is the sky blue?", context={"response_style": "concise"})
        )
        # Verify the concise system prompt was used (check for its key instruction)
        call_args = agent.llm.complete.call_args
        messages = call_args[0][0]
        self.assertIn("concisely", messages[0]["content"].lower())
        self.assertIn("one or two sentences", messages[0]["content"].lower())

    def test_normal_mode_uses_default_prompt(self):
        agent = self._make_core_agent("The sky is blue because of Rayleigh scattering.")
        asyncio.get_event_loop().run_until_complete(
            agent.run("Why is the sky blue?")
        )
        call_args = agent.llm.complete.call_args
        messages = call_args[0][0]
        # Default prompt emphasises structured reasoning, not conciseness
        self.assertIn("reasoning", messages[0]["content"].lower())
        self.assertNotIn("one or two sentences", messages[0]["content"].lower())


class TestPluginManager(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def _write_plugin(self, filename: str, content: str) -> None:
        (Path(self._tmp) / filename).write_text(content)

    def test_no_plugins_dir(self):
        from plugins.plugin_manager import PluginManager
        pm = PluginManager(plugins_dir="/nonexistent/path/xyz")
        self.assertEqual(pm.list_plugins(), [])

    def test_plugin_discovery(self):
        self._write_plugin("hello_plugin.py", """\
from plugins.plugin_manager import Plugin

def register():
    async def handler(task, context):
        return f"Hello: {task}"
    return Plugin(name="HelloPlugin", version="1.0", description="Say hello", handler=handler)
""")
        from plugins.plugin_manager import PluginManager
        pm = PluginManager(plugins_dir=self._tmp)
        plugins = pm.list_plugins()
        names = [p["name"] for p in plugins]
        self.assertIn("HelloPlugin", names)

    def test_manual_register(self):
        from plugins.plugin_manager import Plugin, PluginManager

        async def my_handler(task, ctx):
            return "done"

        pm = PluginManager(plugins_dir=self._tmp)
        pm.register_plugin(Plugin("MyPlugin", "0.1", "Test plugin", my_handler))
        self.assertIsNotNone(pm.get_plugin("MyPlugin"))

    def test_invoke_plugin(self):
        from plugins.plugin_manager import Plugin, PluginManager

        async def my_handler(task, ctx):
            return f"result:{task}"

        pm = PluginManager(plugins_dir=self._tmp)
        pm.register_plugin(Plugin("InvokePlugin", "1.0", "Invoke test", my_handler))
        result = asyncio.get_event_loop().run_until_complete(
            pm.invoke("InvokePlugin", "test_task")
        )
        self.assertEqual(result, "result:test_task")

    def test_invoke_unknown_plugin(self):
        from plugins.plugin_manager import PluginManager
        pm = PluginManager(plugins_dir=self._tmp)
        with self.assertRaises(KeyError):
            asyncio.get_event_loop().run_until_complete(
                pm.invoke("NonExistent", "task")
            )


# ── Studio IDE ────────────────────────────────────────────────────────────────

class TestStudioIDE(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        from studio.ide import StudioIDE
        self.ide = StudioIDE(sandbox_dir=self._tmp)

    def test_unsupported_language(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.ide.execute("code", language="brainfuck")
        )
        self.assertFalse(result["success"])
        self.assertIn("Unsupported", result["stderr"])

    def test_deploy_github_steps(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.ide.deploy("/my/project", "github", {"repo": "me/repo"})
        )
        self.assertTrue(result["success"])
        self.assertIn("github", result["target"])
        self.assertTrue(len(result["steps"]) > 0)

    def test_deploy_unknown_target(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.ide.deploy("/my/project", "unknown_cloud")
        )
        self.assertFalse(result["success"])

    def test_execute_python(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.ide.execute('print("wispr")', language="python")
        )
        self.assertTrue(result["success"])
        self.assertIn("wispr", result["stdout"])


# ── FastAPI app smoke test ────────────────────────────────────────────────────

class TestFastAPIApp(unittest.TestCase):
    def test_app_creation(self):
        """Verify the FastAPI app object is created without errors."""
        # We need to avoid hitting real LLM/network during import
        from main import create_app
        app = create_app()
        self.assertIsNotNone(app)
        self.assertEqual(app.title, "WISPR AI OS")

    def test_routes_registered(self):
        from main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        # The system status endpoint should return 200 without LLM
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("agents", data)

    def test_agents_endpoint(self):
        from main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        response = client.get("/agents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        agent_names = [a["name"] for a in data["agents"]]
        for name in ("CoreAgent", "CoderAgent", "SearchAgent", "StudioAgent"):
            self.assertIn(name, agent_names)

    def test_memory_endpoint(self):
        from main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        response = client.get("/memory")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("short_term_entries", data)
        self.assertIn("long_term_entries", data)

    def test_memory_store_and_retrieve(self):
        from main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        store_resp = client.post(
            "/memory/store",
            json={"key": "test_key", "value": "test_value", "tags": ["test"]},
        )
        self.assertEqual(store_resp.status_code, 200)

        retrieve_resp = client.get("/memory/retrieve/test_key")
        self.assertEqual(retrieve_resp.status_code, 200)
        self.assertEqual(retrieve_resp.json()["value"], "test_value")

    def test_memory_retrieve_not_found(self):
        from main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        response = client.get("/memory/retrieve/nonexistent_key_xyz")
        self.assertEqual(response.status_code, 404)

    def test_plugins_endpoint(self):
        from main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        response = client.get("/plugins")
        self.assertEqual(response.status_code, 200)
        self.assertIn("plugins", response.json())

    def test_studio_execute_python(self):
        from main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/studio/execute",
            json={"code": 'print("hello wispr")', "language": "python", "timeout": 10},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("hello wispr", data["stdout"])

    def test_code_unsupported_action(self):
        from main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/code",
            json={"task": "do something", "action": "invalid_action"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
