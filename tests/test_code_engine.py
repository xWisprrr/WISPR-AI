"""Tests for the WISPR AI Code Engine subsystem.

Covers:
  • Mode router behaviour (Ask cannot write files)
  • Session persistence and resume
  • File tool auditing
  • Orchestrator task breakdown (unit-level)
  • Deployment engine provider list
  • File tools safety (path traversal / denied paths)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies so tests run without a full install.
# Only stub modules that are NOT already loaded (avoids contaminating
# sys.modules for other test files that import the real modules).
# ---------------------------------------------------------------------------
_llm_stub = types.ModuleType("litellm")
sys.modules.setdefault("litellm", _llm_stub)

# Only inject the llm.router stub when the real module isn't already loaded.
# This prevents this file from replacing a real loaded module when tests are
# run together with test_wispr.py (which imports the actual llm.router).
if "llm.router" not in sys.modules:
    _llm_router_stub = types.ModuleType("llm.router")
    _llm_router_stub.LLMRouter = MagicMock  # type: ignore[attr-defined]
    _llm_router_stub.TaskType = MagicMock()  # type: ignore[attr-defined]
    _llm_router_stub.get_router = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    sys.modules.setdefault("llm", types.ModuleType("llm"))
    sys.modules["llm.router"] = _llm_router_stub


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run(coro):
    return asyncio.run(coro)


def _mock_llm(response: str = "OK") -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


# ══════════════════════════════════════════════════════════════════════════════
# Session manager
# ══════════════════════════════════════════════════════════════════════════════

class TestCodeSessionManager(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        from coding.session.manager import CodeSessionManager
        self.mgr = CodeSessionManager(store_path=os.path.join(self._tmp, "sessions.jsonl"))

    def test_create_and_retrieve(self):
        s = self.mgr.create(mode="code", workspace="/tmp/proj")
        retrieved = self.mgr.get(s.session_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.mode, "code")
        self.assertEqual(retrieved.workspace, "/tmp/proj")

    def test_persistence_across_instances(self):
        from coding.session.manager import CodeSessionManager
        s = self.mgr.create(mode="debug")
        s.add_message("user", "hello")
        s.add_message("assistant", "hi")
        self.mgr.save(s)

        mgr2 = CodeSessionManager(store_path=str(self.mgr._path))
        s2 = mgr2.get(s.session_id)
        self.assertIsNotNone(s2)
        self.assertEqual(len(s2.history), 2)
        self.assertEqual(s2.history[0]["content"], "hello")

    def test_resume_keeps_mode(self):
        s = self.mgr.create(mode="architect")
        s.add_message("user", "design my app")
        self.mgr.save(s)

        from coding.session.manager import CodeSessionManager
        mgr2 = CodeSessionManager(store_path=str(self.mgr._path))
        resumed = mgr2.get(s.session_id)
        self.assertEqual(resumed.mode, "architect")

    def test_delete(self):
        s = self.mgr.create()
        self.assertTrue(self.mgr.delete(s.session_id))
        self.assertIsNone(self.mgr.get(s.session_id))

    def test_list_all(self):
        self.mgr.create(mode="ask")
        self.mgr.create(mode="code")
        sessions = self.mgr.list_all()
        self.assertEqual(len(sessions), 2)
        self.assertIn("session_id", sessions[0])

    def test_set_mode_valid(self):
        s = self.mgr.create(mode="ask")
        s.set_mode("orchestrator")
        self.assertEqual(s.mode, "orchestrator")

    def test_set_mode_invalid(self):
        s = self.mgr.create()
        with self.assertRaises(ValueError):
            s.set_mode("invalid_mode")

    def test_variables(self):
        s = self.mgr.create()
        s.set_var("project_name", "MyApp")
        self.assertEqual(s.get_var("project_name"), "MyApp")
        self.assertIsNone(s.get_var("missing"))


# ══════════════════════════════════════════════════════════════════════════════
# Audit log
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditLog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        from coding.tools.audit_log import AuditLog
        self.log = AuditLog(log_path=os.path.join(self._tmp, "audit.jsonl"))

    def test_record_and_recent(self):
        self.log.record(operation="create", path="/tmp/test.py", session_id="s1", agent="CodeAgent")
        records = self.log.recent(n=10)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["op"], "create")
        self.assertEqual(records[0]["path"], "/tmp/test.py")
        self.assertEqual(records[0]["session_id"], "s1")

    def test_multiple_records(self):
        for i in range(5):
            self.log.record(operation="read", path=f"/tmp/file{i}.py")
        self.assertEqual(len(self.log.recent(n=10)), 5)

    def test_recent_limit(self):
        for i in range(10):
            self.log.record(operation="write", path=f"/tmp/f{i}.py")
        self.assertEqual(len(self.log.recent(n=3)), 3)

    def test_clear(self):
        self.log.record(operation="delete", path="/tmp/x.py")
        self.log.clear()
        self.assertEqual(len(self.log.recent()), 0)

    def test_bytes_recorded(self):
        self.log.record(operation="write", path="/tmp/a.py", bytes_changed=1024)
        records = self.log.recent()
        self.assertEqual(records[0]["bytes"], 1024)


# ══════════════════════════════════════════════════════════════════════════════
# File tools — safety
# ══════════════════════════════════════════════════════════════════════════════

class TestFileToolsSafety(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        from coding.tools.audit_log import AuditLog
        from coding.tools.file_tools import FileTools
        audit = AuditLog(log_path=os.path.join(self._tmp, "audit.jsonl"))
        self.ft = FileTools(audit=audit, session_id="test-session")

    def test_read_nonexistent(self):
        result = _run(self.ft.read_file("/nonexistent/path/file.txt"))
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_write_and_read(self):
        path = os.path.join(self._tmp, "hello.txt")
        result = _run(self.ft.write_file(path, "Hello, world!"))
        self.assertTrue(result["success"])
        self.assertEqual(result["operation"], "create")

        read_result = _run(self.ft.read_file(path))
        self.assertTrue(read_result["success"])
        self.assertEqual(read_result["content"], "Hello, world!")

    def test_create_fails_if_exists(self):
        path = os.path.join(self._tmp, "exists.txt")
        _run(self.ft.write_file(path, "initial"))
        result = _run(self.ft.create_file(path, "new content"))
        self.assertFalse(result["success"])
        self.assertIn("already exists", result["error"])

    def test_edit_file(self):
        path = os.path.join(self._tmp, "edit_me.py")
        _run(self.ft.write_file(path, "x = 1\ny = 2\n"))
        result = _run(self.ft.edit_file(path, "x = 1", "x = 42"))
        self.assertTrue(result["success"])

        read = _run(self.ft.read_file(path))
        self.assertIn("x = 42", read["content"])

    def test_edit_file_old_str_not_found(self):
        path = os.path.join(self._tmp, "no_match.py")
        _run(self.ft.write_file(path, "hello = 1\n"))
        result = _run(self.ft.edit_file(path, "NOTFOUND", "replacement"))
        self.assertFalse(result["success"])

    def test_delete_file(self):
        path = os.path.join(self._tmp, "to_delete.txt")
        _run(self.ft.write_file(path, "bye"))
        result = _run(self.ft.delete_file(path))
        self.assertTrue(result["success"])
        self.assertFalse(Path(path).exists())

    def test_list_dir(self):
        # Write a couple of files
        _run(self.ft.write_file(os.path.join(self._tmp, "a.py"), ""))
        _run(self.ft.write_file(os.path.join(self._tmp, "b.py"), ""))
        result = _run(self.ft.list_dir(self._tmp))
        self.assertTrue(result["success"])
        names = [e["name"] for e in result["entries"]]
        self.assertIn("a.py", names)
        self.assertIn("b.py", names)

    def test_search_files(self):
        _run(self.ft.write_file(os.path.join(self._tmp, "search_me.py"), "def hello(): pass\n"))
        _run(self.ft.write_file(os.path.join(self._tmp, "other.py"), "x = 1\n"))
        result = _run(self.ft.search_files(self._tmp, "def hello"))
        self.assertTrue(result["success"])
        self.assertGreater(result["count"], 0)
        self.assertIn("search_me.py", result["matches"][0]["file"])

    def test_move_file(self):
        src = os.path.join(self._tmp, "src.txt")
        dst = os.path.join(self._tmp, "dst.txt")
        _run(self.ft.write_file(src, "move me"))
        result = _run(self.ft.move_file(src, dst))
        self.assertTrue(result["success"])
        self.assertFalse(Path(src).exists())
        self.assertTrue(Path(dst).exists())

    def test_mkdir(self):
        new_dir = os.path.join(self._tmp, "new", "nested", "dir")
        result = _run(self.ft.mkdir(new_dir))
        self.assertTrue(result["success"])
        self.assertTrue(Path(new_dir).is_dir())

    def test_denied_path_blocked(self):
        """Writing to ~/.ssh should be blocked by the denylist."""
        ssh_path = str(Path.home() / ".ssh" / "wispr_test_should_not_exist")
        result = _run(self.ft.write_file(ssh_path, "secret"))
        self.assertFalse(result["success"])
        self.assertIn("protected", result["error"].lower())

    def test_audit_log_populated_after_write(self):
        from coding.tools.audit_log import AuditLog
        tmp2 = tempfile.mkdtemp()
        audit = AuditLog(log_path=os.path.join(tmp2, "a.jsonl"))
        from coding.tools.file_tools import FileTools
        ft = FileTools(audit=audit, session_id="audit-test")
        path = os.path.join(tmp2, "audited.txt")
        _run(ft.write_file(path, "audited content"))
        records = audit.recent()
        self.assertEqual(len(records), 1)
        self.assertIn(records[0]["op"], ("create", "overwrite"))


# ══════════════════════════════════════════════════════════════════════════════
# Ask agent — must NOT produce file write actions
# ══════════════════════════════════════════════════════════════════════════════

class TestAskAgentNoFileWrites(unittest.TestCase):
    def test_ask_agent_no_files_changed(self):
        from coding.agents.ask import AskAgent
        agent = AskAgent(llm_router=_mock_llm("Here is your answer."))
        result = _run(agent.run("What is a decorator in Python?"))
        self.assertTrue(result["success"])
        self.assertEqual(result["files_changed"], [])
        self.assertEqual(result["mode"], "ask")

    def test_ask_agent_cannot_invoke_write_ops(self):
        """Verify AskAgent has no reference to FileTools write operations."""
        import inspect
        from coding.agents import ask
        source = inspect.getsource(ask)
        # The Ask agent module should not import or call write_file / create_file
        self.assertNotIn("write_file", source)
        self.assertNotIn("create_file", source)
        self.assertNotIn("delete_file", source)
        self.assertNotIn("edit_file", source)

    def test_ask_agent_files_changed_always_empty(self):
        """Ask agent must return an empty files_changed list, even if prompted to write."""
        from coding.agents.ask import AskAgent
        agent = AskAgent(llm_router=_mock_llm("I cannot write files, but here is a code example: ```python\nx=1\n```"))
        result = _run(agent.run("Please write a file to /tmp/evil.txt with content 'hacked'"))
        self.assertEqual(result["files_changed"], [], "AskAgent must never report file writes")


# ══════════════════════════════════════════════════════════════════════════════
# Code agent — file writing
# ══════════════════════════════════════════════════════════════════════════════

class TestCodeAgent(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        from coding.tools.audit_log import AuditLog
        self.audit = AuditLog(log_path=os.path.join(self._tmp, "audit.jsonl"))

    def _make_agent(self, llm_response: str):
        from coding.agents.code import CodeAgent
        return CodeAgent(
            llm_router=_mock_llm(llm_response),
            audit=self.audit,
            workspace=self._tmp,
            session_id="code-test",
        )

    def test_code_agent_no_actions(self):
        """When LLM returns no <action> blocks, no files are changed."""
        agent = self._make_agent("Here is your code:\n```python\nprint('hi')\n```\n")
        result = _run(agent.run("Write a hello world", context={"workspace": self._tmp}))
        self.assertTrue(result["success"])
        self.assertEqual(result["files_changed"], [])

    def test_code_agent_write_action(self):
        """When LLM returns a valid <action> block, the file is created."""
        target = os.path.join(self._tmp, "hello.py")
        action_json = json.dumps({"op": "write_file", "path": target, "content": "print('hello')\n"})
        llm_response = f"<action>{action_json}</action>\nDone."
        agent = self._make_agent(llm_response)
        result = _run(agent.run("Write hello.py", context={"workspace": self._tmp}))
        self.assertTrue(result["success"])
        self.assertIn(target, result["files_changed"])
        self.assertTrue(Path(target).exists())
        self.assertEqual(Path(target).read_text(), "print('hello')\n")

    def test_code_agent_edit_action(self):
        """edit_file action replaces text in an existing file."""
        target = os.path.join(self._tmp, "edit_target.py")
        Path(target).write_text("x = 1\n")
        action_json = json.dumps({"op": "edit_file", "path": target, "old_str": "x = 1", "new_str": "x = 99"})
        llm_response = f"<action>{action_json}</action>\nDone."
        agent = self._make_agent(llm_response)
        result = _run(agent.run("Change x", context={"workspace": self._tmp}))
        self.assertTrue(result["success"])
        self.assertIn("x = 99", Path(target).read_text())

    def test_code_mode_label(self):
        agent = self._make_agent("no actions")
        result = _run(agent.run("task"))
        self.assertEqual(result["mode"], "code")


# ══════════════════════════════════════════════════════════════════════════════
# Debug agent
# ══════════════════════════════════════════════════════════════════════════════

class TestDebugAgent(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        from coding.tools.audit_log import AuditLog
        self.audit = AuditLog(log_path=os.path.join(self._tmp, "audit.jsonl"))

    def test_debug_agent_applies_fix(self):
        """DebugAgent should apply an edit_file action from the LLM response."""
        from coding.agents.debug import DebugAgent
        target = os.path.join(self._tmp, "buggy.py")
        Path(target).write_text("x = 1 / 0\n")
        action_json = json.dumps({
            "op": "edit_file",
            "path": target,
            "old_str": "x = 1 / 0",
            "new_str": "x = 1",
        })
        llm_response = f"The bug is a ZeroDivisionError.\n<action>{action_json}</action>\nFixed."
        agent = DebugAgent(
            llm_router=_mock_llm(llm_response),
            audit=self.audit,
            workspace=self._tmp,
            session_id="debug-test",
        )
        result = _run(agent.run(
            "Fix the ZeroDivisionError",
            code_snippet="x = 1 / 0",
            error_output="ZeroDivisionError: division by zero",
        ))
        self.assertTrue(result["success"])
        self.assertIn(target, result["files_changed"])
        self.assertIn("x = 1", Path(target).read_text())

    def test_debug_mode_label(self):
        from coding.agents.debug import DebugAgent
        agent = DebugAgent(llm_router=_mock_llm("analysis"), audit=self.audit)
        result = _run(agent.run("diagnose this"))
        self.assertEqual(result["mode"], "debug")


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — task decomposition
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorCodeAgent(unittest.TestCase):
    def _make_orchestrator(self, plan_response: str, task_response: str = "Done."):
        from coding.agents.orchestrator import OrchestratorCodeAgent
        from coding.tools.audit_log import AuditLog

        tmp = tempfile.mkdtemp()
        audit = AuditLog(log_path=os.path.join(tmp, "audit.jsonl"))

        call_count = [0]

        async def _complete(messages, task_type=None):
            call_count[0] += 1
            # First call = plan, subsequent = task execution + synthesis
            if call_count[0] == 1:
                return plan_response
            return task_response

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=_complete)

        return OrchestratorCodeAgent(llm_router=llm, audit=audit)

    def test_plan_decomposition_two_tasks(self):
        plan = json.dumps({
            "tasks": [
                {"id": 1, "mode": "architect", "description": "Design the system", "depends_on": []},
                {"id": 2, "mode": "code", "description": "Implement it", "depends_on": [1]},
            ]
        })
        orch = self._make_orchestrator(plan_response=plan)
        result = _run(orch.run("Build a REST API", parallel=False))
        self.assertTrue(result["success"])
        self.assertEqual(len(result["plan"]), 2)
        self.assertEqual(result["plan"][0]["mode"], "architect")
        self.assertEqual(result["plan"][1]["mode"], "code")

    def test_orchestrator_mode_label(self):
        plan = json.dumps({"tasks": [
            {"id": 1, "mode": "ask", "description": "quick question", "depends_on": []}
        ]})
        orch = self._make_orchestrator(plan_response=plan)
        result = _run(orch.run("Do something"))
        self.assertEqual(result["mode"], "orchestrator")

    def test_plan_failure_handled_gracefully(self):
        """If plan JSON is malformed, orchestrator returns success=False gracefully."""
        from coding.agents.orchestrator import OrchestratorCodeAgent
        from coding.tools.audit_log import AuditLog
        tmp = tempfile.mkdtemp()
        audit = AuditLog(log_path=os.path.join(tmp, "audit.jsonl"))
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="not json at all")
        orch = OrchestratorCodeAgent(llm_router=llm, audit=audit)
        result = _run(orch.run("Impossible"))
        self.assertFalse(result["success"])
        self.assertIn("error", result)


# ══════════════════════════════════════════════════════════════════════════════
# Chat router — mode dispatch
# ══════════════════════════════════════════════════════════════════════════════

class TestChatRouter(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        from coding.session.manager import CodeSessionManager
        from coding.tools.audit_log import AuditLog
        from coding.chat.router import ChatRouter
        self.sessions = CodeSessionManager(store_path=os.path.join(self._tmp, "s.jsonl"))
        self.audit = AuditLog(log_path=os.path.join(self._tmp, "a.jsonl"))
        self.router = ChatRouter(
            llm_router=_mock_llm("response text"),
            session_manager=self.sessions,
            audit=self.audit,
        )

    def test_creates_session_if_none(self):
        result = _run(self.router.chat("hello", mode="ask"))
        self.assertIn("session_id", result)
        self.assertIsNotNone(result["session_id"])

    def test_resumes_existing_session(self):
        session = self.sessions.create(mode="code", workspace="/tmp/ws")
        result = _run(self.router.chat("write code", session_id=session.session_id))
        self.assertEqual(result["session_id"], session.session_id)
        self.assertEqual(result["mode"], "code")

    def test_mode_override_for_turn(self):
        """Passing mode= overrides the session mode for that turn."""
        session = self.sessions.create(mode="ask")
        result = _run(self.router.chat("hello", session_id=session.session_id, mode="debug"))
        self.assertEqual(result["mode"], "debug")

    def test_history_persisted(self):
        result = _run(self.router.chat("first message", mode="ask"))
        sid = result["session_id"]
        result2 = _run(self.router.chat("second message", session_id=sid))
        session = self.sessions.get(sid)
        self.assertGreaterEqual(len(session.history), 2)

    def test_unknown_mode_returns_error(self):
        session = self.sessions.create(mode="ask")
        # Force unknown mode by directly setting it
        session.mode = "bogus"
        self.sessions.save(session)
        result = _run(self.router.chat("hello", session_id=session.session_id))
        self.assertFalse(result["success"])


# ══════════════════════════════════════════════════════════════════════════════
# Deployment engine
# ══════════════════════════════════════════════════════════════════════════════

class TestDeploymentEngine(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def _make_engine(self, llm_response: str = "# generated config\n"):
        from coding.deployment import DeploymentEngine
        return DeploymentEngine(llm_router=_mock_llm(llm_response))

    def test_unsupported_provider(self):
        engine = self._make_engine()
        result = _run(engine.deploy("heroku", project_path=self._tmp))
        self.assertFalse(result.success)
        self.assertIn("Unsupported", result.message)

    def test_vercel_generates_config(self):
        engine = self._make_engine()
        result = _run(engine.deploy("vercel", project_path=self._tmp))
        self.assertTrue(result.success)
        vercel_json = Path(self._tmp) / "vercel.json"
        self.assertTrue(vercel_json.exists())
        data = json.loads(vercel_json.read_text())
        self.assertIn("version", data)

    def test_netlify_generates_toml(self):
        engine = self._make_engine()
        result = _run(engine.deploy("netlify", project_path=self._tmp))
        self.assertTrue(result.success)
        toml_path = Path(self._tmp) / "netlify.toml"
        self.assertTrue(toml_path.exists())

    def test_railway_generates_toml(self):
        engine = self._make_engine()
        result = _run(engine.deploy("railway", project_path=self._tmp))
        self.assertTrue(result.success)
        railway_toml = Path(self._tmp) / "railway.toml"
        self.assertTrue(railway_toml.exists())

    def test_docker_generates_dockerfile(self):
        engine = self._make_engine()
        result = _run(engine.deploy("docker", project_path=self._tmp))
        self.assertTrue(result.success)
        self.assertTrue((Path(self._tmp) / "Dockerfile").exists())
        self.assertTrue((Path(self._tmp) / "docker-compose.yml").exists())

    def test_supported_providers_list(self):
        from coding.deployment import SUPPORTED_PROVIDERS
        self.assertIn("github", SUPPORTED_PROVIDERS)
        self.assertIn("vercel", SUPPORTED_PROVIDERS)
        self.assertIn("netlify", SUPPORTED_PROVIDERS)
        self.assertIn("railway", SUPPORTED_PROVIDERS)
        self.assertIn("docker", SUPPORTED_PROVIDERS)

    def test_github_generates_workflow(self):
        engine = self._make_engine("name: CI\non: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n")
        result = _run(engine.deploy("github", project_path=self._tmp))
        self.assertTrue(result.success)
        workflow = Path(self._tmp) / ".github" / "workflows" / "ci.yml"
        self.assertTrue(workflow.exists())


# ══════════════════════════════════════════════════════════════════════════════
# Code Engine facade — smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestCodeEngineFacade(unittest.TestCase):
    def test_detect_language_still_works(self):
        """Ensure the low-level CodingEngine API is still accessible via CodeEngine."""
        from coding.engine import CodeEngine
        engine = CodeEngine.__new__(CodeEngine)
        from coding.engine import CodingEngine
        engine._low_level = CodingEngine.__new__(CodingEngine)
        lang = engine.detect_language("def foo(): pass")
        self.assertEqual(lang, "python")

    def test_modes_are_five(self):
        from coding.session.manager import VALID_MODES
        self.assertEqual(len(VALID_MODES), 5)
        for mode in ("ask", "architect", "code", "debug", "orchestrator"):
            self.assertIn(mode, VALID_MODES)


if __name__ == "__main__":
    unittest.main()
