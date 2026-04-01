"""Tests for the Xencode spec-driven project compiler pipeline.

Covers:
  • Schema validation: XencodeSpec, XencodePlan, XencodeManifest
  • Manifest serialisation / deserialisation (JSON round-trip)
  • SpecParser heuristic fallback (no LLM) — language detection from prompt
  • Validator static checks (missing entry point, missing README, syntax errors)
  • Finalizer ZIP creation from a temp workspace (no build command path)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies so tests run without a full install.
# ---------------------------------------------------------------------------
_litellm_stub = types.ModuleType("litellm")
sys.modules.setdefault("litellm", _litellm_stub)

_llm_stub = types.ModuleType("llm")
sys.modules.setdefault("llm", _llm_stub)

_llm_router_stub = types.ModuleType("llm.router")
_llm_router_stub.LLMRouter = MagicMock  # type: ignore[attr-defined]
_llm_router_stub.TaskType = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("llm.router", _llm_router_stub)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run(coro):
    return asyncio.run(coro)


def _mock_llm(response: str = "{}") -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


# ══════════════════════════════════════════════════════════════════════════════
# Schema validation
# ══════════════════════════════════════════════════════════════════════════════

class TestXencodeSchemas(unittest.TestCase):
    def test_spec_defaults(self):
        from coding.xencode.schemas import XencodeSpec
        spec = XencodeSpec(project_name="hello", description="A test project")
        self.assertEqual(spec.language, "python")
        self.assertEqual(spec.entry_point, "main.py")
        self.assertIsNone(spec.build_command)
        self.assertIsNone(spec.test_command)
        self.assertEqual(spec.requirements, [])
        self.assertEqual(spec.extra, {})

    def test_spec_custom_fields(self):
        from coding.xencode.schemas import XencodeSpec
        spec = XencodeSpec(
            project_name="my-api",
            description="REST API",
            language="go",
            requirements=["gin"],
            entry_point="main.go",
            build_command="go build ./...",
            test_command="go test ./...",
            extra={"author": "alice"},
        )
        self.assertEqual(spec.language, "go")
        self.assertEqual(spec.requirements, ["gin"])
        self.assertEqual(spec.entry_point, "main.go")
        self.assertEqual(spec.extra["author"], "alice")

    def test_planned_file(self):
        from coding.xencode.schemas import PlannedFile
        pf = PlannedFile(path="src/main.py", content="print('hi')")
        self.assertEqual(pf.path, "src/main.py")
        self.assertEqual(pf.description, "")

    def test_xencode_plan(self):
        from coding.xencode.schemas import XencodeSpec, XencodePlan, PlannedFile
        spec = XencodeSpec(project_name="demo", description="demo")
        plan = XencodePlan(
            spec=spec,
            files=[PlannedFile(path="main.py", content="# main")],
            workspace="/tmp/demo",
            readme_included=True,
        )
        self.assertEqual(len(plan.files), 1)
        self.assertTrue(plan.readme_included)

    def test_manifest_defaults(self):
        from coding.xencode.schemas import XencodeManifest
        m = XencodeManifest(
            project_name="proj",
            language="python",
            workspace="/ws/proj",
        )
        self.assertFalse(m.build_success)
        self.assertEqual(m.repair_attempts, 0)
        self.assertIsNone(m.zip_path)
        self.assertIsNotNone(m.created_at)

    def test_compile_request(self):
        from coding.xencode.schemas import XencodeCompileRequest
        req = XencodeCompileRequest(prompt="Build a CLI tool", project_name="cli-tool")
        self.assertEqual(req.language, "python")
        self.assertIsNone(req.workspace)
        self.assertIsNone(req.build_command)

    def test_compile_response(self):
        from coding.xencode.schemas import XencodeCompileResponse
        resp = XencodeCompileResponse(
            success=True,
            project_name="cli-tool",
            workspace="/ws/cli-tool",
        )
        self.assertTrue(resp.success)
        self.assertIsNone(resp.error)


# ══════════════════════════════════════════════════════════════════════════════
# Manifest serialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestManifestSerialization(unittest.TestCase):
    def test_json_round_trip(self):
        from coding.xencode.schemas import XencodeManifest
        m = XencodeManifest(
            project_name="roundtrip",
            language="rust",
            workspace="/ws/roundtrip",
            files_written=["src/main.rs", "Cargo.toml"],
            build_command="cargo build --release",
            build_success=True,
            build_output="Compiling roundtrip v0.1.0\nFinished",
            zip_path="/artifacts/roundtrip.zip",
            validation_errors=[],
            repair_attempts=0,
        )
        raw_json = m.model_dump_json(indent=2)
        data = json.loads(raw_json)

        self.assertEqual(data["project_name"], "roundtrip")
        self.assertEqual(data["language"], "rust")
        self.assertEqual(data["files_written"], ["src/main.rs", "Cargo.toml"])
        self.assertTrue(data["build_success"])

    def test_manifest_from_dict(self):
        from coding.xencode.schemas import XencodeManifest
        data = {
            "project_name": "from-dict",
            "language": "typescript",
            "workspace": "/ws/from-dict",
            "files_written": ["index.ts"],
            "build_success": False,
            "repair_attempts": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        m = XencodeManifest(**data)
        self.assertEqual(m.project_name, "from-dict")
        self.assertEqual(m.repair_attempts, 1)
        self.assertFalse(m.build_success)

    def test_manifest_write_and_read_file(self):
        from coding.xencode.schemas import XencodeManifest
        with tempfile.TemporaryDirectory() as tmp:
            m = XencodeManifest(
                project_name="file-test",
                language="python",
                workspace=tmp,
                files_written=["main.py"],
                build_success=True,
            )
            manifest_path = Path(tmp) / ".xencode" / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(m.model_dump_json(indent=2), encoding="utf-8")

            loaded = XencodeManifest.model_validate_json(manifest_path.read_text())
            self.assertEqual(loaded.project_name, "file-test")
            self.assertTrue(loaded.build_success)
            self.assertEqual(loaded.files_written, ["main.py"])


# ══════════════════════════════════════════════════════════════════════════════
# SpecParser — heuristic fallback (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

class TestSpecParserHeuristic(unittest.TestCase):
    """Test the heuristic path of SpecParser (no real LLM calls)."""

    def _make_parser(self):
        from coding.xencode.spec_parser import SpecParser
        parser = SpecParser.__new__(SpecParser)
        parser.llm = _mock_llm()
        return parser

    def test_detect_python_from_prompt(self):
        from coding.xencode.spec_parser import SpecParser
        parser = self._make_parser()
        spec = parser._parse_heuristic(
            "Build a FastAPI REST API with pytest tests", "my-api", "python"
        )
        self.assertEqual(spec.language, "python")
        self.assertIn("fastapi", spec.requirements)
        self.assertIn("pytest", spec.requirements)

    def test_detect_javascript_from_prompt(self):
        from coding.xencode.spec_parser import SpecParser
        parser = self._make_parser()
        spec = parser._parse_heuristic(
            "Create a React web app with Express backend using npm", "react-app", "python"
        )
        self.assertEqual(spec.language, "javascript")

    def test_detect_typescript_from_prompt(self):
        from coding.xencode.spec_parser import SpecParser
        parser = self._make_parser()
        spec = parser._parse_heuristic(
            "Build an Angular TypeScript application", "ng-app", "python"
        )
        self.assertEqual(spec.language, "typescript")

    def test_detect_rust_from_prompt(self):
        from coding.xencode.spec_parser import SpecParser
        parser = self._make_parser()
        # Detects Rust via '.rs' and 'rust' keywords in the prompt
        spec = parser._parse_heuristic(
            "Write a Rust CLI tool — compile with `rustc` and use .rs source files", "rust-cli", "python"
        )
        self.assertEqual(spec.language, "rust")

    def test_detect_go_from_prompt(self):
        from coding.xencode.spec_parser import SpecParser
        parser = self._make_parser()
        spec = parser._parse_heuristic(
            "Create a goroutine-based HTTP server in golang", "go-srv", "python"
        )
        self.assertEqual(spec.language, "go")

    def test_fallback_to_default_language(self):
        from coding.xencode.spec_parser import SpecParser
        parser = self._make_parser()
        spec = parser._parse_heuristic(
            "Build something awesome without any language hints", "generic-app", "csharp"
        )
        self.assertEqual(spec.language, "csharp")

    def test_project_name_slugified(self):
        from coding.xencode.spec_parser import SpecParser
        parser = self._make_parser()
        spec = parser._parse_heuristic("Some project", "My Cool Project!", "python")
        self.assertNotIn(" ", spec.project_name)
        self.assertNotIn("!", spec.project_name)

    def test_entry_point_matches_language(self):
        from coding.xencode.spec_parser import SpecParser
        parser = self._make_parser()
        spec = parser._parse_heuristic("A Go service using goroutine", "svc", "go")
        self.assertEqual(spec.entry_point, "main.go")

    def test_build_command_python(self):
        from coding.xencode.spec_parser import SpecParser
        parser = self._make_parser()
        spec = parser._parse_heuristic("A Python FastAPI app", "api", "python")
        self.assertIn("pip", spec.build_command)

    def test_llm_failure_falls_back_to_heuristic(self):
        """parse() should use _parse_heuristic when LLM raises."""
        from coding.xencode.spec_parser import SpecParser
        parser = SpecParser.__new__(SpecParser)
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("no API key"))
        parser.llm = llm

        spec = _run(parser.parse("A Python project with flask", "flask-app", "python"))
        self.assertEqual(spec.language, "python")
        self.assertIn("flask", spec.requirements)

    def test_llm_success_uses_llm_result(self):
        """parse() should use LLM response when it succeeds."""
        from coding.xencode.spec_parser import SpecParser
        llm_json = json.dumps({
            "project_name": "llm-proj",
            "description": "An LLM project",
            "language": "typescript",
            "requirements": ["express"],
            "entry_point": "index.ts",
            "build_command": "npm install",
            "test_command": "npm test",
        })
        parser = SpecParser.__new__(SpecParser)
        parser.llm = _mock_llm(llm_json)

        spec = _run(parser.parse("Build a TypeScript Express API", "llm-proj", "typescript"))
        self.assertEqual(spec.language, "typescript")
        self.assertEqual(spec.entry_point, "index.ts")
        self.assertIn("express", spec.requirements)


# ══════════════════════════════════════════════════════════════════════════════
# Validator — static checks
# ══════════════════════════════════════════════════════════════════════════════

class TestValidator(unittest.TestCase):
    def setUp(self):
        from coding.xencode.validator import Validator
        from coding.xencode.schemas import XencodeSpec
        self.Validator = Validator
        self.XencodeSpec = XencodeSpec
        self._tmp = tempfile.mkdtemp()

    def _spec(self, **kwargs):
        defaults = dict(project_name="test", description="test project", language="python")
        defaults.update(kwargs)
        return self.XencodeSpec(**defaults)

    def test_nonexistent_workspace(self):
        v = self.Validator()
        result = v.validate("/nonexistent/path/xyz", self._spec())
        self.assertFalse(result.valid)
        self.assertTrue(any("does not exist" in e for e in result.errors))

    def test_missing_entry_point(self):
        v = self.Validator()
        # workspace exists but entry point file does not
        result = v.validate(self._tmp, self._spec(entry_point="main.py"))
        self.assertFalse(result.valid)
        self.assertTrue(any("Entry point" in e for e in result.errors))

    def test_missing_readme_warning(self):
        v = self.Validator()
        # Create entry point but no README
        Path(self._tmp, "main.py").write_text("print('hello')")
        result = v.validate(self._tmp, self._spec())
        self.assertTrue(any("README" in w for w in result.warnings))

    def test_valid_python_project(self):
        v = self.Validator()
        Path(self._tmp, "main.py").write_text("def main():\n    pass\n")
        Path(self._tmp, "README.md").write_text("# Test")
        result = v.validate(self._tmp, self._spec())
        self.assertTrue(result.valid)
        self.assertEqual(result.errors, [])

    def test_python_syntax_error_detected(self):
        v = self.Validator()
        Path(self._tmp, "main.py").write_text("def broken(:\n    pass\n")
        Path(self._tmp, "README.md").write_text("# Test")
        result = v.validate(self._tmp, self._spec())
        self.assertFalse(result.valid)
        self.assertTrue(any("Syntax error" in e for e in result.errors))

    def test_js_missing_package_json_warning(self):
        v = self.Validator()
        Path(self._tmp, "index.js").write_text("console.log('hi');")
        Path(self._tmp, "README.md").write_text("# Test")
        result = v.validate(
            self._tmp,
            self._spec(language="javascript", entry_point="index.js"),
        )
        self.assertTrue(any("package.json" in w for w in result.warnings))

    def test_go_missing_go_mod_warning(self):
        v = self.Validator()
        Path(self._tmp, "main.go").write_text('package main\nfunc main() {}')
        Path(self._tmp, "README.md").write_text("# Test")
        result = v.validate(
            self._tmp,
            self._spec(language="go", entry_point="main.go"),
        )
        self.assertTrue(any("go.mod" in w for w in result.warnings))

    def test_rust_missing_cargo_toml_error(self):
        v = self.Validator()
        src_dir = Path(self._tmp, "src")
        src_dir.mkdir()
        (src_dir / "main.rs").write_text("fn main() {}")
        Path(self._tmp, "README.md").write_text("# Test")
        result = v.validate(
            self._tmp,
            self._spec(language="rust", entry_point="src/main.rs"),
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("Cargo.toml" in e for e in result.errors))


# ══════════════════════════════════════════════════════════════════════════════
# Finalizer — ZIP creation
# ══════════════════════════════════════════════════════════════════════════════

class TestFinalizer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._artifacts = os.path.join(self._tmp, "artifacts")

    def _spec(self, **kwargs):
        from coding.xencode.schemas import XencodeSpec
        defaults = dict(
            project_name="my-project",
            description="test project",
            language="python",
            build_command=None,
        )
        defaults.update(kwargs)
        return XencodeSpec(**defaults)

    def test_zip_created_no_build_command(self):
        """Finalizer should create a ZIP even when build_command is None."""
        from coding.xencode.finalizer import Finalizer
        ws = Path(self._tmp, "workspace")
        ws.mkdir()
        (ws / "main.py").write_text("print('hello')")
        (ws / "README.md").write_text("# My Project")

        result = _run(
            Finalizer().finalize(str(ws), self._spec(), self._artifacts)
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.zip_path)
        self.assertTrue(os.path.exists(result.zip_path))

    def test_zip_contains_expected_files(self):
        from coding.xencode.finalizer import Finalizer
        ws = Path(self._tmp, "ws2")
        ws.mkdir()
        (ws / "main.py").write_text("x = 1")
        (ws / "helper.py").write_text("y = 2")
        (ws / "README.md").write_text("# Readme")

        result = _run(Finalizer().finalize(str(ws), self._spec(), self._artifacts))

        with zipfile.ZipFile(result.zip_path) as zf:
            names = zf.namelist()
        self.assertIn("main.py", names)
        self.assertIn("helper.py", names)
        self.assertIn("README.md", names)

    def test_zip_excludes_pycache(self):
        from coding.xencode.finalizer import Finalizer
        ws = Path(self._tmp, "ws3")
        ws.mkdir()
        (ws / "main.py").write_text("x = 1")
        cache_dir = ws / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "main.cpython-311.pyc").write_bytes(b"\x00\x01\x02")

        result = _run(Finalizer().finalize(str(ws), self._spec(), self._artifacts))

        with zipfile.ZipFile(result.zip_path) as zf:
            names = zf.namelist()
        self.assertNotIn("__pycache__/main.cpython-311.pyc", names)
        self.assertIn("main.py", names)

    def test_zip_excludes_dot_xencode(self):
        """The .xencode manifest directory must not be included in the ZIP."""
        from coding.xencode.finalizer import Finalizer
        ws = Path(self._tmp, "ws4")
        ws.mkdir()
        (ws / "main.py").write_text("pass")
        xencode_dir = ws / ".xencode"
        xencode_dir.mkdir()
        (xencode_dir / "manifest.json").write_text("{}")

        result = _run(Finalizer().finalize(str(ws), self._spec(), self._artifacts))

        with zipfile.ZipFile(result.zip_path) as zf:
            names = zf.namelist()
        self.assertFalse(any(".xencode" in n for n in names))

    def test_zip_named_after_project(self):
        from coding.xencode.finalizer import Finalizer
        ws = Path(self._tmp, "ws5")
        ws.mkdir()
        (ws / "main.py").write_text("pass")
        spec = self._spec(project_name="cool-project")

        result = _run(Finalizer().finalize(str(ws), spec, self._artifacts))

        self.assertIn("cool-project", os.path.basename(result.zip_path))

    def test_repair_skipped_when_no_llm(self):
        """When llm=None and build fails, _attempt_repair returns False immediately.
        repair_attempts is incremented (in finalizer.py) before _attempt_repair is
        called, so after the early break it equals 1 even though no repair was applied."""
        from coding.xencode.finalizer import Finalizer
        ws = Path(self._tmp, "ws6")
        ws.mkdir()
        (ws / "main.py").write_text("pass")
        spec = self._spec(build_command="false")  # always exits non-zero

        result = _run(
            Finalizer(max_repair_attempts=2).finalize(str(ws), spec, self._artifacts, llm=None)
        )

        # Build should fail; repair loop runs but no repair is applied (no llm)
        self.assertFalse(result.success)
        # The loop increments the counter on the first attempt then breaks early
        self.assertEqual(result.repair_attempts, 1)


# ══════════════════════════════════════════════════════════════════════════════
# XencodeEngine — integration (all heavy deps mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestXencodeEngine(unittest.TestCase):
    """Smoke-tests for XencodeEngine.compile() with mocked sub-components."""

    def _make_engine(self, tmp_workspace: str) -> "XencodeEngine":
        from coding.xencode.engine import XencodeEngine
        from coding.xencode.schemas import XencodeSpec, XencodePlan, PlannedFile

        spec = XencodeSpec(
            project_name="smoke-test",
            description="A smoke test project",
            language="python",
        )
        plan = XencodePlan(
            spec=spec,
            files=[PlannedFile(path="main.py", content="print('smoke')")],
            workspace=tmp_workspace,
        )

        engine = XencodeEngine.__new__(XencodeEngine)
        engine.llm = _mock_llm()
        engine.audit = MagicMock()

        engine.spec_parser = MagicMock()
        engine.spec_parser.parse = AsyncMock(return_value=spec)

        engine.planner = MagicMock()
        engine.planner.plan = AsyncMock(return_value=plan)

        engine.validator = MagicMock()
        from coding.xencode.validator import ValidationResult
        engine.validator.validate = MagicMock(return_value=ValidationResult(valid=True))

        from coding.xencode.finalizer import FinalizeResult
        engine.finalizer = MagicMock()
        engine.finalizer.finalize = AsyncMock(
            return_value=FinalizeResult(
                success=True,
                build_output="ok",
                zip_path=os.path.join(tmp_workspace, "smoke-test.zip"),
                repair_attempts=0,
            )
        )

        # Writer needs to write actual files
        from coding.xencode.writer import Writer
        real_writer = Writer(audit=engine.audit)
        engine._real_writer = real_writer

        return engine

    def test_compile_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)

            # Patch Writer so it writes files and returns a list
            with patch("coding.xencode.engine.Writer") as MockWriter:
                mock_writer_inst = MagicMock()
                mock_writer_inst.write = AsyncMock(return_value=["main.py"])
                MockWriter.return_value = mock_writer_inst

                from coding.xencode.schemas import XencodeCompileRequest
                req = XencodeCompileRequest(
                    prompt="A Python hello world",
                    project_name="smoke-test",
                    workspace=tmp,
                )
                resp = _run(engine.compile(req))

            self.assertTrue(resp.success)
            self.assertIsNone(resp.error)
            self.assertEqual(resp.project_name, "smoke-test")
            self.assertIn("main.py", resp.files_written)

    def test_compile_error_returns_failure_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            engine.spec_parser.parse = AsyncMock(side_effect=RuntimeError("LLM down"))

            from coding.xencode.schemas import XencodeCompileRequest
            req = XencodeCompileRequest(
                prompt="anything",
                project_name="error-test",
                workspace=tmp,
            )
            resp = _run(engine.compile(req))

            self.assertFalse(resp.success)
            self.assertIsNotNone(resp.error)

    def test_stream_compile_yields_expected_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)

            with patch("coding.xencode.engine.Writer") as MockWriter:
                mock_writer_inst = MagicMock()
                mock_writer_inst.write = AsyncMock(return_value=["main.py"])
                MockWriter.return_value = mock_writer_inst

                from coding.xencode.schemas import XencodeCompileRequest
                req = XencodeCompileRequest(
                    prompt="A Python project",
                    project_name="smoke-test",
                    workspace=tmp,
                )

                async def collect_events():
                    events = []
                    async for event in engine.stream_compile(req):
                        events.append(event)
                    return events

                events = _run(collect_events())

            event_names = [e["event"] for e in events]
            self.assertIn("spec_parsed", event_names)
            self.assertIn("plan", event_names)
            self.assertIn("files_written", event_names)
            self.assertIn("validation", event_names)
            self.assertIn("final_build_started", event_names)
            self.assertIn("done", event_names)

    def test_stream_compile_done_event_has_zip_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)

            with patch("coding.xencode.engine.Writer") as MockWriter:
                mock_writer_inst = MagicMock()
                mock_writer_inst.write = AsyncMock(return_value=["main.py"])
                MockWriter.return_value = mock_writer_inst

                from coding.xencode.schemas import XencodeCompileRequest
                req = XencodeCompileRequest(
                    prompt="A project",
                    project_name="smoke-test",
                    workspace=tmp,
                )

                async def collect_events():
                    events = []
                    async for event in engine.stream_compile(req):
                        events.append(event)
                    return events

                events = _run(collect_events())

            done = next(e for e in events if e["event"] == "done")
            self.assertTrue(done["data"]["success"])
            self.assertIn("zip_path", done["data"])


if __name__ == "__main__":
    unittest.main()
