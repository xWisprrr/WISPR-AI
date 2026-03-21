"""
Tests for the Studio IDE.
"""
from __future__ import annotations

import pytest

from studio.ide import StudioIDE


class TestStudioIDE:
    def setup_method(self):
        self._ide = StudioIDE()

    def test_parse_project_files_with_convention(self):
        llm_output = (
            "### FILE: src/main.py\n"
            "```python\n"
            "print('hello')\n"
            "```\n"
            "### FILE: README.md\n"
            "```markdown\n"
            "# My Project\n"
            "```"
        )
        files = self._ide.parse_project_files(llm_output)
        assert "src/main.py" in files
        assert "README.md" in files
        assert "print" in files["src/main.py"]

    def test_parse_project_files_fallback(self):
        llm_output = "Some unstructured output without FILE markers"
        files = self._ide.parse_project_files(llm_output)
        assert "main.py" in files

    def test_create_and_get_project(self):
        files = {"index.py": "print('hello')"}
        self._ide.create_project("test-proj", files)
        project = self._ide.get_project("test-proj")
        assert project is not None
        assert "index.py" in project

    def test_list_projects(self):
        self._ide.create_project("proj-a", {"a.py": ""})
        self._ide.create_project("proj-b", {"b.py": ""})
        names = self._ide.list_projects()
        assert "proj-a" in names and "proj-b" in names

    def test_simulate_run_success(self):
        self._ide.create_project("run-test", {"main.py": "print('ok')"})
        result = self._ide.simulate_run("run-test")
        assert result["status"] == "success"
        assert result["entry_point_detected"] is True

    def test_simulate_run_missing_project(self):
        result = self._ide.simulate_run("does-not-exist")
        assert result["status"] == "error"

    def test_deployment_instructions_github(self):
        instructions = self._ide.deployment_instructions("github", "my-project")
        assert "github" in instructions.lower()
        assert "git" in instructions.lower()

    def test_deployment_instructions_vercel(self):
        instructions = self._ide.deployment_instructions("vercel", "my-project")
        assert "vercel" in instructions.lower()

    def test_deployment_instructions_netlify(self):
        instructions = self._ide.deployment_instructions("netlify", "my-project")
        assert "netlify" in instructions.lower()

    def test_deployment_instructions_unsupported(self):
        instructions = self._ide.deployment_instructions("heroku", "my-project")
        assert "not supported" in instructions.lower()
