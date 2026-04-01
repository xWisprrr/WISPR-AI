"""Final build step — runs the build command once and creates a ZIP deliverable."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from config import get_settings
from coding.xencode.schemas import XencodeSpec

logger = logging.getLogger(__name__)
_settings = get_settings()

_BUILD_TIMEOUT_SECONDS: int = _settings.xencode_build_timeout
_REPAIR_ERROR_MAX_CHARS: int = 1000

_ZIP_EXCLUDED: List[str] = [
    "__pycache__",
    ".pyc",
    ".pyo",
    ".git",
    "node_modules",
    "target/debug",
    "target/release",
    ".xencode",
]


@dataclass
class FinalizeResult:
    """Result of the finalize step."""

    success: bool = False
    build_output: str = ""
    zip_path: Optional[str] = None
    repair_attempts: int = 0
    errors: List[str] = field(default_factory=list)


class Finalizer:
    """Runs the build command and packages the project as a ZIP."""

    def __init__(self, max_repair_attempts: int = 2) -> None:
        self.max_repair_attempts = max_repair_attempts

    async def finalize(
        self,
        workspace: str,
        spec: XencodeSpec,
        artifacts_dir: str,
        llm: Any = None,
    ) -> FinalizeResult:
        """Run build once, ZIP the workspace, repair on failure up to the configured limit."""
        result = FinalizeResult()
        ws = Path(workspace)

        if spec.build_command:
            build_ok, build_output = await self._run_build(ws, spec.build_command)
            result.build_output = build_output

            if not build_ok:
                result.errors.append(f"Initial build failed: {build_output[:500]}")
                for attempt in range(self.max_repair_attempts):
                    result.repair_attempts += 1
                    logger.info(
                        "Build repair attempt %d/%d", attempt + 1, self.max_repair_attempts
                    )
                    repaired = await self._attempt_repair(ws, spec, build_output, llm)
                    if not repaired:
                        logger.info("No repair applied — stopping repair loop.")
                        break
                    build_ok, build_output = await self._run_build(ws, spec.build_command)
                    result.build_output = build_output
                    if build_ok:
                        result.errors.clear()
                        logger.info("Build succeeded after repair attempt %d.", attempt + 1)
                        break
                    result.errors.append(f"Post-repair build failed (attempt {attempt + 1})")

            result.success = build_ok
        else:
            result.success = True  # No build step

        result.zip_path = await self._create_zip(
            ws, spec.project_name, artifacts_dir
        )
        return result

    async def _run_build(
        self, workspace: Path, build_command: str
    ) -> Tuple[bool, str]:
        """Execute build_command in *workspace*. Returns (success, combined output)."""
        try:
            proc = await asyncio.create_subprocess_shell(
                build_command,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_BUILD_TIMEOUT_SECONDS)
            output = stdout.decode("utf-8", errors="replace")
            return proc.returncode == 0, output
        except asyncio.TimeoutError:
            return False, f"Build timed out after {_BUILD_TIMEOUT_SECONDS} seconds."
        except Exception as exc:
            return False, str(exc)

    async def _attempt_repair(
        self,
        workspace: Path,
        spec: XencodeSpec,
        error_output: str,
        llm: Any,
    ) -> bool:
        """Apply one LLM-suggested repair. Returns True if a file was modified."""
        if llm is None:
            return False
        try:
            from llm.router import TaskType

            prompt = (
                f"The following build command failed in a {spec.language} project:\n"
                f"Command: {spec.build_command}\n"
                f"Error output (truncated):\n{error_output[:_REPAIR_ERROR_MAX_CHARS]}\n\n"
                "Identify the most likely single fix. Reply with a JSON object only:\n"
                '{"action": "add_file|edit_file|skip", "path": "relative/path", "content": "..."}'
            )
            messages = [
                {"role": "system", "content": "You are a build-repair assistant. Reply only with JSON."},
                {"role": "user", "content": prompt},
            ]
            raw = await llm.complete(messages, task_type=TaskType.CODING, temperature=0.0)
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start < 0:
                return False
            data = json.loads(raw[start:end])
            action = data.get("action", "skip")
            if action == "skip":
                return False
            path = data.get("path", "")
            content = data.get("content", "")
            if path and content:
                target = workspace / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                logger.info("Repair wrote %s", path)
                return True
        except Exception as exc:
            logger.warning("Repair attempt failed: %s", exc)
        return False

    async def _create_zip(
        self, workspace: Path, project_name: str, artifacts_dir: str
    ) -> str:
        """Create a ZIP of the workspace. Returns the zip file path."""
        artifacts = Path(artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        zip_path = artifacts / f"{project_name}.zip"

        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(workspace.rglob("*")):
                if not file_path.is_file():
                    continue
                path_str = str(file_path)
                if any(excl in path_str for excl in _ZIP_EXCLUDED):
                    continue
                arcname = file_path.relative_to(workspace)
                zf.write(str(file_path), str(arcname))

        logger.info("Created ZIP at %s", zip_path)
        return str(zip_path)
