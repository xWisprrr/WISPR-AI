"""WISPR AI Studio — in-process code execution and deployment engine."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Mapping language name -> file extension and run command template
_LANG_CONFIG: Dict[str, Dict[str, str]] = {
    "python":     {"ext": ".py",   "cmd": "python3 {file}"},
    "javascript": {"ext": ".js",   "cmd": "node {file}"},
    "typescript": {"ext": ".ts",   "cmd": "npx ts-node {file}"},
    "go":         {"ext": ".go",   "cmd": "go run {file}"},
    "rust":       {"ext": ".rs",   "cmd": "rustc {file} -o {out} && {out}"},
    "java":       {"ext": ".java", "cmd": "javac {file} && java -cp {dir} Main"},
    "c":          {"ext": ".c",    "cmd": "gcc {file} -o {out} && {out}"},
    "cpp":        {"ext": ".cpp",  "cmd": "g++ {file} -o {out} && {out}"},
    "ruby":       {"ext": ".rb",   "cmd": "ruby {file}"},
    "bash":       {"ext": ".sh",   "cmd": "bash {file}"},
}

# Maximum output size returned to the caller (bytes).
_MAX_OUTPUT_BYTES = 1024 * 512  # 512 KB


class StudioIDE:
    """Sandboxed code execution environment with optional deployment support."""

    def __init__(self, sandbox_dir: Optional[str] = None) -> None:
        self._sandbox = Path(sandbox_dir or settings.studio_sandbox_dir)
        self._sandbox.mkdir(parents=True, exist_ok=True)

    # -- Execution ------------------------------------------------------------

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Run *code* in a temporary sandbox and return stdout/stderr.

        Args:
            code:     Source code string to execute.
            language: Programming language identifier (see supported list).
            timeout:  Maximum wall-clock seconds to allow (capped at 120 s).
        """
        lang = language.lower()
        cfg = _LANG_CONFIG.get(lang)
        if not cfg:
            supported = sorted(_LANG_CONFIG)
            return {
                "success": False,
                "stdout": "",
                "stderr": (
                    f"Unsupported language: {language!r}. "
                    f"Supported: {', '.join(supported)}"
                ),
                "language": language,
            }

        # Cap timeout to a safe maximum
        timeout = min(max(timeout, 1), 120)

        # Enforce an output size limit by truncating long output
        with tempfile.TemporaryDirectory(dir=self._sandbox) as tmp:
            src_file = Path(tmp) / f"main{cfg['ext']}"
            out_file = Path(tmp) / "out"
            src_file.write_text(code, encoding="utf-8")

            cmd = (
                cfg["cmd"]
                .replace("{file}", str(src_file))
                .replace("{out}", str(out_file))
                .replace("{dir}", tmp)
            )

            result = await self._run_subprocess(cmd, timeout, language)

        # Truncate oversized output to avoid unbounded memory usage
        for field in ("stdout", "stderr"):
            raw = result.get(field, "")
            raw_bytes = raw.encode("utf-8", errors="replace")
            if len(raw_bytes) > _MAX_OUTPUT_BYTES:
                cutoff = _MAX_OUTPUT_BYTES // 2
                truncated = raw_bytes[:cutoff].decode("utf-8", errors="ignore")
                result[field] = (
                    truncated
                    + f"\n\n[... output truncated after {cutoff} bytes ...]"
                )

        return result

    async def _run_subprocess(
        self, cmd: str, timeout: int, language: str
    ) -> Dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()  # reap the zombie process
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout}s.",
                    "language": language,
                    "returncode": -1,
                }

            return {
                "success": proc.returncode == 0,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "returncode": proc.returncode,
                "language": language,
            }
        except Exception as exc:
            logger.exception("StudioIDE subprocess error")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(exc),
                "language": language,
                "returncode": -1,
            }

    # -- Deployment -----------------------------------------------------------

    async def deploy(
        self,
        project_path: str,
        target: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return deployment instructions/steps for *target*."""
        target = target.lower()
        dispatchers = {
            "github":  self._deploy_github,
            "vercel":  self._deploy_vercel,
            "netlify": self._deploy_netlify,
        }
        handler = dispatchers.get(target)
        if not handler:
            supported = sorted(dispatchers)
            return {
                "success": False,
                "error": (
                    f"Unknown deploy target: {target!r}. "
                    f"Supported: {', '.join(supported)}"
                ),
            }
        return await handler(project_path, config or {})

    async def _deploy_github(self, path: str, config: Dict) -> Dict[str, Any]:
        repo   = config.get("repo", "your-username/your-repo")
        branch = config.get("branch", "main")
        return {
            "success": True,
            "target": "github",
            "steps": [
                f"cd {path}",
                "git init",
                f"git remote add origin https://github.com/{repo}.git",
                "git add .",
                'git commit -m "Deploy from WISPR AI Studio"',
                f"git push -u origin {branch}",
            ],
            "note": (
                "Ensure your GitHub credentials are configured via "
                "`gh auth login` or an SSH key before running these steps."
            ),
        }

    async def _deploy_vercel(self, path: str, config: Dict) -> Dict[str, Any]:
        return {
            "success": True,
            "target": "vercel",
            "steps": [
                "npm install -g vercel",
                f"cd {path}",
                "vercel --prod",
            ],
            "note": "Run `vercel login` first if not authenticated.",
        }

    async def _deploy_netlify(self, path: str, config: Dict) -> Dict[str, Any]:
        build_dir = config.get("build_dir", "dist")
        return {
            "success": True,
            "target": "netlify",
            "steps": [
                "npm install -g netlify-cli",
                f"cd {path}",
                f"netlify deploy --prod --dir {build_dir}",
            ],
            "note": "Run `netlify login` first if not authenticated.",
        }
