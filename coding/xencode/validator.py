"""Static validation of a Xencode workspace — no code execution."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from coding.xencode.schemas import XencodeSpec

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of static workspace validation."""

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class Validator:
    """Static validator for a Xencode workspace (no code is executed)."""

    def validate(self, workspace: str, spec: XencodeSpec) -> ValidationResult:
        """Validate the workspace contents and return a ValidationResult."""
        result = ValidationResult()
        ws = Path(workspace)

        if not ws.exists() or not ws.is_dir():
            result.valid = False
            result.errors.append(f"Workspace directory does not exist: {workspace}")
            return result

        # Check entry point exists
        entry = ws / spec.entry_point
        if not entry.exists():
            result.errors.append(f"Entry point not found: {spec.entry_point}")
            result.valid = False

        # README check
        if not (ws / "README.md").exists():
            result.warnings.append("README.md is missing.")

        # Language-specific static checks
        if spec.language == "python":
            self._validate_python(ws, result)
        elif spec.language in ("javascript", "typescript"):
            self._validate_js(ws, result)
        elif spec.language == "go":
            self._validate_go(ws, result)
        elif spec.language == "rust":
            self._validate_rust(ws, result)

        return result

    def _validate_python(self, ws: Path, result: ValidationResult) -> None:
        """Syntax-check all Python files using ast.parse."""
        for py_file in ws.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                ast.parse(source, filename=str(py_file))
            except SyntaxError as exc:
                rel = py_file.relative_to(ws)
                result.errors.append(
                    f"Syntax error in {rel} line {exc.lineno}: {exc.msg}"
                )
                result.valid = False
            except Exception as exc:
                logger.debug("Could not parse %s: %s", py_file, exc)

    def _validate_js(self, ws: Path, result: ValidationResult) -> None:
        if not (ws / "package.json").exists():
            result.warnings.append("package.json not found.")

    def _validate_go(self, ws: Path, result: ValidationResult) -> None:
        if not (ws / "go.mod").exists():
            result.warnings.append("go.mod not found.")

    def _validate_rust(self, ws: Path, result: ValidationResult) -> None:
        if not (ws / "Cargo.toml").exists():
            result.errors.append("Cargo.toml not found.")
            result.valid = False
