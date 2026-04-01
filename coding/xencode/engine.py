"""Xencode pipeline engine — orchestrates the full compile pipeline with streaming support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from config import get_settings
from llm.router import LLMRouter
from coding.tools.audit_log import AuditLog
from coding.xencode.finalizer import Finalizer
from coding.xencode.planner import Planner
from coding.xencode.schemas import (
    XencodeCompileRequest,
    XencodeCompileResponse,
    XencodeManifest,
)
from coding.xencode.spec_parser import SpecParser
from coding.xencode.validator import Validator
from coding.xencode.writer import Writer

logger = logging.getLogger(__name__)
settings = get_settings()


class XencodeEngine:
    """End-to-end Xencode pipeline with optional SSE streaming."""

    def __init__(self, llm_router: Optional[LLMRouter] = None) -> None:
        self.llm = llm_router or LLMRouter()
        self.audit = AuditLog()
        self.spec_parser = SpecParser(llm=self.llm)
        self.planner = Planner(llm=self.llm)
        self.validator = Validator()
        self.finalizer = Finalizer(
            max_repair_attempts=settings.xencode_max_repair_attempts
        )

    # ── public API ────────────────────────────────────────────────────────

    async def compile(self, request: XencodeCompileRequest) -> XencodeCompileResponse:
        """Run the full pipeline and return a compile response."""
        workspace = self._resolve_workspace(request)
        try:
            spec = await self.spec_parser.parse(
                request.prompt, request.project_name, language=request.language
            )
            if request.build_command:
                spec.build_command = request.build_command

            plan = await self.planner.plan(spec, workspace)

            writer = Writer(audit=self.audit)
            files_written = await writer.write(plan)

            val_result = self.validator.validate(workspace, spec)

            fin_result = await self.finalizer.finalize(
                workspace,
                spec,
                settings.xencode_artifacts_dir,
                llm=self.llm,
            )

            manifest = XencodeManifest(
                project_name=spec.project_name,
                language=spec.language,
                workspace=workspace,
                files_written=files_written,
                build_command=spec.build_command,
                build_success=fin_result.success,
                build_output=fin_result.build_output,
                zip_path=fin_result.zip_path,
                validation_errors=val_result.errors,
                repair_attempts=fin_result.repair_attempts,
            )
            manifest_path = self._save_manifest(workspace, manifest)

            return XencodeCompileResponse(
                success=True,
                project_name=spec.project_name,
                workspace=workspace,
                files_written=files_written,
                build_success=fin_result.success,
                build_output=fin_result.build_output,
                zip_path=fin_result.zip_path,
                manifest_path=manifest_path,
                validation_errors=val_result.errors,
                repair_attempts=fin_result.repair_attempts,
            )
        except Exception as exc:
            logger.exception("Xencode compile error")
            return XencodeCompileResponse(
                success=False,
                project_name=request.project_name,
                workspace=workspace,
                error=str(exc),
            )

    async def stream_compile(
        self, request: XencodeCompileRequest
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield pipeline stage events as dicts for SSE streaming."""
        workspace = self._resolve_workspace(request)
        try:
            # Stage 1 — parse spec
            spec = await self.spec_parser.parse(
                request.prompt, request.project_name, language=request.language
            )
            if request.build_command:
                spec.build_command = request.build_command
            yield {"event": "spec_parsed", "data": spec.model_dump()}

            # Stage 2 — plan
            plan = await self.planner.plan(spec, workspace)
            yield {
                "event": "plan",
                "data": {
                    "file_count": len(plan.files),
                    "files": [f.path for f in plan.files],
                    "build_command": spec.build_command,
                    "workspace": workspace,
                },
            }

            # Stage 3 — write files
            writer = Writer(audit=self.audit)
            files_written = await writer.write(plan)
            yield {
                "event": "files_written",
                "data": {"files": files_written, "count": len(files_written)},
            }

            # Stage 4 — validate
            val_result = self.validator.validate(workspace, spec)
            yield {
                "event": "validation",
                "data": {
                    "valid": val_result.valid,
                    "errors": val_result.errors,
                    "warnings": val_result.warnings,
                },
            }

            # Stage 5 — final build
            yield {
                "event": "final_build_started",
                "data": {"build_command": spec.build_command},
            }
            fin_result = await self.finalizer.finalize(
                workspace, spec, settings.xencode_artifacts_dir, llm=self.llm
            )
            yield {
                "event": "final_build_output",
                "data": {
                    "success": fin_result.success,
                    "output": fin_result.build_output,
                    "repair_attempts": fin_result.repair_attempts,
                    "zip_path": fin_result.zip_path,
                },
            }

            # Stage 6 — persist manifest and signal done
            manifest = XencodeManifest(
                project_name=spec.project_name,
                language=spec.language,
                workspace=workspace,
                files_written=files_written,
                build_command=spec.build_command,
                build_success=fin_result.success,
                build_output=fin_result.build_output,
                zip_path=fin_result.zip_path,
                validation_errors=val_result.errors,
                repair_attempts=fin_result.repair_attempts,
            )
            manifest_path = self._save_manifest(workspace, manifest)

            yield {
                "event": "done",
                "data": {
                    "success": True,
                    "project_name": spec.project_name,
                    "workspace": workspace,
                    "files_written": files_written,
                    "build_success": fin_result.success,
                    "zip_path": fin_result.zip_path,
                    "manifest_path": manifest_path,
                    "validation_errors": val_result.errors,
                },
            }
        except Exception as exc:
            logger.exception("Xencode stream_compile error")
            yield {"event": "error", "data": {"error": "Pipeline error. Check server logs."}}

    # ── internals ─────────────────────────────────────────────────────────

    def _resolve_workspace(self, request: XencodeCompileRequest) -> str:
        if request.workspace:
            return request.workspace
        root = Path(settings.xencode_workspace_root)
        return str(root / request.project_name)

    def _save_manifest(self, workspace: str, manifest: XencodeManifest) -> str:
        """Persist manifest to <workspace>/.xencode/manifest.json."""
        manifest_dir = Path(workspace) / ".xencode"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return str(manifest_path)
