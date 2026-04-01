"""Writes planned files to disk using FileTools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from coding.tools.audit_log import AuditLog
from coding.tools.file_tools import FileTools
from coding.xencode.schemas import XencodePlan

logger = logging.getLogger(__name__)


class Writer:
    """Writes all files in an XencodePlan to the workspace."""

    def __init__(
        self,
        audit: Optional[AuditLog] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.audit = audit or AuditLog()
        self.session_id = session_id

    async def write(self, plan: XencodePlan) -> List[str]:
        """Write all files in *plan* to disk. Returns list of relative paths written."""
        workspace = Path(plan.workspace)
        ft = FileTools(
            audit=self.audit,
            session_id=self.session_id,
            agent_name="xencode-writer",
        )
        await ft.mkdir(str(workspace))

        written: List[str] = []
        for planned_file in plan.files:
            abs_path = workspace / planned_file.path
            result = await ft.write_file(str(abs_path), planned_file.content)
            if result.get("success"):
                written.append(planned_file.path)
                logger.debug("Wrote %s", planned_file.path)
            else:
                logger.warning(
                    "Failed to write %s: %s", planned_file.path, result.get("error")
                )
        return written
