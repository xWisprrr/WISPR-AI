"""Pydantic models for the Xencode pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class XencodeSpec(BaseModel):
    """Parsed specification for a Xencode project."""

    project_name: str
    description: str
    language: str = "python"
    requirements: List[str] = Field(default_factory=list)
    entry_point: str = "main.py"
    build_command: Optional[str] = None
    test_command: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class PlannedFile(BaseModel):
    """A single file in the project plan."""

    path: str
    content: str
    description: str = ""


class XencodePlan(BaseModel):
    """Project plan generated from a spec."""

    spec: XencodeSpec
    files: List[PlannedFile] = Field(default_factory=list)
    workspace: str = ""
    readme_included: bool = False


class XencodeManifest(BaseModel):
    """Persisted manifest for a Xencode build."""

    project_name: str
    language: str
    workspace: str
    files_written: List[str] = Field(default_factory=list)
    build_command: Optional[str] = None
    build_success: bool = False
    build_output: str = ""
    zip_path: Optional[str] = None
    validation_errors: List[str] = Field(default_factory=list)
    repair_attempts: int = 0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class XencodeCompileRequest(BaseModel):
    """Input for the Xencode compile endpoint."""

    prompt: str = Field(..., description="Project description or README content")
    project_name: str = Field(
        ..., description="Name for the project (used as workspace dirname)"
    )
    language: str = Field(
        default="python",
        description="Target language (auto-detected from prompt if not set)",
    )
    workspace: Optional[str] = Field(
        default=None, description="Override workspace path"
    )
    build_command: Optional[str] = Field(
        default=None, description="Override build command"
    )


class XencodeCompileResponse(BaseModel):
    """Response from the Xencode compile endpoint."""

    success: bool
    project_name: str
    workspace: str
    files_written: List[str] = Field(default_factory=list)
    build_success: bool = False
    build_output: str = ""
    zip_path: Optional[str] = None
    manifest_path: str = ""
    validation_errors: List[str] = Field(default_factory=list)
    repair_attempts: int = 0
    error: Optional[str] = None


class GitHubPublishRequest(BaseModel):
    """Input for the GitHub create-and-publish endpoint."""

    workspace: str = Field(..., description="Absolute path to the workspace to publish")
    repo_name: str = Field(..., description="GitHub repository name to create")
    private: bool = Field(default=False, description="Make the repository private")
    description: str = Field(default="", description="Repository description")


class GitHubPublishResponse(BaseModel):
    """Response from the GitHub create-and-publish endpoint."""

    success: bool
    repo_url: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
