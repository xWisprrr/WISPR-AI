"""Code Engine — real-time filesystem tools.

Provides unrestricted local filesystem access (user-selected option 3) with:
  • Path normalisation & traversal prevention
  • Dangerous-path denylist (configurable via WISPR_DENIED_PATHS env var)
  • Full audit logging for every operation
  • Async-first API (aiofiles) with sync fallback helpers

Supported operations
--------------------
read_file, write_file, create_file, edit_file, delete_file,
list_dir, search_files, move_file, mkdir, file_info
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import aiofiles.os

from coding.tools.audit_log import AuditLog

logger = logging.getLogger(__name__)

# ── Dangerous-path denylist ───────────────────────────────────────────────────
# Paths listed here (glob-style) will be blocked unconditionally.
# Users can prepend extra patterns via the WISPR_DENIED_PATHS env var
# (colon-separated on Unix, semicolon-separated on Windows).

_BUILTIN_DENIED: List[str] = [
    # SSH & credentials
    str(Path.home() / ".ssh"),
    str(Path.home() / ".gnupg"),
    str(Path.home() / ".aws"),
    str(Path.home() / ".config" / "gcloud"),
    # System directories (Unix)
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64",
    "/boot", "/proc", "/sys", "/dev",
    # macOS system
    "/System", "/Library", "/private",
    # Windows system
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
]


def _load_denied_paths() -> List[str]:
    extra_raw = os.environ.get("WISPR_DENIED_PATHS", "")
    sep = ";" if os.name == "nt" else ":"
    extra = [p.strip() for p in extra_raw.split(sep) if p.strip()]
    return _BUILTIN_DENIED + extra


_DENIED_PATHS: List[str] = _load_denied_paths()


def _is_denied(resolved: Path) -> Tuple[bool, str]:
    """Return (True, reason) if *resolved* is inside a denied path."""
    resolved_str = str(resolved)
    for denied in _DENIED_PATHS:
        # Always resolve the denied path, regardless of whether it exists,
        # to prevent symlink-based bypasses.
        denied_resolved = str(Path(denied).resolve())
        if resolved_str == denied_resolved or resolved_str.startswith(denied_resolved + os.sep):
            return True, f"Path '{resolved}' is inside a protected directory '{denied}'"
    return False, ""


def _safe_resolve(path: str) -> Path:
    """Resolve *path* to an absolute Path, raising ValueError on traversal or deny."""
    resolved = Path(path).expanduser().resolve()
    blocked, reason = _is_denied(resolved)
    if blocked:
        raise PermissionError(reason)
    return resolved


class FileTools:
    """Async-first local filesystem toolkit used by Code Engine agents."""

    def __init__(
        self,
        audit: Optional[AuditLog] = None,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> None:
        self.audit = audit or AuditLog()
        self.session_id = session_id
        self.agent_name = agent_name

    # ── helpers ───────────────────────────────────────────────────────────

    def _log(self, op: str, path: str, bytes_changed: Optional[int] = None, **kw) -> None:
        self.audit.record(
            operation=op,
            path=path,
            session_id=self.session_id,
            agent=self.agent_name,
            bytes_changed=bytes_changed,
            extra=kw or None,
        )

    # ── read ──────────────────────────────────────────────────────────────

    async def read_file(self, path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read and return the contents of a file."""
        try:
            resolved = _safe_resolve(path)
            if not resolved.is_file():
                return {"success": False, "error": f"'{path}' is not a file or does not exist."}
            async with aiofiles.open(resolved, "r", encoding=encoding) as fh:
                content = await fh.read()
            self._log("read", str(resolved), bytes_changed=len(content.encode(encoding)))
            return {"success": True, "path": str(resolved), "content": content, "size": len(content)}
        except (PermissionError, ValueError) as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("read_file error")
            return {"success": False, "error": str(exc)}

    # ── write (create or overwrite) ────────────────────────────────────────

    async def write_file(self, path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Write *content* to *path*, creating parent directories if needed."""
        try:
            resolved = _safe_resolve(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            existed = resolved.exists()
            op = "overwrite" if existed else "create"
            async with aiofiles.open(resolved, "w", encoding=encoding) as fh:
                await fh.write(content)
            self._log(op, str(resolved), bytes_changed=len(content.encode(encoding)))
            return {"success": True, "path": str(resolved), "operation": op, "bytes": len(content.encode(encoding))}
        except (PermissionError, ValueError) as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("write_file error")
            return {"success": False, "error": str(exc)}

    # ── create (fail if exists) ────────────────────────────────────────────

    async def create_file(self, path: str, content: str = "", encoding: str = "utf-8") -> Dict[str, Any]:
        """Create a new file, failing if it already exists."""
        try:
            resolved = _safe_resolve(path)
            if resolved.exists():
                return {"success": False, "error": f"'{path}' already exists. Use write_file to overwrite."}
            resolved.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(resolved, "w", encoding=encoding) as fh:
                await fh.write(content)
            self._log("create", str(resolved), bytes_changed=len(content.encode(encoding)))
            return {"success": True, "path": str(resolved), "bytes": len(content.encode(encoding))}
        except (PermissionError, ValueError) as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("create_file error")
            return {"success": False, "error": str(exc)}

    # ── edit (targeted find-and-replace) ──────────────────────────────────

    async def edit_file(self, path: str, old_str: str, new_str: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Replace the *first* occurrence of *old_str* with *new_str* in *path*.

        Note: only the first occurrence is replaced. If the string appears
        multiple times and you need a specific occurrence, read the file,
        make targeted edits, then use write_file.
        """
        try:
            resolved = _safe_resolve(path)
            if not resolved.is_file():
                return {"success": False, "error": f"'{path}' is not a file or does not exist."}
            async with aiofiles.open(resolved, "r", encoding=encoding) as fh:
                original = await fh.read()
            if old_str not in original:
                return {"success": False, "error": "old_str not found in file."}
            updated = original.replace(old_str, new_str, 1)
            async with aiofiles.open(resolved, "w", encoding=encoding) as fh:
                await fh.write(updated)
            delta = len(updated.encode(encoding)) - len(original.encode(encoding))
            self._log("edit", str(resolved), bytes_changed=abs(delta))
            return {"success": True, "path": str(resolved), "bytes_delta": delta}
        except (PermissionError, ValueError) as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("edit_file error")
            return {"success": False, "error": str(exc)}

    # ── delete ────────────────────────────────────────────────────────────

    async def delete_file(self, path: str) -> Dict[str, Any]:
        """Delete a single file."""
        try:
            resolved = _safe_resolve(path)
            if not resolved.exists():
                return {"success": False, "error": f"'{path}' does not exist."}
            if resolved.is_dir():
                return {"success": False, "error": f"'{path}' is a directory. Use delete_dir."}
            size = resolved.stat().st_size
            await aiofiles.os.remove(resolved)
            self._log("delete", str(resolved), bytes_changed=size)
            return {"success": True, "path": str(resolved)}
        except (PermissionError, ValueError) as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("delete_file error")
            return {"success": False, "error": str(exc)}

    # ── list directory ────────────────────────────────────────────────────

    async def list_dir(self, path: str = ".", max_entries: int = 500) -> Dict[str, Any]:
        """List the contents of a directory."""
        try:
            resolved = _safe_resolve(path)
            if not resolved.is_dir():
                return {"success": False, "error": f"'{path}' is not a directory."}
            entries = []
            for item in sorted(resolved.iterdir()):
                entries.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                })
                if len(entries) >= max_entries:
                    break
            self._log("list_dir", str(resolved))
            return {"success": True, "path": str(resolved), "entries": entries, "count": len(entries)}
        except (PermissionError, ValueError) as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("list_dir error")
            return {"success": False, "error": str(exc)}

    # ── search files ──────────────────────────────────────────────────────

    async def search_files(
        self,
        directory: str,
        pattern: str,
        *,
        glob_pattern: str = "**/*",
        max_matches: int = 100,
        use_regex: bool = False,
    ) -> Dict[str, Any]:
        """Search for *pattern* inside files under *directory*.

        Args:
            directory:    Root directory to search.
            pattern:      Text or regex to search for.
            glob_pattern: File glob to restrict which files are searched.
            max_matches:  Cap on total match records returned.
            use_regex:    If True, treat *pattern* as a regex.
        """
        try:
            resolved = _safe_resolve(directory)
            if not resolved.is_dir():
                return {"success": False, "error": f"'{directory}' is not a directory."}

            flags = re.IGNORECASE
            regex = re.compile(pattern, flags) if use_regex else None
            matches: List[Dict[str, Any]] = []

            for fpath in sorted(resolved.glob(glob_pattern)):
                if not fpath.is_file():
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for lineno, line in enumerate(text.splitlines(), 1):
                    hit = (regex.search(line) is not None) if regex else (pattern.lower() in line.lower())
                    if hit:
                        matches.append({
                            "file": str(fpath),
                            "line": lineno,
                            "content": line.rstrip(),
                        })
                    if len(matches) >= max_matches:
                        break
                if len(matches) >= max_matches:
                    break

            self._log("search", str(resolved), extra={"pattern": pattern})
            return {"success": True, "directory": str(resolved), "matches": matches, "count": len(matches)}
        except (PermissionError, ValueError) as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("search_files error")
            return {"success": False, "error": str(exc)}

    # ── move / rename ─────────────────────────────────────────────────────

    async def move_file(self, src: str, dst: str) -> Dict[str, Any]:
        """Move or rename *src* to *dst*."""
        try:
            src_resolved = _safe_resolve(src)
            dst_resolved = _safe_resolve(dst)
            if not src_resolved.exists():
                return {"success": False, "error": f"'{src}' does not exist."}
            dst_resolved.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_resolved), str(dst_resolved))
            self._log("move", str(src_resolved), extra={"destination": str(dst_resolved)})
            return {"success": True, "src": str(src_resolved), "dst": str(dst_resolved)}
        except (PermissionError, ValueError) as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("move_file error")
            return {"success": False, "error": str(exc)}

    # ── mkdir ─────────────────────────────────────────────────────────────

    async def mkdir(self, path: str) -> Dict[str, Any]:
        """Create a directory (and parents) at *path*."""
        try:
            resolved = _safe_resolve(path)
            resolved.mkdir(parents=True, exist_ok=True)
            self._log("mkdir", str(resolved))
            return {"success": True, "path": str(resolved)}
        except (PermissionError, ValueError) as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("mkdir error")
            return {"success": False, "error": str(exc)}

    # ── file info ─────────────────────────────────────────────────────────

    async def file_info(self, path: str) -> Dict[str, Any]:
        """Return metadata for *path*."""
        try:
            resolved = _safe_resolve(path)
            if not resolved.exists():
                return {"success": False, "error": f"'{path}' does not exist."}
            stat = resolved.stat()
            return {
                "success": True,
                "path": str(resolved),
                "type": "dir" if resolved.is_dir() else "file",
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
            }
        except (PermissionError, ValueError) as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("file_info error")
            return {"success": False, "error": str(exc)}
