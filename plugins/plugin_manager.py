"""Plugin Manager — discovers and loads third-party WISPR plugins."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Plugin:
    """Metadata container for a loaded plugin."""

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        handler: Callable,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self.handler = handler
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Plugin(name={self.name!r}, version={self.version!r})"


class PluginManager:
    """Discovers, loads, and invokes WISPR plugins from the plugins directory.

    A plugin is any Python file inside the plugins directory (or a sub-package)
    that exposes a top-level ``register()`` function returning a ``Plugin``
    instance.

    Example plugin skeleton::

        # plugins/my_plugin.py
        from plugins.plugin_manager import Plugin

        def register() -> Plugin:
            async def handler(task: str, context: dict) -> str:
                return f"My plugin handled: {task}"

            return Plugin(
                name="MyPlugin",
                version="0.1.0",
                description="A sample plugin.",
                handler=handler,
            )
    """

    def __init__(self, plugins_dir: Optional[str] = None) -> None:
        self._dir = Path(plugins_dir or settings.plugins_dir)
        self._plugins: Dict[str, Plugin] = {}
        self._discover()

    # ── public API ────────────────────────────────────────────────────────

    def list_plugins(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "metadata": p.metadata,
            }
            for p in self._plugins.values()
        ]

    def get_plugin(self, name: str) -> Optional[Plugin]:
        return self._plugins.get(name)

    async def invoke(
        self, name: str, task: str, context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Call the named plugin's handler."""
        plugin = self._plugins.get(name)
        if not plugin:
            raise KeyError(f"Plugin '{name}' not found.")
        return await plugin.handler(task, context or {})

    def register_plugin(self, plugin: Plugin) -> None:
        """Manually register a plugin (useful in tests or dynamic scenarios)."""
        self._plugins[plugin.name] = plugin
        logger.info("Registered plugin: %s v%s", plugin.name, plugin.version)

    # ── internals ─────────────────────────────────────────────────────────

    def _discover(self) -> None:
        if not self._dir.exists():
            logger.debug("Plugins directory %s does not exist; skipping discovery.", self._dir)
            return

        for path in sorted(self._dir.rglob("*.py")):
            # Skip __init__ and plugin_manager itself
            if path.stem.startswith("__") or path.stem == "plugin_manager":
                continue
            self._load_plugin_file(path)

    def _load_plugin_file(self, path: Path) -> None:
        module_name = f"_wispr_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if not spec or not spec.loader:
                return
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]

            register_fn = getattr(module, "register", None)
            if callable(register_fn):
                plugin: Plugin = register_fn()
                if isinstance(plugin, Plugin):
                    self._plugins[plugin.name] = plugin
                    logger.info("Loaded plugin '%s' from %s", plugin.name, path)
        except Exception as exc:
            logger.warning("Failed to load plugin from %s: %s", path, exc)
