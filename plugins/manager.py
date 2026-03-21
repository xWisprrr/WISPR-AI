"""
Plugin System — dynamically discovers and loads agent plugins from the /plugins directory.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional, Type

from loguru import logger

from agents.base_agent import BaseAgent
from config import settings


class PluginManager:
    """
    Scans the plugins directory for Python modules that define a subclass
    of BaseAgent and registers them for use by the orchestrator.
    """

    def __init__(self, plugins_dir: Path = settings.plugins_dir) -> None:
        self._dir = Path(plugins_dir)
        self._plugins: dict[str, Type[BaseAgent]] = {}

    def discover(self) -> list[str]:
        """
        Walk the plugins directory and load any Python module that exposes
        a class named ``Plugin`` that inherits from ``BaseAgent``.

        Returns the names of successfully loaded plugins.
        """
        if not self._dir.exists():
            logger.debug(f"[PluginManager] Plugins directory '{self._dir}' does not exist.")
            return []

        loaded: list[str] = []
        for plugin_path in sorted(self._dir.glob("*.py")):
            if plugin_path.name.startswith("_"):
                continue
            try:
                module_name = f"plugins.{plugin_path.stem}"
                spec = importlib.util.spec_from_file_location(module_name, plugin_path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)  # type: ignore[attr-defined]

                plugin_class: Optional[Type[BaseAgent]] = getattr(module, "Plugin", None)
                if plugin_class and issubclass(plugin_class, BaseAgent):
                    name = getattr(plugin_class, "name", plugin_path.stem)
                    self._plugins[name] = plugin_class
                    loaded.append(name)
                    logger.info(f"[PluginManager] Loaded plugin '{name}'")
            except Exception as exc:
                logger.error(f"[PluginManager] Failed to load '{plugin_path.name}': {exc}")

        return loaded

    def get(self, name: str) -> Optional[Type[BaseAgent]]:
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict[str, str]]:
        return [
            {"name": name, "description": getattr(cls, "description", "")}
            for name, cls in self._plugins.items()
        ]

    def instantiate(self, name: str, **kwargs: Any) -> Optional[BaseAgent]:
        cls = self.get(name)
        if cls is None:
            return None
        return cls(**kwargs)

    @property
    def plugin_count(self) -> int:
        return len(self._plugins)
