"""Coding engine package."""

# Imports are intentionally deferred — import submodules directly to avoid
# pulling in heavy optional dependencies (litellm, etc.) at package load time.

__all__ = ["CodingEngine"]
