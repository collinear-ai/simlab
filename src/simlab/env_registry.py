"""Helpers for loading built-in and env-local tool definitions."""

from __future__ import annotations

from pathlib import Path

from simlab.catalog.registry import ToolRegistry

CUSTOM_TOOLS_DIRNAME = "custom-tools"


def get_custom_tools_dir(env_dir: Path) -> Path:
    """Return the directory containing env-local custom tool definitions."""
    return env_dir / CUSTOM_TOOLS_DIRNAME


def build_registry(*, env_dir: Path | None = None) -> ToolRegistry:
    """Build a tool registry that includes env-local custom tools when available."""
    registry = ToolRegistry()
    registry.load_all(env_dir=env_dir)
    return registry
