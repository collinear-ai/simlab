"""Dynamic agent loading for import paths."""

from __future__ import annotations

import importlib

from simlab.agents.base import BaseAgent


def load_agent_class(import_path: str) -> type[BaseAgent]:
    """Load an agent class from module:Class import path."""
    if ":" not in import_path:
        raise ValueError("agent import path must be in format 'module.path:ClassName'")
    module_path, class_name = import_path.split(":", 1)
    module = importlib.import_module(module_path)
    agent_cls = getattr(module, class_name, None)
    if agent_cls is None:
        raise ValueError(f"Could not find class '{class_name}' in module '{module_path}'")
    if not isinstance(agent_cls, type) or not issubclass(agent_cls, BaseAgent):
        raise TypeError(f"{import_path} is not a BaseAgent subclass")
    return agent_cls
