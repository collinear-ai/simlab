"""Helpers for env-local custom tool scaffolding and activation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import yaml

from simlab.env_artifacts import regenerate_env_artifacts
from simlab.env_registry import build_registry
from simlab.env_registry import get_custom_tools_dir

_CUSTOM_TOOL_SCAFFOLD = dedent(
    """\
    # Edit this scaffold before using the tool in a real task run.
    name: {name}
    display_name: {display_name}
    description: Custom tool scaffold for {name}.
    category: custom
    tool_server_url: http://localhost:9000
    """
)


@dataclass(frozen=True)
class CustomToolAddResult:
    """Result of scaffolding and enabling one env-local custom tool."""

    env_yaml: Path
    tool_file: Path


def add_custom_tool(env_dir: Path, name: str, *, force: bool) -> CustomToolAddResult:
    """Scaffold one env-local custom tool, enable it, and regenerate env artifacts."""
    env_yaml = env_dir / "env.yaml"
    builtins = build_registry()
    if builtins.get_tool(name) is not None:
        raise ValueError(
            f"Custom tool name '{name}' conflicts with a built-in tool. Choose a different name."
        )

    custom_tools_dir = get_custom_tools_dir(env_dir)
    custom_tools_dir.mkdir(parents=True, exist_ok=True)
    tool_file = custom_tools_dir / f"{name}.yaml"
    if tool_file.exists() and not force:
        raise FileExistsError(f"Custom tool scaffold already exists: {tool_file}")

    display_name = name.replace("-", " ").replace("_", " ").title()
    tool_file.write_text(
        _CUSTOM_TOOL_SCAFFOLD.format(name=name, display_name=display_name),
        encoding="utf-8",
    )

    config_data = yaml.safe_load(env_yaml.read_text(encoding="utf-8")) or {}
    tools = list(config_data.get("tools") or [])
    if name not in tools:
        tools.append(name)
    config_data["tools"] = tools
    env_yaml.write_text(
        yaml.dump(config_data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    regenerate_env_artifacts(env_dir)
    return CustomToolAddResult(env_yaml=env_yaml, tool_file=tool_file)
