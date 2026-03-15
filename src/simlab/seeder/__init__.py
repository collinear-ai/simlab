"""Shared utilities for seeding data into tool servers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from simlab.catalog.registry import ToolRegistry
from simlab.composer.engine import EnvConfig


def query_tool_server(
    url: str,
    tool_name: str,
    parameters: dict[str, Any],
) -> Any:
    """Execute a tool action against a tool server.

    Sends POST to ``{url}/step`` with ``{"action": {"tool_name": ..., "parameters": ...}}``.
    """
    payload = json.dumps({"action": {"tool_name": tool_name, "parameters": parameters}}).encode()
    req = urllib.request.Request(
        f"{url}/step",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def get_tool_endpoints(config: EnvConfig, config_path: Path | None = None) -> dict[str, str]:
    """Build ``{tool_name: "http://localhost:{port}"}`` from config + registry."""
    _ = config_path
    registry = ToolRegistry()
    registry.load_all()
    endpoints: dict[str, str] = {}
    for tool_name in config.tools:
        tool = registry.get_tool(tool_name)
        if tool:
            if tool.is_external:
                endpoints[tool.name] = tool.tool_server_url  # type: ignore[assignment]
            else:
                endpoints[tool.name] = f"http://localhost:{tool.tool_server_port}"
    return endpoints
