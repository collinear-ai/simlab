"""Shared utilities for seeding data into tool servers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from simlab.catalog.registry import ToolRegistry
from simlab.composer.engine import ComposeEngine
from simlab.composer.engine import EnvConfig
from simlab.composer.engine import get_mcp_gateway_host_port
from simlab.mcp_config import get_mcp_command_servers
from simlab.mcp_config import load_mcp_servers_from_env_dir


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
    """Build ``{tool_name: "http://localhost:{port}"}`` from config + registry.

    When config_path is set and the env has command-based MCP servers, adds the
    MCP gateway endpoint so callers can use it for the gateway URL.
    """
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
    if config_path is not None:
        env_dir = config_path.parent
        mcp_config = load_mcp_servers_from_env_dir(env_dir)
        if mcp_config and get_mcp_command_servers(mcp_config):
            gateway_port = get_mcp_gateway_host_port(env_dir)
            endpoints[ComposeEngine.MCP_GATEWAY_SERVICE_NAME] = (
                f"http://localhost:{gateway_port}/mcp"
            )
    return endpoints
