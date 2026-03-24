"""Sync MCP client handle for use by the reference agent (list_tools / call_tool)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from simlab.agents.base import ToolCallResult


def _extract_text(result: object) -> tuple[str, bool]:
    """Extract observation text and error flag from MCP call_tool result."""
    text_parts: list[str] = []
    if hasattr(result, "content"):
        text_parts.extend(item.text for item in result.content if hasattr(item, "text"))
    text = "\n".join(text_parts) if text_parts else json.dumps(result, default=str)
    is_error = getattr(result, "isError", False) if hasattr(result, "isError") else False
    return text, is_error


async def _async_list_tools(
    url: str,
    timeout: float = 15.0,  # noqa: ASYNC109
) -> list[dict[str, Any]]:
    """Connect to MCP server at url, list tools; return dicts (name, description, input_schema)."""
    base_url = url.rstrip("/")
    async with (
        asyncio.timeout(timeout),
        streamablehttp_client(base_url, timeout=timeout) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await asyncio.wait_for(session.list_tools(), timeout=timeout)
        return [
            {
                "name": t.name,
                "description": (t.description or ""),
                "input_schema": getattr(t, "inputSchema", None) or {"type": "object"},
            }
            for t in result.tools
        ]


async def _async_call_tool(
    url: str,
    tool_name: str,
    parameters: dict[str, Any],
    timeout: float = 60.0,  # noqa: ASYNC109
) -> ToolCallResult:
    """Connect to MCP server at url, call tool, return ToolCallResult."""
    base_url = url.rstrip("/")
    async with (
        asyncio.timeout(timeout),
        streamablehttp_client(base_url, timeout=timeout) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await asyncio.wait_for(session.call_tool(tool_name, parameters), timeout=timeout)
        text, is_error = _extract_text(result)
        return ToolCallResult(observation=text, is_error=is_error)


class MCPClientHandle:
    """Sync handle to an MCP server: list_tools and call_tool run via asyncio.run()."""

    def __init__(self, url: str, server_name: str, tool_prefix: str | None = None) -> None:
        """Store URL and server name for this MCP server handle."""
        self._url = url
        self._server_name = server_name
        self._tool_prefix = tool_prefix

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tools from this MCP server (name, description, input_schema, tool_server)."""
        tools = asyncio.run(_async_list_tools(self._url))
        if self._tool_prefix:
            tools = [
                {**t, "name": str(t.get("name", ""))[len(self._tool_prefix) :]}
                for t in tools
                if str(t.get("name", "")).startswith(self._tool_prefix)
            ]
        for t in tools:
            t["tool_server"] = self._server_name
        return tools

    def call_tool(self, tool_name: str, parameters: dict[str, Any]) -> ToolCallResult:
        """Call a tool on this MCP server."""
        actual_name = (
            f"{self._tool_prefix}{tool_name}"
            if self._tool_prefix and not tool_name.startswith(self._tool_prefix)
            else tool_name
        )
        return asyncio.run(_async_call_tool(self._url, actual_name, parameters))
