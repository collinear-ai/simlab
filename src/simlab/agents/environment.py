"""Concrete environments for CLI external-agent runs."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
import warnings
from typing import Any

from simlab.agents.base import BaseEnvironment
from simlab.agents.base import ToolCallResult
from simlab.agents.base import ToolNamespace
from simlab.agents.mcp_client import MCPClientHandle


class UnifiedToolEnvironment(BaseEnvironment):
    """Transport-agnostic environment over HTTP tool servers and optional MCP servers."""

    def __init__(
        self,
        tool_servers: dict[str, str],
        timeout_seconds: float = 30.0,
        mcp_clients: dict[str, MCPClientHandle] | None = None,
    ) -> None:
        """Configure tool server URLs, optional MCP clients, and request timeout."""
        self._http_namespace_endpoints = dict(tool_servers)
        self._mcp_clients = dict(mcp_clients or {})
        duplicate_names = sorted(
            set(self._http_namespace_endpoints).intersection(self._mcp_clients)
        )
        if duplicate_names:
            joined = ", ".join(duplicate_names)
            raise ValueError(
                f"Tool namespace names must be unique across HTTP and MCP transports: {joined}"
            )
        self._timeout_seconds = timeout_seconds

    def list_tool_namespaces(self) -> list[ToolNamespace]:
        """Return the available tool namespaces across all transports."""
        namespaces = [
            ToolNamespace(name=name, transport="http", endpoint=url)
            for name, url in self._http_namespace_endpoints.items()
        ]
        namespaces.extend(
            ToolNamespace(
                name=name,
                transport="mcp",
                endpoint=getattr(handle, "_url", None),
            )
            for name, handle in self._mcp_clients.items()
        )
        return namespaces

    @property
    def tool_servers(self) -> dict[str, str]:
        """Deprecated compatibility mapping of namespace name -> endpoint."""
        warnings.warn(
            "UnifiedToolEnvironment.tool_servers is deprecated; use list_tool_namespaces() "
            "instead. This compatibility property will be removed in 0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            namespace.name: namespace.endpoint or "" for namespace in self.list_tool_namespaces()
        }

    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        """Fetch tool definitions from one or all configured tool servers."""
        results = await self._alist_http_tools(tool_server)
        results.extend(await self._alist_mcp_tools(tool_server))
        return results

    async def _alist_http_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_http_tools_sync, tool_server)

    def _list_http_tools_sync(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        server_names = [tool_server] if tool_server else list(self._http_namespace_endpoints.keys())
        results: list[dict[str, Any]] = []
        for name in server_names:
            if name is None or name not in self._http_namespace_endpoints:
                continue
            url = self._http_namespace_endpoints[name]
            req = urllib.request.Request(f"{url}/tools")  # noqa: S310
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:  # noqa: S310
                    payload = json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                continue
            tools = payload.get("tools", []) if isinstance(payload, dict) else []
            results.extend(
                [
                    {"tool_server": name, "transport": "http", **t}
                    for t in tools
                    if isinstance(t, dict)
                ]
            )
        return results

    async def _alist_mcp_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        server_names = [tool_server] if tool_server else list(self._mcp_clients.keys())
        results: list[dict[str, Any]] = []
        failures: list[str] = []
        for name in server_names:
            if name is None or name not in self._mcp_clients:
                continue
            try:
                tools = await self._mcp_clients[name].alist_tools()
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                continue
            results.extend(
                [
                    {"tool_server": name, "transport": "mcp", **tool}
                    for tool in tools
                    if isinstance(tool, dict)
                ]
            )
        if failures:
            failures_text = "; ".join(failures)
            raise RuntimeError(
                f"MCP tool discovery failed for configured server(s): {failures_text}"
            )
        return results

    async def acall_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        """Invoke a tool on the given tool server."""
        if tool_server in self._mcp_clients:
            return await self._mcp_clients[tool_server].acall_tool(tool_name, parameters)
        return await self._acall_http_tool(tool_server, tool_name, parameters)

    async def _acall_http_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        return await asyncio.to_thread(
            self._call_http_tool_sync,
            tool_server,
            tool_name,
            parameters,
        )

    def _call_http_tool_sync(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        """Invoke a tool on the given HTTP tool server via POST to /step."""
        if tool_server not in self._http_namespace_endpoints:
            return ToolCallResult(observation=f"Unknown tool server: {tool_server}", is_error=True)
        url = self._http_namespace_endpoints[tool_server]
        payload = json.dumps(
            {"action": {"tool_name": tool_name, "parameters": parameters}}
        ).encode()
        req = urllib.request.Request(  # noqa: S310
            f"{url}/step",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:  # noqa: S310
                return ToolCallResult(observation=json.loads(resp.read()), is_error=False)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return ToolCallResult(observation=str(exc), is_error=True)


class HttpToolEnvironment(UnifiedToolEnvironment):
    """Deprecated compatibility alias for UnifiedToolEnvironment.

    Deprecated: use ``UnifiedToolEnvironment`` instead. Will be removed in 0.4.0.
    """

    def __init__(
        self,
        tool_servers: dict[str, str],
        timeout_seconds: float = 30.0,
        mcp_clients: dict[str, MCPClientHandle] | None = None,
    ) -> None:
        """Initialize the deprecated compatibility alias."""
        warnings.warn(
            "HttpToolEnvironment is deprecated; use UnifiedToolEnvironment instead. "
            "This compatibility alias will be removed in 0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            tool_servers=tool_servers,
            timeout_seconds=timeout_seconds,
            mcp_clients=mcp_clients,
        )
