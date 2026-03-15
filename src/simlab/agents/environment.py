"""Concrete environments for CLI external-agent runs."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from simlab.agents.base import BaseEnvironment
from simlab.agents.base import ToolCallResult


class HttpToolEnvironment(BaseEnvironment):
    """HTTP implementation against RL-Gym style `/tools` and `/step` endpoints."""

    def __init__(self, tool_servers: dict[str, str], timeout_seconds: float = 30.0) -> None:
        """Configure tool server URLs and request timeout."""
        self._tool_servers = dict(tool_servers)
        self._timeout_seconds = timeout_seconds

    @property
    def tool_servers(self) -> dict[str, str]:
        """Return a copy of the tool server name -> base URL mapping."""
        return dict(self._tool_servers)

    def list_tools(self, tool_server: str | None = None) -> list[dict]:
        """Fetch tool definitions from one or all configured tool servers."""
        server_names = [tool_server] if tool_server else list(self._tool_servers.keys())
        results: list[dict] = []
        for name in server_names:
            if name is None or name not in self._tool_servers:
                continue
            url = self._tool_servers[name]
            req = urllib.request.Request(f"{url}/tools")  # noqa: S310
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:  # noqa: S310
                    payload = json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                continue
            tools = payload.get("tools", []) if isinstance(payload, dict) else []
            results.extend([{"tool_server": name, **t} for t in tools if isinstance(t, dict)])
        return results

    def call_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict,
    ) -> ToolCallResult:
        """Invoke a tool on the given tool server via HTTP POST to /step."""
        if tool_server not in self._tool_servers:
            return ToolCallResult(observation=f"Unknown tool server: {tool_server}", is_error=True)
        url = self._tool_servers[tool_server]
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
