"""Claude Agent SDK tool adapters for SimLab environments."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from simlab.agents.adapters.artifacts import ToolEventRecorder
from simlab.agents.adapters.core import ToolDescriptor
from simlab.agents.adapters.core import build_tool_dispatch
from simlab.agents.adapters.core import list_tool_descriptors
from simlab.agents.adapters.core import stringify_observation
from simlab.agents.base import BaseEnvironment
from simlab.agents.base import ToolCallResult

_tool_decorator: Any
_create_sdk_mcp_server_fn: Any
try:
    from claude_agent_sdk import create_sdk_mcp_server
    from claude_agent_sdk import tool as _tool_dec
except (
    ImportError,
    ModuleNotFoundError,
):  # pragma: no cover - exercised when optional deps are absent
    _tool_decorator = None
    _create_sdk_mcp_server_fn = None
else:  # pragma: no cover - exercised when optional deps are present
    _tool_decorator = _tool_dec
    _create_sdk_mcp_server_fn = create_sdk_mcp_server


@dataclass
class _ToolRuntime:
    """Runtime state shared across Claude SDK tool wrappers."""

    environment: BaseEnvironment
    recorder: ToolEventRecorder | None = None
    counter: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    def next_call_id(self) -> str:
        """Return a deterministic tool call id for artifact recording."""
        return f"call_{next(self.counter)}"


def _build_claude_tool_handler(
    descriptor: ToolDescriptor,
    runtime: _ToolRuntime,
) -> object:
    """Build a ``@tool``-decorated async handler for one SimLab tool."""

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        tool_call_id = runtime.next_call_id()
        if runtime.recorder is not None:
            runtime.recorder.on_assistant_tool_call(
                tool_call_id,
                descriptor.tool_server,
                descriptor.tool_name,
                args,
            )
        result: ToolCallResult = await runtime.environment.acall_tool(
            descriptor.tool_server,
            descriptor.tool_name,
            args,
        )
        if runtime.recorder is not None:
            runtime.recorder.on_tool_invocation(
                tool_call_id,
                descriptor.tool_server,
                descriptor.tool_name,
                args,
                ToolCallResult(
                    observation=result.observation,
                    is_error=result.is_error,
                ),
            )
        text = stringify_observation(result.observation)
        payload: dict[str, Any] = {
            "content": [{"type": "text", "text": text}],
        }
        if result.is_error:
            payload["is_error"] = True
        return payload

    return _tool_decorator(
        name=descriptor.tool_name,
        description=descriptor.description,
        input_schema=descriptor.input_schema,
    )(_handler)


def build_claude_agent_tools(
    environment: BaseEnvironment,
    *,
    recorder: ToolEventRecorder | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Build Claude Agent SDK MCP servers and allowed-tools list from a SimLab environment.

    Returns ``(mcp_servers, allowed_tools)`` where *mcp_servers* maps server
    names to ``McpSdkServerConfig`` instances and *allowed_tools* lists the
    ``mcp__<server>__<tool>`` identifiers the agent should be permitted to call.
    """
    if _tool_decorator is None or _create_sdk_mcp_server_fn is None:
        msg = "claude-agent-sdk is required to build Claude Agent SDK tools"
        raise ModuleNotFoundError(msg)

    runtime = _ToolRuntime(environment=environment, recorder=recorder)
    descriptors = list_tool_descriptors(environment)
    _ = build_tool_dispatch(descriptors)

    # Group tools by tool_server so each becomes one MCP server.
    server_tools: dict[str, list[Any]] = defaultdict(list)
    for descriptor in descriptors:
        handler = _build_claude_tool_handler(descriptor, runtime)
        server_tools[descriptor.tool_server].append(handler)

    mcp_servers: dict[str, Any] = {}
    allowed_tools: list[str] = []
    for server_name, tools in server_tools.items():
        mcp_servers[server_name] = _create_sdk_mcp_server_fn(
            name=server_name,
            version="1.0.0",
            tools=tools,
        )
        allowed_tools.extend(
            f"mcp__{server_name}__{d.tool_name}"
            for d in descriptors
            if d.tool_server == server_name
        )

    return mcp_servers, allowed_tools
