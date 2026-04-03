from __future__ import annotations

import asyncio
from typing import Any
from typing import cast

import pytest
from simlab.agents.adapters import claude_agent
from simlab.agents.base import ToolCallResult

from tests.adapter_contract import ContractEnvironment
from tests.adapter_contract import ToolAdapterHarness
from tests.adapter_contract import assert_exposes_demo_tools
from tests.adapter_contract import assert_records_tool_invocation
from tests.adapter_contract import assert_rejects_duplicate_wire_names

# ---------------------------------------------------------------------------
# Fakes — stand in for the real claude_agent_sdk types without the dependency
# ---------------------------------------------------------------------------

_registered_tools: list[FakeToolDef] = []


class FakeToolDef:
    """Minimal fake for a @tool-decorated handler."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler


class FakeMcpServerConfig:
    """Minimal fake for McpSdkServerConfig."""

    def __init__(self, name: str, tools: list[FakeToolDef]) -> None:
        self.name = name
        self.tools = tools


def fake_tool_decorator(
    name: str,
    description: str,
    input_schema: Any,
) -> Any:
    """Replacement for ``claude_agent_sdk.tool``."""

    def _wrap(fn: Any) -> FakeToolDef:
        return FakeToolDef(
            name=name,
            description=description,
            input_schema=input_schema if isinstance(input_schema, dict) else {},
            handler=fn,
        )

    return _wrap


def fake_create_sdk_mcp_server(
    *,
    name: str,
    version: str,  # noqa: ARG001
    tools: list[Any],
) -> FakeMcpServerConfig:
    """Replacement for ``claude_agent_sdk.create_sdk_mcp_server``."""
    return FakeMcpServerConfig(name=name, tools=tools)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class ClaudeAgentToolLike:
    """Structural type for invoking fake tool handlers."""

    name: str
    handler: Any


class ClaudeAgentHarness(ToolAdapterHarness):
    def build_tools(self, environment: Any, *, recorder: Any = None) -> list[object]:
        mcp_servers, _ = claude_agent.build_claude_agent_tools(environment, recorder=recorder)
        # Flatten all tools from all MCP servers
        tools: list[object] = []
        for server in mcp_servers.values():
            tools.extend(cast(FakeMcpServerConfig, server).tools)
        return tools

    def tool_name(self, tool: object) -> str:
        fake = cast(FakeToolDef, tool)
        # Reconstruct wire_name from the server the tool belongs to
        # The tool name alone is not unique; for contract tests we need
        # the full wire name, so we search across servers.
        return self._find_wire_name(fake)

    def invoke(self, tool: object, payload: dict[str, Any]) -> str:
        fake = cast(FakeToolDef, tool)
        result = asyncio.run(fake.handler(payload))
        # Extract text from the MCP response format
        content = result.get("content", [])
        if content:
            return content[0].get("text", "")
        return ""

    def _find_wire_name(self, tool: FakeToolDef) -> str:
        """Find the wire name by looking up which server owns this tool."""
        # We stored the server name in the MCP config; for the harness
        # we need to track this. Use a class-level cache.
        return getattr(tool, "_wire_name", tool.name)


@pytest.fixture(autouse=True)
def _patch_claude_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace Claude SDK imports with fakes for all tests in this module."""
    monkeypatch.setattr(claude_agent, "_tool_decorator", fake_tool_decorator)
    monkeypatch.setattr(claude_agent, "_create_sdk_mcp_server_fn", fake_create_sdk_mcp_server)


# ---------------------------------------------------------------------------
# We need to override the harness to track wire names since tools are grouped
# by server and the tool.name is just the tool_name portion.
# ---------------------------------------------------------------------------


class WireNameTrackingHarness(ToolAdapterHarness):
    """Harness that tracks wire names by pairing server name with tool name."""

    def build_tools(self, environment: Any, *, recorder: Any = None) -> list[object]:
        mcp_servers, _ = claude_agent.build_claude_agent_tools(environment, recorder=recorder)
        tools: list[object] = []
        for server in mcp_servers.values():
            server_cfg = cast(FakeMcpServerConfig, server)
            for tool in server_cfg.tools:
                # Tag each tool with its wire name for lookup
                tool._wire_name = f"{server_cfg.name}__{tool.name}"  # type: ignore[attr-defined]
                tools.append(tool)
        return tools

    def tool_name(self, tool: object) -> str:
        return cast(FakeToolDef, tool)._wire_name  # type: ignore[attr-defined]

    def invoke(self, tool: object, payload: dict[str, Any]) -> str:
        fake = cast(FakeToolDef, tool)
        result = asyncio.run(fake.handler(payload))
        content = result.get("content", [])
        if content:
            return content[0].get("text", "")
        return ""


# ---------------------------------------------------------------------------
# Shared contract tests
# ---------------------------------------------------------------------------


def test_build_claude_agent_tools_exposes_http_and_mcp_tools() -> None:
    assert_exposes_demo_tools(WireNameTrackingHarness())


def test_claude_agent_tool_invocation_records_artifacts() -> None:
    assert_records_tool_invocation(WireNameTrackingHarness())


def test_claude_agent_tools_reject_duplicate_wire_names() -> None:
    assert_rejects_duplicate_wire_names(WireNameTrackingHarness())


# ---------------------------------------------------------------------------
# Claude-specific tests
# ---------------------------------------------------------------------------


def test_tools_grouped_by_tool_server() -> None:
    mcp_servers, _ = claude_agent.build_claude_agent_tools(ContractEnvironment())
    assert set(mcp_servers.keys()) == {"email-env", "demo"}
    email_server = cast(FakeMcpServerConfig, mcp_servers["email-env"])
    demo_server = cast(FakeMcpServerConfig, mcp_servers["demo"])
    assert len(email_server.tools) == 1
    assert len(demo_server.tools) == 1


def test_allowed_tools_list_uses_mcp_naming() -> None:
    _, allowed_tools = claude_agent.build_claude_agent_tools(ContractEnvironment())
    assert "mcp__email-env__search_emails" in allowed_tools
    assert "mcp__demo__ping" in allowed_tools


def test_tool_handler_returns_mcp_format() -> None:
    mcp_servers, _ = claude_agent.build_claude_agent_tools(ContractEnvironment())
    demo_server = cast(FakeMcpServerConfig, mcp_servers["demo"])
    tool = demo_server.tools[0]
    result = asyncio.run(cast(FakeToolDef, tool).handler({"value": 42}))
    assert "content" in result
    assert result["content"][0]["type"] == "text"
    assert "is_error" not in result  # successful call has no is_error


def test_tool_handler_signals_error_via_is_error() -> None:
    class ErrorEnvironment(ContractEnvironment):
        async def acall_tool(
            self,
            tool_server: str,
            tool_name: str,
            parameters: dict[str, Any],
        ) -> ToolCallResult:
            _ = tool_server, tool_name, parameters
            return ToolCallResult(observation="something went wrong", is_error=True)

    mcp_servers, _ = claude_agent.build_claude_agent_tools(ErrorEnvironment())
    demo_server = cast(FakeMcpServerConfig, mcp_servers["demo"])
    tool = demo_server.tools[0]
    result = asyncio.run(cast(FakeToolDef, tool).handler({"value": 1}))
    assert result["is_error"] is True
    assert result["content"][0]["text"] == "something went wrong"


def test_tool_handler_propagates_environment_exceptions() -> None:
    class FailingEnvironment(ContractEnvironment):
        async def acall_tool(
            self,
            tool_server: str,
            tool_name: str,
            parameters: dict[str, Any],
        ) -> ToolCallResult:
            _ = tool_server, tool_name, parameters
            raise RuntimeError("transport failed")

    mcp_servers, _ = claude_agent.build_claude_agent_tools(FailingEnvironment())
    demo_server = cast(FakeMcpServerConfig, mcp_servers["demo"])
    tool = demo_server.tools[0]
    with pytest.raises(RuntimeError, match="transport failed"):
        asyncio.run(cast(FakeToolDef, tool).handler({}))


def test_tool_input_schema_passed_to_decorator() -> None:
    mcp_servers, _ = claude_agent.build_claude_agent_tools(ContractEnvironment())
    email_server = cast(FakeMcpServerConfig, mcp_servers["email-env"])
    tool = email_server.tools[0]
    assert tool.input_schema == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }
