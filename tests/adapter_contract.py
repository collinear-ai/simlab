"""Shared behavioral contract for SimLab agent adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol

import pytest
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.base import BaseEnvironment
from simlab.agents.base import RunArtifacts
from simlab.agents.base import ToolCallResult
from simlab.agents.base import ToolNamespace


class ToolAdapterHarness(Protocol):
    """Test-only harness for exercising one framework adapter."""

    def build_tools(
        self,
        environment: BaseEnvironment,
        *,
        recorder: RunArtifactsRecorder | None = None,
    ) -> list[object]:
        """Build framework-native tools from a SimLab environment."""

    def tool_name(self, tool: object) -> str:
        """Return the public tool name exposed to the framework."""

    def invoke(self, tool: object, payload: dict[str, Any]) -> object:
        """Invoke one framework-native tool and return the observation."""


@dataclass
class ContractEnvironment(BaseEnvironment):
    """Fixture environment shared across adapter contract tests."""

    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        _ = tool_server
        return [
            {
                "tool_server": "email-env",
                "name": "search_emails",
                "description": "Search email",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
                "transport": "http",
            },
            {
                "tool_server": "demo",
                "name": "ping",
                "description": "Ping the demo MCP server",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                    },
                },
                "transport": "mcp",
            },
        ]

    async def acall_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        return ToolCallResult(
            observation={
                "tool_server": tool_server,
                "tool_name": tool_name,
                "parameters": parameters,
            }
        )

    def list_tool_namespaces(self) -> list[ToolNamespace]:
        return [
            ToolNamespace(name="email-env", transport="http", endpoint="http://localhost:8040"),
            ToolNamespace(name="demo", transport="mcp", endpoint="http://localhost:8081/mcp"),
        ]


class InvalidNestedSchemaEnvironment(ContractEnvironment):
    """Fixture environment exposing a nested schema OpenAI rejects today."""

    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        _ = tool_server
        return [
            {
                "tool_server": "edgar-mcp",
                "name": "get_key_metrics",
                "description": "Get key metrics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {
                                "properties": {
                                    "name": {"type": "string"},
                                }
                            },
                        }
                    },
                },
                "transport": "mcp",
            }
        ]


class EmptyItemsSchemaEnvironment(ContractEnvironment):
    """Fixture matching the live EDGAR schema with ``items: {}``."""

    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        _ = tool_server
        return [
            {
                "tool_server": "edgar-mcp",
                "name": "get_key_metrics",
                "description": "Get key metrics",
                "input_schema": {
                    "type": "object",
                    "title": "get_key_metricsArguments",
                    "properties": {
                        "identifier": {
                            "title": "Identifier",
                            "type": "string",
                        },
                        "metrics": {
                            "title": "Metrics",
                            "type": "array",
                            "default": None,
                            "items": {},
                        },
                    },
                    "required": ["identifier"],
                },
                "transport": "mcp",
            }
        ]


class EmptyPropertiesSchemaEnvironment(ContractEnvironment):
    """Fixture matching tools that legitimately expose ``properties: {}``."""

    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        _ = tool_server
        return [
            {
                "tool_server": "email",
                "name": "delete_all_emails",
                "description": "Delete all emails",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                "transport": "http",
            }
        ]


class DuplicateWireNameEnvironment(ContractEnvironment):
    """Fixture exposing colliding framework-facing tool names."""

    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        _ = tool_server
        return [
            {
                "tool_server": "email-env",
                "name": "send_email",
                "description": "Send email",
                "input_schema": {"type": "object"},
            },
            {
                "tool_server": "email-env",
                "name": "send_email",
                "description": "Duplicate send email",
                "input_schema": {"type": "object"},
            },
        ]


def assert_exposes_demo_tools(harness: ToolAdapterHarness) -> None:
    """Shared contract: adapters expose stable wire names across transports."""
    recorder = RunArtifactsRecorder(RunArtifacts(task_id="task-1", task="demo"))
    tools = harness.build_tools(ContractEnvironment(), recorder=recorder)
    assert [harness.tool_name(tool) for tool in tools] == [
        "email-env__search_emails",
        "demo__ping",
    ]


def assert_records_tool_invocation(harness: ToolAdapterHarness) -> None:
    """Shared contract: tool execution records artifacts using the common shape."""
    artifacts = RunArtifacts(task_id="task-1", task="demo")
    recorder = RunArtifactsRecorder(artifacts)
    tools = harness.build_tools(ContractEnvironment(), recorder=recorder)

    output = harness.invoke(tools[1], {"value": 1})

    if isinstance(output, str):
        assert '"tool_name": "ping"' in output
    else:
        assert isinstance(output, dict)
        assert output["tool_name"] == "ping"
    assert artifacts.tool_calls[0].tool_server == "demo"
    assert artifacts.tool_results[0].observation["parameters"] == {"value": 1}
    assert artifacts.messages[0]["role"] == "assistant"
    assert artifacts.messages[1]["role"] == "tool"


def assert_rejects_duplicate_wire_names(harness: ToolAdapterHarness) -> None:
    """Shared contract: adapters reject ambiguous framework-facing tool names."""

    with pytest.raises(RuntimeError, match="Duplicate tool wire name detected"):
        harness.build_tools(DuplicateWireNameEnvironment())
