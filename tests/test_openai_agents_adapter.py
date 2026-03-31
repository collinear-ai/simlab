from __future__ import annotations

import asyncio
import json
from typing import Any
from typing import Protocol
from typing import cast

import pytest
from simlab.agents import RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters import openai_agents
from simlab.agents.base import ToolCallResult

from tests.adapter_contract import ContractEnvironment
from tests.adapter_contract import EmptyItemsSchemaEnvironment
from tests.adapter_contract import EmptyPropertiesSchemaEnvironment
from tests.adapter_contract import InvalidNestedSchemaEnvironment
from tests.adapter_contract import ToolAdapterHarness
from tests.adapter_contract import assert_exposes_demo_tools
from tests.adapter_contract import assert_records_tool_invocation
from tests.adapter_contract import assert_rejects_duplicate_wire_names


class FakeFunctionTool:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        params_json_schema: dict[str, Any],
        on_invoke_tool: object,
        strict_json_schema: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        self.params_json_schema = params_json_schema
        self.on_invoke_tool = on_invoke_tool
        self.strict_json_schema = strict_json_schema


class OpenAIAgentsToolLike(Protocol):
    name: str
    params_json_schema: dict[str, Any]

    async def on_invoke_tool(self, ctx: object, args: str) -> str: ...


class OpenAIAgentsHarness(ToolAdapterHarness):
    def build_tools(self, environment, *, recorder=None) -> list[object]:  # noqa: ANN001
        return openai_agents.build_openai_agents_tools(environment, recorder=recorder)

    def tool_name(self, tool: object) -> str:
        return cast(OpenAIAgentsToolLike, tool).name

    def invoke(self, tool: object, payload: dict[str, Any]) -> str:
        return asyncio.run(
            cast(OpenAIAgentsToolLike, tool).on_invoke_tool(None, json.dumps(payload))
        )


class ValueKeywordSchemaEnvironment(ContractEnvironment):
    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        _ = tool_server
        return [
            {
                "tool_server": "demo",
                "name": "configure_defaults",
                "description": "Configure defaults",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "object",
                            "default": {},
                            "examples": [{}],
                        }
                    },
                },
            }
        ]

    async def acall_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        _ = tool_server, tool_name, parameters
        return ToolCallResult(observation="ok")


class FailingToolEnvironment(ContractEnvironment):
    async def acall_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        _ = tool_server, tool_name, parameters
        raise RuntimeError("tool transport failed")


def test_build_openai_agents_tools_exposes_http_and_mcp_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_agents, "_function_tool_class", FakeFunctionTool)
    assert_exposes_demo_tools(OpenAIAgentsHarness())


def test_openai_agents_tool_invocation_records_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_agents, "_function_tool_class", FakeFunctionTool)
    assert_records_tool_invocation(OpenAIAgentsHarness())


def test_openai_agents_tool_invocation_coerces_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_agents, "_function_tool_class", FakeFunctionTool)

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    recorder = RunArtifactsRecorder(artifacts)
    tools = openai_agents.build_openai_agents_tools(ContractEnvironment(), recorder=recorder)

    asyncio.run(cast(OpenAIAgentsToolLike, tools[0]).on_invoke_tool(None, "{not-json"))

    assert artifacts.tool_calls[0].parameters == {}


def test_openai_agents_tool_invocation_propagates_environment_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_agents, "_function_tool_class", FakeFunctionTool)

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    recorder = RunArtifactsRecorder(artifacts)
    tools = openai_agents.build_openai_agents_tools(FailingToolEnvironment(), recorder=recorder)

    with pytest.raises(RuntimeError, match="tool transport failed"):
        asyncio.run(cast(OpenAIAgentsToolLike, tools[0]).on_invoke_tool(None, "{}"))

    assert len(artifacts.tool_calls) == 0
    assert len(artifacts.tool_results) == 0


def test_openai_agents_tools_normalize_invalid_nested_array_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_agents, "_function_tool_class", FakeFunctionTool)
    tools = openai_agents.build_openai_agents_tools(InvalidNestedSchemaEnvironment())

    assert cast(OpenAIAgentsToolLike, tools[0]).params_json_schema["properties"]["metrics"][
        "items"
    ] == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
    }


def test_openai_agents_tools_normalize_empty_array_item_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_agents, "_function_tool_class", FakeFunctionTool)
    tools = openai_agents.build_openai_agents_tools(EmptyItemsSchemaEnvironment())

    assert cast(OpenAIAgentsToolLike, tools[0]).params_json_schema["properties"]["metrics"][
        "items"
    ] == {
        "type": "object",
    }


def test_openai_agents_tools_preserve_empty_properties_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_agents, "_function_tool_class", FakeFunctionTool)
    tools = openai_agents.build_openai_agents_tools(EmptyPropertiesSchemaEnvironment())

    assert cast(OpenAIAgentsToolLike, tools[0]).params_json_schema["properties"] == {}


def test_openai_agents_tools_disable_sdk_strict_schema_rewriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_agents, "_function_tool_class", FakeFunctionTool)
    tools = openai_agents.build_openai_agents_tools(EmptyItemsSchemaEnvironment())

    assert cast(FakeFunctionTool, tools[0]).strict_json_schema is False


def test_openai_agents_tools_raise_on_duplicate_wire_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_agents, "_function_tool_class", FakeFunctionTool)
    assert_rejects_duplicate_wire_names(OpenAIAgentsHarness())


def test_openai_agents_tools_preserve_non_schema_value_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_agents, "_function_tool_class", FakeFunctionTool)
    tools = openai_agents.build_openai_agents_tools(ValueKeywordSchemaEnvironment())

    payload_schema = cast(OpenAIAgentsToolLike, tools[0]).params_json_schema["properties"][
        "payload"
    ]
    assert payload_schema["default"] == {}
    assert payload_schema["examples"] == [{}]
