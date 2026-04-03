from __future__ import annotations

import json
from typing import Any
from typing import Protocol
from typing import cast

import pytest

pytest.importorskip("langchain_core")

from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.langchain import build_langchain_tools
from simlab.agents.base import RunArtifacts
from simlab.agents.base import ToolCallResult

from tests.adapter_contract import ContractEnvironment
from tests.adapter_contract import ToolAdapterHarness
from tests.adapter_contract import assert_exposes_demo_tools
from tests.adapter_contract import assert_records_tool_invocation
from tests.adapter_contract import assert_rejects_duplicate_wire_names


class LangChainHarness(ToolAdapterHarness):
    def build_tools(self, environment, *, recorder=None) -> list[object]:  # noqa: ANN001
        return build_langchain_tools(environment, recorder=recorder)

    def tool_name(self, tool: object) -> str:
        return cast(LangChainToolLike, tool).name

    def invoke(self, tool: object, payload: dict[str, Any]) -> str:
        return cast(LangChainToolLike, tool).invoke(payload)


class LangChainToolLike(Protocol):
    name: str

    def invoke(self, payload: dict[str, Any]) -> str: ...


def test_build_langchain_tools_exposes_http_and_mcp_tools() -> None:
    assert_exposes_demo_tools(LangChainHarness())


def test_langchain_tool_invocation_records_artifacts() -> None:
    assert_records_tool_invocation(LangChainHarness())


def test_langchain_tools_reject_duplicate_wire_names() -> None:
    assert_rejects_duplicate_wire_names(LangChainHarness())


def test_run_artifacts_recorder_records_tool_calls() -> None:
    artifacts = RunArtifacts(task_id="task-1", task="demo")
    recorder = RunArtifactsRecorder(artifacts)

    recorder.on_user_message("summarize inbox")
    recorder.on_assistant_tool_call(
        "call_1",
        "demo",
        "ping",
        {"value": 1},
    )
    recorder.on_tool_invocation(
        "call_1",
        "demo",
        "ping",
        {"value": 1},
        ToolCallResult(observation={"ok": True}),
    )
    recorder.on_assistant_message("## Email Summary\n...\n## Todo List\n...")

    assert artifacts.messages[0]["role"] == "user"
    assert artifacts.messages[1]["role"] == "assistant"
    assert artifacts.messages[1]["content"] == {
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "demo__ping",
                    "arguments": json.dumps({"value": 1}, sort_keys=True),
                },
            }
        ],
    }
    assert isinstance(artifacts.messages[1]["timestamp"], str)
    assert artifacts.tool_calls[0].tool_server == "demo"
    assert artifacts.tool_results[0].observation == {"ok": True}
    assert artifacts.messages[2]["role"] == "tool"
    assert artifacts.messages[2]["content"] == {
        "tool_call_id": "call_1",
        "tool_server": "demo",
        "tool_name": "ping",
        "is_error": False,
    }
    assert artifacts.messages[-1]["role"] == "assistant"


def test_langchain_tools_accept_nested_schema_without_crashing() -> None:
    tools = build_langchain_tools(ContractEnvironment())
    result = tools[1].invoke({"value": 2})
    parsed = json.loads(result)

    assert parsed["tool_name"] == "ping"
    assert parsed["parameters"] == {"value": 2}
