# ruff: noqa: D100, D101, D102, D103, S101, SLF001

from __future__ import annotations

import json
from typing import Any

from langgraph_email_agent import simlab_adapter
from simlab.agents import BaseEnvironment
from simlab.agents import RunArtifacts
from simlab.agents import ToolNamespace
from simlab.agents import ToolCallResult
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.langchain import build_langchain_tools


class FakeEnvironment(BaseEnvironment):
    def list_tool_namespaces(self) -> list[ToolNamespace]:
        return [
            ToolNamespace(
                name="email-env", transport="http", endpoint="http://localhost:8040"
            ),
            ToolNamespace(
                name="demo", transport="mcp", endpoint="http://localhost:8081/mcp"
            ),
        ]

    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        _ = tool_server
        return [
            {
                "tool_server": "email-env",
                "name": "search_emails",
                "description": "Search email",
                "input_schema": {"type": "object"},
                "transport": "http",
            },
            {
                "tool_server": "demo",
                "name": "ping",
                "description": "Ping the demo MCP server",
                "input_schema": {"type": "object"},
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


def test_simlab_tool_adapter_exposes_http_and_mcp_tools() -> None:
    recorder = RunArtifactsRecorder(RunArtifacts(task_id="task-1", task="demo"))
    tools = build_langchain_tools(FakeEnvironment(), recorder=recorder)
    assert {tool.name for tool in tools} == {
        "email-env__search_emails",
        "demo__ping",
    }


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
    assert artifacts.tool_calls[0].tool_server == "demo"
    assert artifacts.tool_results[0].observation == {"ok": True}
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
    assert artifacts.messages[2]["content"] == {
        "tool_call_id": "call_1",
        "tool_server": "demo",
        "tool_name": "ping",
        "is_error": False,
    }
    assert artifacts.messages[-1]["role"] == "assistant"


def test_simlab_agent_records_workflows_in_metadata(monkeypatch) -> None:
    class StubAssistant:
        def __init__(
            self,
            model: object,
            tool_recursion_limit: int,
            workflow_recursion_limit: int,
            mailbox_owner: str | None = None,
        ) -> None:
            _ = model, mailbox_owner
            assert tool_recursion_limit == 8
            assert workflow_recursion_limit == 16

        def workflow_names(self) -> list[str]:
            return [
                "inbox-triage",
                "todo-builder",
                "sales-call-brief",
                "meeting-prep-packet",
            ]

        def run(self, task: str, tools: object, recorder: object) -> object:
            _ = task, tools, recorder

            class Result:
                final_output = "## Inbox Triage\nok\n## Todo List\n- [P1] follow up\n## Sales Call Brief\nok"

            return Result()

    monkeypatch.setattr(simlab_adapter, "build_chat_model_from_env", lambda: object())
    monkeypatch.setattr(simlab_adapter, "LangGraphEmailAssistant", StubAssistant)

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    agent = simlab_adapter.SimLabLangGraphEmailAgent()
    agent.run("triage inbox", FakeEnvironment(), artifacts)

    cookbook_agent = artifacts.metadata["cookbook_agent"]
    assert cookbook_agent["name"] == "langgraph-email-agent"
    assert cookbook_agent["workflows"] == [
        "inbox-triage",
        "todo-builder",
        "sales-call-brief",
        "meeting-prep-packet",
    ]
