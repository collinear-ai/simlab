# ruff: noqa: D100, D101, D102, D103, S101

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from claude_agent_sdk import AssistantMessage
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import TextBlock
from claude_agent_sdk_cookbook import simlab_adapter
from claude_agent_sdk_cookbook.custom_agent import run_custom_agent
from simlab.agents import BaseEnvironment
from simlab.agents import RunArtifacts
from simlab.agents import ToolCallResult
from simlab.agents import ToolNamespace
from simlab.agents.loader import load_agent_class


class FakeEnvironment(BaseEnvironment):
    def list_tool_namespaces(self) -> list[ToolNamespace]:
        return []

    async def alist_tools(
        self, tool_server: str | None = None
    ) -> list[dict[str, object]]:
        _ = tool_server
        return []

    async def acall_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, object],
    ) -> ToolCallResult:
        _ = tool_server, tool_name, parameters
        return ToolCallResult(observation={})


def test_simlab_agent_records_metadata_and_final_output(monkeypatch) -> None:
    monkeypatch.setattr(
        simlab_adapter,
        "build_claude_agent_tools",
        lambda environment, recorder: (
            {},
            ["mcp__email-env__search_emails", "mcp__demo__ping"],
        ),
    )

    async def _fake_run(**_: object) -> str:
        return "final answer"

    monkeypatch.setattr(simlab_adapter, "run_custom_agent", _fake_run)

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    agent = simlab_adapter.SimLabClaudeAgentSDKAgent()
    agent.run("triage inbox", FakeEnvironment(), artifacts)

    assert artifacts.metadata["cookbook_agent"] == {
        "name": "claude-agent-sdk",
        "tool_count": 2,
        "tool_names": ["mcp__email-env__search_emails", "mcp__demo__ping"],
    }
    assert artifacts.final_observation == "final answer"
    assert artifacts.messages[0]["role"] == "user"
    assert artifacts.messages[0]["content"] == "triage inbox"
    assert artifacts.messages[-1]["role"] == "assistant"
    assert artifacts.messages[-1]["content"] == "final answer"


def test_simlab_agent_errors_on_empty_final_output(monkeypatch) -> None:
    monkeypatch.setattr(
        simlab_adapter,
        "build_claude_agent_tools",
        lambda environment, recorder: ({}, []),
    )

    async def _fake_run(**_: object) -> str:
        return ""

    monkeypatch.setattr(simlab_adapter, "run_custom_agent", _fake_run)

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    simlab_adapter.SimLabClaudeAgentSDKAgent().run(
        "triage inbox",
        FakeEnvironment(),
        artifacts,
    )

    assert artifacts.error == "Claude Agent SDK agent produced no final output"


def test_simlab_agent_wraps_runtime_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        simlab_adapter,
        "build_claude_agent_tools",
        lambda environment, recorder: ({}, []),
    )

    async def _raise(**_: object) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(simlab_adapter, "run_custom_agent", _raise)

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    simlab_adapter.SimLabClaudeAgentSDKAgent().run(
        "triage inbox",
        FakeEnvironment(),
        artifacts,
    )

    assert artifacts.error == "Claude Agent SDK agent failed: boom"


def test_agent_import_path_loads_the_adapter_class() -> None:
    agent_cls = load_agent_class(
        "claude_agent_sdk_cookbook.simlab_adapter:SimLabClaudeAgentSDKAgent"
    )

    assert agent_cls is simlab_adapter.SimLabClaudeAgentSDKAgent


# ---------------------------------------------------------------------------
# run_custom_agent — ResultMessage terminal handling
# ---------------------------------------------------------------------------

_COMMON_RESULT_FIELDS = {
    "subtype": "result",
    "duration_ms": 100,
    "duration_api_ms": 80,
    "num_turns": 1,
    "session_id": "test-session",
}


def _make_result_message(**overrides: Any) -> ResultMessage:
    fields = {**_COMMON_RESULT_FIELDS, "is_error": False, "result": None, **overrides}
    return ResultMessage(**fields)


async def _run(messages: list[Any]) -> str:
    async def _fake_query(**_: Any) -> Any:
        for m in messages:
            yield m

    with patch("claude_agent_sdk_cookbook.custom_agent.query", _fake_query):
        return await run_custom_agent(
            instruction="test",
            mcp_servers={},
            allowed_tools=[],
        )


class TestRunCustomAgentResultMessage:
    def test_result_with_string(self) -> None:
        result = asyncio.run(_run([_make_result_message(result="done")]))
        assert result == "done"

    def test_result_none_with_structured_output(self) -> None:
        msg = _make_result_message(result=None)
        msg.structured_output = {"key": "value"}
        result = asyncio.run(_run([msg]))
        assert "key" in result
        assert "value" in result

    def test_result_none_no_structured_output_uses_assistant_text(self) -> None:
        assistant = AssistantMessage(
            content=[TextBlock(text="assistant fallback")],
            model="test",
        )
        result_msg = _make_result_message(result=None)
        result = asyncio.run(_run([assistant, result_msg]))
        assert result == "assistant fallback"

    def test_error_with_result_string(self) -> None:
        msg = _make_result_message(is_error=True, result="bad request")
        with pytest.raises(RuntimeError, match="bad request"):
            asyncio.run(_run([msg]))

    def test_error_with_errors_list(self) -> None:
        msg = _make_result_message(is_error=True, result=None)
        msg.errors = ["boom", "crash"]
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(_run([msg]))

    def test_error_no_result_no_errors(self) -> None:
        msg = _make_result_message(is_error=True, result=None)
        with pytest.raises(RuntimeError, match="unknown error"):
            asyncio.run(_run([msg]))

    def test_assistant_text_blocks_accumulated(self) -> None:
        a1 = AssistantMessage(
            content=[TextBlock(text="first")],
            model="test",
        )
        a2 = AssistantMessage(
            content=[TextBlock(text="second")],
            model="test",
        )
        result_msg = _make_result_message(result=None)
        result = asyncio.run(_run([a1, a2, result_msg]))
        assert result == "second"
