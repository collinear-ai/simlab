from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import simlab.agents.mcp_client as mcp_client_module
from simlab.agents import BaseAgent
from simlab.agents import BaseEnvironment
from simlab.agents import ToolCallResult
from simlab.agents import load_agent_class
from simlab.agents import reference_agent
from simlab.agents import run_with_agent_contract


class FakeEnvironment(BaseEnvironment):
    @property
    def tool_servers(self) -> dict[str, str]:
        return {"email-env": "http://localhost:8040"}

    def list_tools(self, tool_server: str | None = None) -> list[dict]:
        _ = tool_server
        return [
            {
                "tool_server": "email-env",
                "name": "send_email",
                "description": "Send email",
                "input_schema": {"type": "object"},
            }
        ]

    def call_tool(self, tool_server: str, tool_name: str, parameters: dict) -> ToolCallResult:
        _ = tool_server, tool_name, parameters
        return ToolCallResult(observation={"ok": True}, is_error=False)


class RecordingAgent(BaseAgent):
    def setup(self, environment: BaseEnvironment) -> None:
        _ = environment

    def run(self, instruction: str, environment: BaseEnvironment, context) -> None:  # noqa: ANN001
        _ = environment
        context.record_message("user", instruction)
        context.final_observation = "done"
        context.metadata["custom"] = True


class SlowAgent(BaseAgent):
    def setup(self, environment: BaseEnvironment) -> None:
        _ = environment

    def run(self, instruction: str, environment: BaseEnvironment, context) -> None:  # noqa: ANN001
        _ = instruction, environment, context
        time.sleep(0.2)


class FailingAgent(BaseAgent):
    def setup(self, environment: BaseEnvironment) -> None:
        _ = environment

    def run(self, instruction: str, environment: BaseEnvironment, context) -> None:  # noqa: ANN001
        _ = instruction, environment, context
        raise RuntimeError("boom")


def test_run_with_agent_contract_success() -> None:
    artifacts = run_with_agent_contract(
        task_id="t-1",
        instruction="do task",
        model="gpt-test",
        provider="openai",
        max_steps=5,
        environment=FakeEnvironment(),
        agent_import_path=f"{__name__}:RecordingAgent",
        timeout_seconds=1.0,
    )
    assert artifacts.error is None
    assert artifacts.final_observation == "done"
    assert artifacts.metadata["custom"] is True
    assert artifacts.messages


def test_run_with_agent_contract_timeout() -> None:
    artifacts = run_with_agent_contract(
        task_id="t-2",
        instruction="do task",
        model="gpt-test",
        provider="openai",
        max_steps=5,
        environment=FakeEnvironment(),
        agent_import_path=f"{__name__}:SlowAgent",
        timeout_seconds=0.01,
    )
    assert artifacts.error == "Rollout timeout exceeded"
    assert artifacts.metadata["timeout"] is True


def test_run_with_agent_contract_error_capture() -> None:
    artifacts = run_with_agent_contract(
        task_id="t-3",
        instruction="do task",
        model="gpt-test",
        provider="openai",
        max_steps=5,
        environment=FakeEnvironment(),
        agent_import_path=f"{__name__}:FailingAgent",
        timeout_seconds=1.0,
    )
    assert artifacts.error is not None
    assert "boom" in artifacts.error


def test_load_agent_class_from_module_path(tmp_path: Path) -> None:
    mod = tmp_path / "custom_agent_module.py"
    mod.write_text(
        "\n".join(
            [
                "from simlab.agents import BaseAgent",
                "class TempAgent(BaseAgent):",
                "    def setup(self, environment):",
                "        _ = environment",
                "    def run(self, instruction, environment, context):",
                "        _ = instruction, environment",
                "        context.final_observation = 'ok'",
            ]
        )
    )
    sys.path.insert(0, str(tmp_path))
    try:
        cls = load_agent_class("custom_agent_module:TempAgent")
        assert issubclass(cls, BaseAgent)
    finally:
        sys.path.remove(str(tmp_path))


def test_reference_agent_runs_tool_loop(monkeypatch) -> None:  # noqa: ANN001
    def fake_completion(**kwargs):
        messages = kwargs["messages"]
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if not has_tool_result:
            tool_call = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="email-env__send_email",
                    arguments='{"to_addr":"a@example.com"}',
                ),
            )
            message = SimpleNamespace(content="", tool_calls=[tool_call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        message = SimpleNamespace(content="task complete", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    import litellm  # noqa: PLC0415

    monkeypatch.setattr(litellm, "completion", fake_completion)

    artifacts = run_with_agent_contract(
        task_id="t-ref",
        instruction="send an email",
        model="gpt-test",
        provider="openai",
        max_steps=5,
        environment=FakeEnvironment(),
        timeout_seconds=1.0,
        api_key="sk-test",
        base_url=None,
    )
    assert artifacts.error is None
    assert artifacts.final_observation == "task complete"
    assert artifacts.steps_taken == 1
    assert len(artifacts.tool_calls) == 1
    assistant_msgs = [m for m in artifacts.messages if m["role"] == "assistant"]
    tool_call_msg = next(
        (
            m
            for m in assistant_msgs
            if isinstance(m["content"], dict) and m["content"].get("tool_calls")
        ),
        None,
    )
    assert tool_call_msg is not None
    assert tool_call_msg["content"]["tool_calls"][0]["name"] == "email-env__send_email"
    assert tool_call_msg["content"]["tool_calls"][0]["arguments"] == {"to_addr": "a@example.com"}
    tool_msgs = [m for m in artifacts.messages if m["role"] == "tool"]
    assert tool_msgs
    tool_payload = tool_msgs[0]["content"]
    assert tool_payload["tool_call_id"] == "call_1"
    assert tool_payload["tool_name"] == "email-env__send_email"
    assert tool_payload["is_error"] is False


def test_reference_agent_raises_when_mcp_server_fails_tool_listing() -> None:
    class HealthyMCPHandle:
        def list_tools(self) -> list[dict]:
            return [
                {
                    "tool_server": "healthy-mcp",
                    "name": "lookup_weather",
                    "description": "Weather lookup",
                    "input_schema": {"type": "object"},
                }
            ]

    class FailingMCPHandle:
        def list_tools(self) -> list[dict]:
            raise RuntimeError("unauthorized")

    mcp_clients: Any = {
        "healthy-mcp": HealthyMCPHandle(),
        "broken-mcp": FailingMCPHandle(),
    }

    with pytest.raises(RuntimeError, match="broken-mcp: unauthorized"):
        reference_agent._build_openai_tools(
            FakeEnvironment(),
            mcp_clients,
        )


def test_reference_agent_raises_on_duplicate_wire_name() -> None:
    class CollidingMCPHandle:
        def list_tools(self) -> list[dict]:
            return [
                {
                    "tool_server": "email-env",
                    "name": "send_email",
                    "description": "Duplicate send email",
                    "input_schema": {"type": "object"},
                }
            ]

    with pytest.raises(RuntimeError, match="Duplicate tool wire name detected"):
        reference_agent._build_openai_tools(
            FakeEnvironment(),
            {"email-env": CollidingMCPHandle()},  # type: ignore[dict-item]
        )


def test_mcp_client_handle_filters_and_prefixes_namespaced_gateway_tools(monkeypatch) -> None:  # noqa: ANN001
    async def fake_list_tools(_url: str, **kwargs: Any) -> list[dict]:
        _ = kwargs
        return [
            {"name": "weather_search", "description": "Weather search", "input_schema": {}},
            {"name": "notion_search", "description": "Notion search", "input_schema": {}},
        ]

    async def fake_call_tool(_url: str, tool_name: str, parameters: dict, **kwargs: Any):
        _ = _url, kwargs
        return ToolCallResult(observation={"tool_name": tool_name, "parameters": parameters})

    monkeypatch.setattr(mcp_client_module, "_async_list_tools", fake_list_tools)
    monkeypatch.setattr(mcp_client_module, "_async_call_tool", fake_call_tool)

    handle = mcp_client_module.MCPClientHandle(
        "http://localhost:8081/mcp",
        "weather",
        tool_prefix="weather_",
    )

    tools = handle.list_tools()
    assert tools == [
        {
            "name": "search",
            "description": "Weather search",
            "input_schema": {},
            "tool_server": "weather",
        }
    ]

    result = handle.call_tool("search", {"query": "sf"})
    assert result.observation == {
        "tool_name": "weather_search",
        "parameters": {"query": "sf"},
    }
