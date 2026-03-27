from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Self

import pytest
import simlab.agents.mcp_client as mcp_client_module
from simlab.agents import BaseAgent
from simlab.agents import BaseEnvironment
from simlab.agents import HttpToolEnvironment
from simlab.agents import ToolCallResult
from simlab.agents import ToolNamespace
from simlab.agents import UnifiedToolEnvironment
from simlab.agents import load_agent_class
from simlab.agents import reference_agent
from simlab.agents import run_with_agent_contract
from simlab.agents.base import RunArtifacts


class FakeEnvironment(BaseEnvironment):
    def list_tool_namespaces(self) -> list[ToolNamespace]:
        return [ToolNamespace(name="email-env", transport="http", endpoint="http://localhost:8040")]

    @property
    def tool_servers(self) -> dict[str, str]:
        return {"email-env": "http://localhost:8040"}

    async def alist_tools(self, tool_server: str | None = None) -> list[dict]:
        _ = tool_server
        return [
            {
                "tool_server": "email-env",
                "name": "send_email",
                "description": "Send email",
                "input_schema": {"type": "object"},
            }
        ]

    async def acall_tool(
        self, tool_server: str, tool_name: str, parameters: dict
    ) -> ToolCallResult:
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


class AsyncRecordingAgent(BaseAgent):
    def setup(self, environment: BaseEnvironment) -> Any:
        return self._asetup(environment)

    async def _asetup(self, environment: BaseEnvironment) -> None:
        _ = environment

    def run(self, instruction: str, environment: BaseEnvironment, context) -> Any:  # noqa: ANN001
        return self._arun(instruction, environment, context)

    async def _arun(self, instruction: str, environment: BaseEnvironment, context) -> None:  # noqa: ANN001
        _ = environment
        context.record_message("user", instruction)
        context.final_observation = "async-done"
        context.metadata["async_custom"] = True


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
    assert tool_call_msg["content"]["tool_calls"][0]["function"]["name"] == "email-env__send_email"
    assert json.loads(tool_call_msg["content"]["tool_calls"][0]["function"]["arguments"]) == {
        "to_addr": "a@example.com"
    }
    tool_msgs = [m for m in artifacts.messages if m["role"] == "tool"]
    assert tool_msgs
    tool_payload = tool_msgs[0]["content"]
    assert tool_payload["tool_call_id"] == "call_1"
    assert tool_payload["tool_name"] == "email-env__send_email"
    assert tool_payload["is_error"] is False


def test_run_with_agent_contract_supports_async_agent_inside_running_event_loop() -> None:
    async def _run_inside_event_loop() -> RunArtifacts:
        return run_with_agent_contract(
            task_id="t-async-custom",
            instruction="do task",
            model="gpt-test",
            provider="openai",
            max_steps=5,
            environment=FakeEnvironment(),
            agent_import_path=f"{__name__}:AsyncRecordingAgent",
            timeout_seconds=1.0,
        )

    artifacts = asyncio.run(_run_inside_event_loop())

    assert artifacts.error is None
    assert artifacts.final_observation == "async-done"
    assert artifacts.metadata["async_custom"] is True


def test_reference_agent_supports_running_event_loop_hosts(monkeypatch) -> None:  # noqa: ANN001
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

    async def _run_inside_event_loop() -> RunArtifacts:
        return run_with_agent_contract(
            task_id="t-ref-async-host",
            instruction="send an email",
            model="gpt-test",
            provider="openai",
            max_steps=5,
            environment=FakeEnvironment(),
            timeout_seconds=1.0,
            api_key="sk-test",
            base_url=None,
        )

    artifacts = asyncio.run(_run_inside_event_loop())

    assert artifacts.error is None
    assert artifacts.final_observation == "task complete"
    assert artifacts.steps_taken == 1


def test_reference_agent_raises_when_environment_mcp_tool_listing_fails() -> None:
    class FailingEnvironment(FakeEnvironment):
        async def alist_tools(self, tool_server: str | None = None) -> list[dict]:
            _ = tool_server
            raise RuntimeError("broken-mcp: unauthorized")

    with pytest.raises(RuntimeError, match="broken-mcp: unauthorized"):
        asyncio.run(reference_agent._abuild_openai_tools(FailingEnvironment()))


def test_reference_agent_raises_on_duplicate_wire_name() -> None:
    async def _duplicate_tools(self: object, tool_server: str | None = None) -> list[dict]:
        _ = self, tool_server
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

    with pytest.raises(RuntimeError, match="Duplicate tool wire name detected"):
        asyncio.run(
            reference_agent._abuild_openai_tools(
                type(
                    "CollidingEnvironment",
                    (FakeEnvironment,),
                    {"alist_tools": _duplicate_tools},
                )()
            )
        )


def test_http_tool_environment_merges_http_and_mcp_tools(monkeypatch) -> None:  # noqa: ANN001
    async def fake_list_tools(_url: str, **kwargs: Any) -> list[dict]:
        _ = kwargs
        return [{"name": "ping", "description": "Ping", "input_schema": {"type": "object"}}]

    async def fake_call_tool(_url: str, tool_name: str, parameters: dict, **kwargs: Any):
        _ = kwargs
        return ToolCallResult(
            observation={"tool_name": tool_name, "parameters": parameters, "transport": "mcp"}
        )

    class FakeHttpResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._payload

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            _ = exc_type, exc, tb

    def fake_urlopen(request, timeout=30.0):  # noqa: ANN001, ARG001
        full_url = request.full_url
        if full_url.endswith("/tools"):
            return FakeHttpResponse(
                {
                    "tools": [
                        {
                            "name": "send_email",
                            "description": "Send email",
                            "input_schema": {"type": "object"},
                        }
                    ]
                }
            )
        return FakeHttpResponse({"ok": True})

    monkeypatch.setattr(mcp_client_module, "_async_list_tools", fake_list_tools)
    monkeypatch.setattr(mcp_client_module, "_async_call_tool", fake_call_tool)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    environment = UnifiedToolEnvironment(
        tool_servers={"email-env": "http://localhost:8040"},
        mcp_clients={
            "demo": mcp_client_module.MCPClientHandle("http://localhost:8081/mcp", "demo")
        },
    )

    namespaces = environment.list_tool_namespaces()
    assert {(namespace.name, namespace.transport) for namespace in namespaces} == {
        ("email-env", "http"),
        ("demo", "mcp"),
    }
    tools = asyncio.run(environment.alist_tools())
    assert {tool["tool_server"] for tool in tools} == {"email-env", "demo"}
    assert {tool["transport"] for tool in tools} == {"http", "mcp"}

    mcp_result = asyncio.run(environment.acall_tool("demo", "ping", {"value": 1}))
    assert mcp_result.observation == {
        "tool_name": "ping",
        "parameters": {"value": 1},
        "transport": "mcp",
    }


def test_unified_tool_environment_supports_mcp_calls_inside_running_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_list_tools(_url: str, **kwargs: Any) -> list[dict]:
        _ = kwargs
        return [{"name": "ping", "description": "Ping", "input_schema": {"type": "object"}}]

    async def fake_call_tool(_url: str, tool_name: str, parameters: dict, **kwargs: Any):
        _ = kwargs
        return ToolCallResult(
            observation={"tool_name": tool_name, "parameters": parameters, "transport": "mcp"}
        )

    monkeypatch.setattr(mcp_client_module, "_async_list_tools", fake_list_tools)
    monkeypatch.setattr(mcp_client_module, "_async_call_tool", fake_call_tool)

    environment = UnifiedToolEnvironment(
        tool_servers={},
        mcp_clients={
            "demo": mcp_client_module.MCPClientHandle("http://localhost:8081/mcp", "demo")
        },
    )

    async def _exercise_environment() -> tuple[list[dict[str, Any]], ToolCallResult]:
        with pytest.deprecated_call(match="removed in 0.4.0"):
            tools = environment.list_tools()
        with pytest.deprecated_call(match="removed in 0.4.0"):
            result = environment.call_tool("demo", "ping", {"value": 1})
        return tools, result

    tools, result = asyncio.run(_exercise_environment())

    assert tools == [
        {
            "tool_server": "demo",
            "transport": "mcp",
            "name": "ping",
            "description": "Ping",
            "input_schema": {"type": "object"},
        }
    ]
    assert result.observation == {
        "tool_name": "ping",
        "parameters": {"value": 1},
        "transport": "mcp",
    }


def test_unified_tool_environment_rejects_duplicate_http_and_mcp_namespaces() -> None:
    with pytest.raises(
        ValueError,
        match="Tool namespace names must be unique across HTTP and MCP transports: shared",
    ):
        UnifiedToolEnvironment(
            tool_servers={"shared": "http://localhost:8040"},
            mcp_clients={
                "shared": mcp_client_module.MCPClientHandle(
                    "http://localhost:8081/mcp",
                    "shared",
                )
            },
        )


def test_http_tool_environment_is_deprecated() -> None:
    with pytest.deprecated_call(match="removed in 0.4.0"):
        HttpToolEnvironment(tool_servers={"email-env": "http://localhost:8040"})


def test_unified_tool_environment_tool_servers_property_is_deprecated() -> None:
    environment = UnifiedToolEnvironment(tool_servers={"email-env": "http://localhost:8040"})
    with pytest.deprecated_call(match="removed in 0.4.0"):
        assert environment.tool_servers == {"email-env": "http://localhost:8040"}


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

    tools = asyncio.run(handle.alist_tools())
    assert tools == [
        {
            "name": "search",
            "description": "Weather search",
            "input_schema": {},
            "tool_server": "weather",
        }
    ]

    result = asyncio.run(handle.acall_tool("search", {"query": "sf"}))
    assert result.observation == {
        "tool_name": "weather_search",
        "parameters": {"query": "sf"},
    }


def test_base_environment_sync_tool_methods_are_deprecated() -> None:
    environment = FakeEnvironment()
    with pytest.deprecated_call(match="removed in 0.4.0"):
        tools = environment.list_tools()
    with pytest.deprecated_call(match="removed in 0.4.0"):
        result = environment.call_tool("email-env", "send_email", {})
    assert tools[0]["name"] == "send_email"
    assert result.is_error is False


def test_unified_tool_environment_sync_tool_methods_are_deprecated(monkeypatch) -> None:  # noqa: ANN001
    async def fake_list_tools(_url: str, **kwargs: Any) -> list[dict]:
        _ = kwargs
        return [{"name": "ping", "description": "Ping", "input_schema": {"type": "object"}}]

    monkeypatch.setattr(mcp_client_module, "_async_list_tools", fake_list_tools)

    environment = UnifiedToolEnvironment(
        tool_servers={},
        mcp_clients={
            "demo": mcp_client_module.MCPClientHandle("http://localhost:8081/mcp", "demo")
        },
    )

    with pytest.deprecated_call(match="removed in 0.4.0"):
        tools = environment.list_tools()

    assert tools[0]["name"] == "ping"
