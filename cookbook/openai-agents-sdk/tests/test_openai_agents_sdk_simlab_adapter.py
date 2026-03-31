# ruff: noqa: D100, D101, D102, D103, S101

from __future__ import annotations

from openai_agents_sdk import simlab_adapter
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
    class FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        simlab_adapter,
        "build_openai_agents_tools",
        lambda environment, recorder: [
            FakeTool("email-env__search_emails"),
            FakeTool("demo__ping"),
        ],
    )
    monkeypatch.setattr(
        simlab_adapter,
        "run_custom_agent",
        lambda **_: type("Result", (), {"final_output": "final answer"})(),
    )

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    agent = simlab_adapter.SimLabOpenAIAgentsSDKAgent()
    agent.run("triage inbox", FakeEnvironment(), artifacts)

    assert artifacts.metadata["cookbook_agent"] == {
        "name": "openai-agents-sdk",
        "tool_count": 2,
        "tool_names": ["email-env__search_emails", "demo__ping"],
    }
    assert artifacts.final_observation == "final answer"
    assert artifacts.messages[0] == {"role": "user", "content": "triage inbox"}
    assert artifacts.messages[-1] == {"role": "assistant", "content": "final answer"}


def test_simlab_agent_errors_on_empty_final_output(monkeypatch) -> None:
    monkeypatch.setattr(
        simlab_adapter,
        "build_openai_agents_tools",
        lambda environment, recorder: [],
    )
    monkeypatch.setattr(
        simlab_adapter,
        "run_custom_agent",
        lambda **_: type("Result", (), {"final_output": ""})(),
    )

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    simlab_adapter.SimLabOpenAIAgentsSDKAgent().run(
        "triage inbox",
        FakeEnvironment(),
        artifacts,
    )

    assert artifacts.error == "OpenAI Agents SDK agent produced no final output"


def test_simlab_agent_wraps_runtime_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        simlab_adapter,
        "build_openai_agents_tools",
        lambda environment, recorder: [],
    )

    def _raise(**_: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(simlab_adapter, "run_custom_agent", _raise)

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    simlab_adapter.SimLabOpenAIAgentsSDKAgent().run(
        "triage inbox",
        FakeEnvironment(),
        artifacts,
    )

    assert artifacts.error == "OpenAI Agents SDK agent failed: boom"


def test_simlab_agent_records_failed_tool_name_from_runtime_error(monkeypatch) -> None:
    monkeypatch.setattr(
        simlab_adapter,
        "build_openai_agents_tools",
        lambda environment, recorder: [],
    )

    def _raise(**_: object) -> object:
        raise RuntimeError(
            "Invalid schema for function 'edgar-mcp__get_key_metrics': "
            "schema must have a 'type' key."
        )

    monkeypatch.setattr(simlab_adapter, "run_custom_agent", _raise)

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    simlab_adapter.SimLabOpenAIAgentsSDKAgent().run(
        "triage inbox",
        FakeEnvironment(),
        artifacts,
    )

    assert artifacts.metadata["cookbook_agent"]["failed_tool_name"] == (
        "edgar-mcp__get_key_metrics"
    )


def test_agent_import_path_loads_the_adapter_class() -> None:
    agent_cls = load_agent_class(
        "openai_agents_sdk.simlab_adapter:SimLabOpenAIAgentsSDKAgent"
    )

    assert agent_cls is simlab_adapter.SimLabOpenAIAgentsSDKAgent
