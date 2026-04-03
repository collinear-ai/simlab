# ruff: noqa: D100, D101, D102, D103, S101

from __future__ import annotations

from google_adk_agent import simlab_adapter
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
        "build_google_adk_tools",
        lambda environment, recorder: [
            FakeTool("email-env__search_emails"),
            FakeTool("demo__ping"),
        ],
    )
    monkeypatch.setattr(
        simlab_adapter,
        "run_custom_agent",
        lambda **_: "final answer",
    )

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    agent = simlab_adapter.SimLabGoogleADKAgent()
    agent.run("triage inbox", FakeEnvironment(), artifacts)

    assert artifacts.metadata["cookbook_agent"] == {
        "name": "google-adk",
        "tool_count": 2,
        "tool_names": ["email-env__search_emails", "demo__ping"],
    }
    assert artifacts.final_observation == "final answer"
    assert artifacts.messages[0]["role"] == "user"
    assert artifacts.messages[0]["content"] == "triage inbox"
    assert artifacts.messages[-1]["role"] == "assistant"
    assert artifacts.messages[-1]["content"] == "final answer"


def test_simlab_agent_errors_on_empty_final_output(monkeypatch) -> None:
    monkeypatch.setattr(
        simlab_adapter,
        "build_google_adk_tools",
        lambda environment, recorder: [],
    )
    monkeypatch.setattr(
        simlab_adapter,
        "run_custom_agent",
        lambda **_: "",
    )

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    simlab_adapter.SimLabGoogleADKAgent().run(
        "triage inbox",
        FakeEnvironment(),
        artifacts,
    )

    assert artifacts.error == "Google ADK agent produced no final output"


def test_simlab_agent_passes_step_budget_to_custom_agent(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_custom_agent(**kwargs: object) -> str:
        captured.update(kwargs)
        return "final answer"

    monkeypatch.setattr(
        simlab_adapter,
        "build_google_adk_tools",
        lambda environment, recorder: [],
    )
    monkeypatch.setattr(simlab_adapter, "run_custom_agent", fake_run_custom_agent)

    artifacts = RunArtifacts(task_id="task-1", task="demo", max_steps=5)
    simlab_adapter.SimLabGoogleADKAgent().run(
        "triage inbox",
        FakeEnvironment(),
        artifacts,
    )

    assert captured.get("max_llm_calls") == 5


def test_simlab_agent_wraps_runtime_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        simlab_adapter,
        "build_google_adk_tools",
        lambda environment, recorder: [],
    )

    def raise_error(**_: object) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(simlab_adapter, "run_custom_agent", raise_error)

    artifacts = RunArtifacts(task_id="task-1", task="demo")
    simlab_adapter.SimLabGoogleADKAgent().run(
        "triage inbox",
        FakeEnvironment(),
        artifacts,
    )

    assert artifacts.error == "Google ADK agent failed: boom"


def test_agent_import_path_loads_the_adapter_class() -> None:
    agent_cls = load_agent_class("google_adk_agent.simlab_adapter:SimLabGoogleADKAgent")

    assert agent_cls is simlab_adapter.SimLabGoogleADKAgent
