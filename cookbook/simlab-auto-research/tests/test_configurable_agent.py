# ruff: noqa: D100, D101, D102, D103, S101, SLF001

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from simlab.agents import BaseEnvironment
from simlab.agents import RunArtifacts
from simlab.agents import ToolCallResult
from simlab.agents import ToolNamespace

from simlab_auto_research.configurable_agent import ConfigurableAgent


class FakeEnvironment(BaseEnvironment):
    def list_tool_namespaces(self) -> list[ToolNamespace]:
        return [
            ToolNamespace(
                name="coding-env", transport="http", endpoint="http://localhost:8020"
            ),
        ]

    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        _ = tool_server
        return [
            {
                "tool_server": "coding-env",
                "name": "execute_bash",
                "description": "Run a bash command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
                "transport": "http",
            },
        ]

    async def acall_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        return ToolCallResult(
            observation={"tool_server": tool_server, "tool_name": tool_name}
        )


def test_name() -> None:
    assert ConfigurableAgent.name() == "configurable-agent"


def test_loads_system_prompt_from_file(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are a helpful coding assistant.")
    monkeypatch.setenv("SYSTEM_PROMPT_PATH", str(prompt_file))

    agent = ConfigurableAgent()
    loaded = agent._load_system_prompt()
    assert loaded == "You are a helpful coding assistant."


def test_raises_when_prompt_file_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SYSTEM_PROMPT_PATH", str(tmp_path / "nonexistent.md"))

    agent = ConfigurableAgent()
    try:
        agent._load_system_prompt()
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "nonexistent.md" in str(exc)


def test_records_metadata(tmp_path, monkeypatch) -> None:
    """Mock litellm to return a simple non-tool-call response and verify metadata."""
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Test system prompt for coding tasks.")
    monkeypatch.setenv("SYSTEM_PROMPT_PATH", str(prompt_file))
    monkeypatch.setenv("SIMLAB_AGENT_API_KEY", "test-key")

    mock_message = MagicMock()
    mock_message.content = "Task completed successfully."
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    import litellm  # noqa: PLC0415

    monkeypatch.setattr(litellm, "completion", MagicMock(return_value=mock_response))

    artifacts = RunArtifacts(task_id="test-task", task="test", model="gpt-4o-mini")
    agent = ConfigurableAgent()
    agent.run("Write hello world", FakeEnvironment(), artifacts)

    assert "configurable_agent" in artifacts.metadata
    meta = artifacts.metadata["configurable_agent"]
    assert meta["system_prompt_path"] == str(prompt_file)
    assert meta["system_prompt_length"] == len("Test system prompt for coding tasks.")
    assert meta["tool_count"] == 1
    assert artifacts.final_observation == "Task completed successfully."


def test_tool_call_path(tmp_path, monkeypatch) -> None:
    """Ensure the agent correctly handles a tool-call response followed by a final reply."""
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are a coding assistant.")
    monkeypatch.setenv("SYSTEM_PROMPT_PATH", str(prompt_file))
    monkeypatch.setenv("SIMLAB_AGENT_API_KEY", "test-key")

    # First response: a tool call
    mock_fn = MagicMock()
    mock_fn.name = "coding-env__execute_bash"
    mock_fn.arguments = '{"command": "echo hello"}'

    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function = mock_fn

    tool_call_message = MagicMock()
    tool_call_message.content = "Let me run that."
    tool_call_message.tool_calls = [mock_tc]

    # Second response: final text (no tool calls)
    final_message = MagicMock()
    final_message.content = "Done! Output was hello."
    final_message.tool_calls = None

    mock_resp_1 = MagicMock()
    mock_resp_1.choices = [MagicMock(message=tool_call_message)]

    mock_resp_2 = MagicMock()
    mock_resp_2.choices = [MagicMock(message=final_message)]

    import litellm  # noqa: PLC0415

    monkeypatch.setattr(
        litellm, "completion", MagicMock(side_effect=[mock_resp_1, mock_resp_2])
    )

    artifacts = RunArtifacts(task_id="test-task", task="test", model="gpt-4o-mini")
    agent = ConfigurableAgent()
    agent.run("Run echo hello", FakeEnvironment(), artifacts)

    # Should have recorded tool call and completed successfully
    assert len(artifacts.tool_calls) == 1
    assert artifacts.tool_calls[0].tool_name == "execute_bash"
    assert artifacts.steps_taken == 1
    assert artifacts.final_observation == "Done! Output was hello."
    assert artifacts.error is None
