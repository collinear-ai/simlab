"""Unit tests for the vendor management agent adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from simlab.agents.base import (
    BaseEnvironment,
    RunArtifacts,
    ToolCallResult,
    ToolNamespace,
)


class FakeEnvironment(BaseEnvironment):
    def list_tool_namespaces(self) -> list[ToolNamespace]:
        return [
            ToolNamespace(
                name="email-env", transport="http", endpoint="http://localhost:8040"
            ),
            ToolNamespace(
                name="erp-env", transport="http", endpoint="http://localhost:8100"
            ),
        ]

    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "tool_server": "email-env",
                "name": "search_emails",
                "description": "Search emails",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "transport": "http",
            },
            {
                "tool_server": "erp-env",
                "name": "search_suppliers",
                "description": "Search suppliers",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "transport": "http",
            },
        ]

    async def acall_tool(
        self, tool_server: str, tool_name: str, parameters: dict[str, Any]
    ) -> ToolCallResult:
        return ToolCallResult(observation={"results": [{"name": "NovaTech"}]})


def test_vendor_agent_populates_artifacts() -> None:
    from langgraph_vendor_agent.simlab_adapter import VendorManagementAgent

    agent = VendorManagementAgent()
    assert agent.name() == "langgraph-vendor-agent"

    env = FakeEnvironment()
    context = RunArtifacts(task_id="test-1", task="test task", max_steps=7)
    agent.setup(env)

    fake_result = {
        "final_output": "## Vendor Performance Report\n- NovaTech\n## Action Items\n- [P1] Fix delays",
        "plan": [{"step": "search", "done": True}],
        "step_results": ["Step 1: found NovaTech"],
    }

    with patch(
        "langgraph_vendor_agent.simlab_adapter.build_plan_and_execute_graph"
    ) as mock_build:
        mock_graph = MagicMock()
        mock_app = MagicMock()
        mock_app.invoke.return_value = fake_result
        mock_graph.compile.return_value = mock_app
        mock_build.return_value = mock_graph

        with patch("langgraph_vendor_agent.simlab_adapter.build_chat_model_from_env"):
            agent.run("test instruction", env, context)

    _, invoke_kwargs = mock_app.invoke.call_args
    assert invoke_kwargs["config"]["recursion_limit"] == 7

    assert context.final_observation is not None
    assert "NovaTech" in context.final_observation
    assert context.metadata["cookbook_agent"]["name"] == "langgraph-vendor-agent"
    assert context.error is None


def test_vendor_agent_handles_empty_output() -> None:
    from langgraph_vendor_agent.simlab_adapter import VendorManagementAgent

    agent = VendorManagementAgent()
    env = FakeEnvironment()
    context = RunArtifacts(task_id="test-2", task="test task", max_steps=9)

    fake_result = {"final_output": "", "plan": [], "step_results": []}

    with patch(
        "langgraph_vendor_agent.simlab_adapter.build_plan_and_execute_graph"
    ) as mock_build:
        mock_graph = MagicMock()
        mock_app = MagicMock()
        mock_app.invoke.return_value = fake_result
        mock_graph.compile.return_value = mock_app
        mock_build.return_value = mock_graph

        with patch("langgraph_vendor_agent.simlab_adapter.build_chat_model_from_env"):
            agent.run("test instruction", env, context)

    _, invoke_kwargs = mock_app.invoke.call_args
    assert invoke_kwargs["config"]["recursion_limit"] == 9

    assert context.error is not None
    assert "no final output" in context.error
