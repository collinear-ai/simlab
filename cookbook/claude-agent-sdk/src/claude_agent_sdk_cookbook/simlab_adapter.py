"""SimLab BaseAgent adapter for a custom Claude Agent SDK app."""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any

from simlab.agents import BaseAgent
from simlab.agents import BaseEnvironment
from simlab.agents import RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.claude_agent import build_claude_agent_tools

from claude_agent_sdk_cookbook.custom_agent import run_custom_agent
from claude_agent_sdk_cookbook.custom_agent import stringify_final_output


class BaseSimLabClaudeAgentSDKAgent(BaseAgent):
    """Reusable SimLab wrapper for Claude Agent SDK apps."""

    @staticmethod
    def name() -> str:
        """Return the public agent name."""
        return "claude-agent-sdk"

    def setup(self, environment: BaseEnvironment) -> None:
        """No-op setup hook for the SimLab contract."""
        _ = environment

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
    ) -> None:
        """Run a custom SDK app against the provided SimLab environment."""
        recorder = RunArtifactsRecorder(context)
        recorder.on_user_message(instruction)
        mcp_servers, allowed_tools = build_claude_agent_tools(
            environment, recorder=recorder
        )
        context.metadata["cookbook_agent"] = {
            "name": self.name(),
            "tool_count": len(allowed_tools),
            "tool_names": allowed_tools,
        }
        try:
            result = self.run_sdk_agent(
                instruction=instruction,
                mcp_servers=mcp_servers,
                allowed_tools=allowed_tools,
                model=context.model,
                max_turns=context.max_steps,
            )
        except Exception as exc:
            context.error = f"Claude Agent SDK agent failed: {exc}"
            return

        if not result or not str(result).strip():
            context.error = "Claude Agent SDK agent produced no final output"
            return

        final_text = stringify_final_output(result)
        recorder.on_assistant_message(final_text)
        context.final_observation = final_text

    @abstractmethod
    def run_sdk_agent(
        self,
        *,
        instruction: str,
        mcp_servers: dict[str, Any],
        allowed_tools: list[str],
        model: str | None = None,
        max_turns: int | None = None,
    ) -> str:
        """Run the custom SDK app and return the final output string."""


class SimLabClaudeAgentSDKAgent(BaseSimLabClaudeAgentSDKAgent):
    """Cookbook adapter wired to the bundled custom Claude Agent SDK app."""

    def run_sdk_agent(
        self,
        *,
        instruction: str,
        mcp_servers: dict[str, Any],
        allowed_tools: list[str],
        model: str | None = None,
        max_turns: int | None = None,
    ) -> str:
        """Delegate execution to the bundled custom agent module."""
        return asyncio.run(
            run_custom_agent(
                instruction=instruction,
                mcp_servers=mcp_servers,
                allowed_tools=allowed_tools,
                model=model,
                max_turns=max_turns,
            )
        )
