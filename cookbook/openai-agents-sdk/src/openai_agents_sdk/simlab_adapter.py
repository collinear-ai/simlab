"""SimLab BaseAgent adapter for a custom OpenAI Agents SDK app."""

from __future__ import annotations

import re
from abc import abstractmethod

from agents.result import RunResult
from agents.tool import Tool
from simlab.agents import BaseAgent
from simlab.agents import BaseEnvironment
from simlab.agents import RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters import build_openai_agents_tools

from openai_agents_sdk.custom_agent import run_custom_agent
from openai_agents_sdk.custom_agent import stringify_final_output


class BaseSimLabOpenAIAgentsSDKAgent(BaseAgent):
    """Reusable SimLab wrapper for OpenAI Agents SDK apps."""

    @staticmethod
    def name() -> str:
        """Return the public agent name."""
        return "openai-agents-sdk"

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
        tools = build_openai_agents_tools(environment, recorder=recorder)
        context.metadata["cookbook_agent"] = {
            "name": self.name(),
            "tool_count": len(tools),
            "tool_names": [getattr(tool, "name", "") for tool in tools],
        }
        try:
            result = self.run_sdk_agent(
                instruction=instruction,
                tools=tools,
                model=context.model,
                max_turns=context.max_steps,
            )
        except Exception as exc:
            failed_tool = _extract_failed_tool_name(str(exc))
            if failed_tool is not None:
                context.metadata["cookbook_agent"]["failed_tool_name"] = failed_tool
            context.error = f"OpenAI Agents SDK agent failed: {exc}"
            return

        final_output = getattr(result, "final_output", None)
        if final_output is None or not str(final_output).strip():
            context.error = "OpenAI Agents SDK agent produced no final output"
            return

        final_text = stringify_final_output(final_output)
        recorder.on_assistant_message(final_text)
        context.final_observation = final_text

    @abstractmethod
    def run_sdk_agent(
        self,
        *,
        instruction: str,
        tools: list[Tool],
        model: str | None = None,
        max_turns: int | None = None,
    ) -> RunResult:
        """Run the custom SDK app and return the SDK result."""


class SimLabOpenAIAgentsSDKAgent(BaseSimLabOpenAIAgentsSDKAgent):
    """Cookbook adapter wired to the bundled custom OpenAI Agents SDK app."""

    def run_sdk_agent(
        self,
        *,
        instruction: str,
        tools: list[Tool],
        model: str | None = None,
        max_turns: int | None = None,
    ) -> RunResult:
        """Delegate execution to the bundled custom agent module."""
        return run_custom_agent(
            instruction=instruction,
            tools=tools,
            model=model,
            max_turns=max_turns,
        )


def _extract_failed_tool_name(message: str) -> str | None:
    match = re.search(r"function '([^']+)'", message)
    if match is None:
        return None
    return match.group(1)
