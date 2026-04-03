"""SimLab BaseAgent adapter for a custom Google ADK agent."""

from __future__ import annotations

from simlab.agents import BaseAgent
from simlab.agents import BaseEnvironment
from simlab.agents import RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.google_adk import build_google_adk_tools

from google_adk_agent.custom_agent import run_custom_agent


class SimLabGoogleADKAgent(BaseAgent):
    """Cookbook adapter for `simlab tasks run --agent-import-path`."""

    @staticmethod
    def name() -> str:
        """Return the public agent name."""
        return "google-adk"

    def setup(self, environment: BaseEnvironment) -> None:
        """No-op setup hook for the SimLab contract."""
        _ = environment

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
    ) -> None:
        """Run a Google ADK agent against the provided SimLab environment."""
        recorder = RunArtifactsRecorder(context)
        recorder.on_user_message(instruction)
        tools = build_google_adk_tools(environment, recorder=recorder)
        context.metadata["cookbook_agent"] = {
            "name": self.name(),
            "tool_count": len(tools),
            "tool_names": [getattr(tool, "name", "") for tool in tools],
        }
        try:
            final_text = run_custom_agent(
                instruction=instruction,
                tools=tools,
                model=context.model,
                max_llm_calls=context.max_steps,
                session_id=context.task_id,
            )
        except Exception as exc:
            context.error = f"Google ADK agent failed: {exc}"
            return

        if not final_text:
            context.error = "Google ADK agent produced no final output"
            return

        recorder.on_assistant_message(final_text)
        context.final_observation = final_text
