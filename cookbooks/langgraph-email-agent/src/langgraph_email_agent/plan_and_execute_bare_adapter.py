"""SimLab adapter for the bare (no prompt engineering) plan-and-execute agent."""

from __future__ import annotations

from simlab.agents import BaseAgent
from simlab.agents import BaseEnvironment
from simlab.agents import RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.langchain import build_langchain_tools

from langgraph_email_agent.model_factory import build_chat_model_from_env
from langgraph_email_agent.plan_and_execute_bare import build_plan_and_execute_graph


class PlanAndExecuteBareAgent(BaseAgent):
    """Bare plan-and-execute agent — same arch, no prompt tricks."""

    @staticmethod
    def name() -> str:
        """Return the agent name."""
        return "plan-and-execute-bare"

    def setup(self, environment: BaseEnvironment) -> None:
        """Initialize the agent with the environment."""
        _ = environment

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
    ) -> None:
        """Execute the bare plan-and-execute agent loop."""
        model = build_chat_model_from_env()
        recorder = RunArtifactsRecorder(context)
        tools = build_langchain_tools(environment, recorder=recorder)

        graph = build_plan_and_execute_graph(model, tools)
        app = graph.compile()

        recorder.on_user_message(instruction)
        result = app.invoke(
            {
                "task": instruction,
                "plan": [],
                "current_step_idx": 0,
                "step_results": [],
                "final_output": "",
            }
        )

        final = result.get("final_output", "")
        if not final:
            context.error = "Plan-and-execute agent produced no final output"
            return

        recorder.on_assistant_message(final)
        context.final_observation = final
        context.metadata["cookbook_agent"] = {
            "name": self.name(),
            "tool_count": len(tools),
            "plan_steps": len(result.get("plan", [])),
            "steps_completed": sum(1 for s in result.get("plan", []) if s.get("done")),
        }
