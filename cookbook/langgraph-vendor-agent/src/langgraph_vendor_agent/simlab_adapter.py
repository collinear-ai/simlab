"""SimLab BaseAgent adapter for the vendor management agent."""

from __future__ import annotations

from simlab.agents import BaseAgent, BaseEnvironment, RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.langchain import build_langchain_tools

from langgraph_vendor_agent.model_factory import build_chat_model_from_env
from langgraph_vendor_agent.plan_and_execute import build_plan_and_execute_graph


class VendorManagementAgent(BaseAgent):
    """Plan-and-execute vendor agent for ``simlab tasks run --agent-import-path``."""

    @staticmethod
    def name() -> str:
        return "langgraph-vendor-agent"

    def setup(self, environment: BaseEnvironment) -> None:
        _ = environment

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
    ) -> None:
        model = build_chat_model_from_env()
        recorder = RunArtifactsRecorder(context)
        tools = build_langchain_tools(environment, recorder=recorder)

        graph = build_plan_and_execute_graph(model, tools)
        app = graph.compile()

        recorder.on_user_message(instruction)
        recursion_limit = context.max_steps or 16
        result = app.invoke(
            {
                "task": instruction,
                "plan": [],
                "current_step_idx": 0,
                "step_results": [],
                "final_output": "",
            },
            config={"recursion_limit": recursion_limit},
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
