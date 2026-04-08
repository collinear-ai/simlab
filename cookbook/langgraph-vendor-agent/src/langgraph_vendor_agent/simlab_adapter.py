"""SimLab BaseAgent adapter for the vendor management agent."""

from __future__ import annotations

import os

from simlab.agents import BaseAgent, BaseEnvironment, RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.langchain import build_langchain_tools
from simlab.agents.rollout_metrics import RolloutMetricsTracker
from simlab.agents.rollout_metrics import Timer

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
        run_timer = Timer.start()
        tracker = RolloutMetricsTracker()
        model = build_chat_model_from_env()
        model_name = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or os.getenv("LANGGRAPH_VENDOR_MODEL", "").strip()
        )
        resolved_model_name = str(model_name).strip() if model_name else None
        recorder = RunArtifactsRecorder(context)
        tools = build_langchain_tools(environment, recorder=recorder)

        graph = build_plan_and_execute_graph(
            model,
            tools,
            record_usage=tracker.record_token_usage,
        )
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
            tracker.record_duration_seconds(run_timer.elapsed_seconds())
            tracker.merge_into(context.metadata, model=resolved_model_name)
            return

        recorder.on_assistant_message(final)
        context.final_observation = final
        context.metadata["cookbook_agent"] = {
            "name": self.name(),
            "tool_count": len(tools),
            "plan_steps": len(result.get("plan", [])),
            "steps_completed": sum(1 for s in result.get("plan", []) if s.get("done")),
        }
        tracker.record_duration_seconds(run_timer.elapsed_seconds())
        tracker.merge_into(context.metadata, model=resolved_model_name)
