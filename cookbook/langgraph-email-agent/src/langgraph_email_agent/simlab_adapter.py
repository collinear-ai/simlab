"""SimLab BaseAgent adapter for the standalone LangGraph assistant."""

from __future__ import annotations

import os

from simlab.agents import BaseAgent
from simlab.agents import BaseEnvironment
from simlab.agents import RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.langchain import build_langchain_tools
from simlab.agents.rollout_metrics import RolloutMetricsTracker
from simlab.agents.rollout_metrics import Timer

from langgraph_email_agent.email_assistant import LangGraphEmailAssistant
from langgraph_email_agent.model_factory import build_chat_model_from_env


class SimLabLangGraphEmailAgent(BaseAgent):
    """Cookbook LangGraph adapter for `simlab tasks run --agent-import-path`."""

    @staticmethod
    def name() -> str:
        """Return the public agent name."""
        return "langgraph-email-agent"

    def setup(self, environment: BaseEnvironment) -> None:
        """No-op setup hook for the SimLab contract."""
        _ = environment

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
    ) -> None:
        """Run the standalone assistant against the provided SimLab environment."""
        run_timer = Timer.start()
        tracker = RolloutMetricsTracker()
        model = build_chat_model_from_env()
        model_name = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or os.getenv("LANGGRAPH_EMAIL_MODEL", "").strip()
        )
        resolved_model_name = str(model_name).strip() if model_name else None
        assistant = LangGraphEmailAssistant(
            model=model,
            tool_recursion_limit=8,
            workflow_recursion_limit=context.max_steps or 16,
            mailbox_owner=(
                os.getenv("LANGGRAPH_EMAIL_MAILBOX_OWNER", "").strip()
                or "agent@weaverenterprises.com"
            ),
        )
        recorder = RunArtifactsRecorder(context)
        tools = build_langchain_tools(environment, recorder=recorder)
        context.metadata["cookbook_agent"] = {
            "name": self.name(),
            "tool_count": len(tools),
            "workflows": assistant.workflow_names(),
        }
        try:
            result = assistant.run(
                instruction,
                tools,
                recorder,
                record_usage=tracker.record_token_usage,
            )
        except Exception as exc:
            context.error = f"LangGraph assistant failed: {exc}"
            tracker.record_duration_seconds(run_timer.elapsed_seconds())
            tracker.merge_into(context.metadata, model=resolved_model_name)
            return
        if not result.final_output:
            context.error = "LangGraph assistant produced no final output"
            tracker.record_duration_seconds(run_timer.elapsed_seconds())
            tracker.merge_into(context.metadata, model=resolved_model_name)
            return
        context.final_observation = result.final_output
        tracker.record_duration_seconds(run_timer.elapsed_seconds())
        tracker.merge_into(context.metadata, model=resolved_model_name)
