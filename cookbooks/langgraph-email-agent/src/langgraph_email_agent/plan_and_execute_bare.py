"""Bare plan-and-execute agent — same architecture, no prompt engineering.

Used for A/B testing: does the architecture matter, or just the prompts?
"""

from __future__ import annotations

import json
import operator
from typing import Annotated
from typing import Any
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent


class PlanStep(TypedDict):
    """A single step in the execution plan."""

    step: str
    done: bool


class PlanExecState(TypedDict):
    """State for the plan-and-execute graph."""

    task: str
    plan: list[PlanStep]
    current_step_idx: int
    step_results: Annotated[list[str], operator.add]
    final_output: str


def build_plan_and_execute_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
) -> StateGraph:
    """Build a 4-node graph with generic prompts and no domain hints."""
    # No tool filtering — pass all tools through
    react_executor = create_react_agent(
        model,
        tools,
        prompt=SystemMessage(
            content="You are an execution agent. Use the available tools "
            "to complete the step you are given."
        ),
    )

    def planner(state: PlanExecState) -> dict[str, Any]:
        plan_prompt = SystemMessage(
            content=(
                "You are a planning agent. Given a task, produce a JSON array of steps "
                "to complete it. Each step should be a short action string. "
                "Return ONLY a JSON array of strings, no other text. "
                "Keep the plan to 4-5 steps."
            )
        )
        task_msg = HumanMessage(content=state["task"])
        response = model.invoke([plan_prompt, task_msg])
        raw = response.content
        if isinstance(raw, list):
            raw = raw[0] if raw else "[]"
        text = str(raw).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            steps_list = json.loads(text)
        except json.JSONDecodeError:
            steps_list = [text]
        plan = [PlanStep(step=s, done=False) for s in steps_list]
        return {"plan": plan, "current_step_idx": 0}

    def executor(state: PlanExecState) -> dict[str, Any]:
        idx = state["current_step_idx"]
        plan = state["plan"]
        if idx >= len(plan):
            return {}
        step = plan[idx]
        prior = state.get("step_results", [])
        context = ""
        if prior:
            context = "Results from prior steps:\n" + "\n".join(prior) + "\n\n"
        instruction = f"{context}Execute step {idx + 1}/{len(plan)}: {step['step']}"
        result = react_executor.invoke({"messages": [HumanMessage(content=instruction)]})
        messages = result.get("messages", [])
        final_text = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                content = m.content
                if isinstance(content, list):
                    content = content[0] if content else ""
                final_text = str(content)
                break
        return {"step_results": [f"Step {idx + 1}: {final_text}"]}

    def reviewer(state: PlanExecState) -> dict[str, Any]:
        plan = list(state["plan"])
        idx = state["current_step_idx"]
        if idx < len(plan):
            plan[idx] = PlanStep(step=plan[idx]["step"], done=True)
        return {"plan": plan, "current_step_idx": idx + 1}

    def should_continue(state: PlanExecState) -> str:
        if state["current_step_idx"] < len(state["plan"]):
            return "executor"
        return "compiler"

    def compiler(state: PlanExecState) -> dict[str, Any]:
        results_text = "\n\n".join(state.get("step_results", []))
        compile_msg = HumanMessage(
            content=(
                f"Original task:\n{state['task']}\n\n"
                f"Completed step results:\n{results_text}\n\n"
                "Produce the final response. Follow the task's formatting instructions exactly."
            )
        )
        response = model.invoke(
            [
                SystemMessage(content="You are a report compiler."),
                compile_msg,
            ]
        )
        content = response.content
        if isinstance(content, list):
            content = content[0] if content else ""
        return {"final_output": str(content)}

    graph = StateGraph(PlanExecState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("reviewer", reviewer)
    graph.add_node("compiler", compiler)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        should_continue,
        {"executor": "executor", "compiler": "compiler"},
    )
    graph.add_edge("compiler", END)
    return graph
