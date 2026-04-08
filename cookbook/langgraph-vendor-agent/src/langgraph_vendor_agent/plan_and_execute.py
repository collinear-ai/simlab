"""Plan-and-execute agent: planner → react executor → reviewer → compiler.

A domain-agnostic LangGraph agent that decomposes tasks into steps,
executes each with a react tool-calling loop, and compiles results.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent


class PlanStep(TypedDict):
    step: str
    done: bool


class PlanExecState(TypedDict):
    task: str
    plan: list[PlanStep]
    current_step_idx: int
    step_results: Annotated[list[str], operator.add]
    final_output: str


def _message_text(content: str | dict[str, Any] | list[str | dict[str, Any]]) -> str:
    """Normalize LangChain message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [_message_text(item) for item in content]
        return "\n".join(part for part in text_parts if part).strip()

    text = content.get("text")
    if isinstance(text, str):
        return text
    return json.dumps(content, default=str)


def _tool_names_summary(tools: list[BaseTool]) -> str:
    """Group tool names by server prefix for the planner prompt."""
    by_server: dict[str, list[str]] = {}
    for t in tools:
        parts = t.name.split("__", 1)
        server = parts[0] if len(parts) == 2 else "other"
        name = parts[1] if len(parts) == 2 else t.name
        by_server.setdefault(server, []).append(name)
    lines = []
    for server, names in sorted(by_server.items()):
        lines.append(f"  {server}: {', '.join(sorted(names))}")
    return "\n".join(lines)


def build_plan_and_execute_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    *,
    safe_tool_filter: bool = True,
    record_usage: Callable[[object], None] | None = None,
) -> StateGraph:
    """Build a 4-node StateGraph: planner -> executor (react) -> reviewer -> compiler.

    Args:
        model: LangChain chat model to use for all nodes.
        tools: LangChain tools from ``build_langchain_tools``.
        safe_tool_filter: If True, filter out destructive tools (delete, send).
    """
    tools_summary = _tool_names_summary(tools)

    if safe_tool_filter:
        exec_tools = [
            t
            for t in tools
            if not any(
                d in t.name.lower() for d in ["delete", "send_email", "create_email"]
            )
        ]
    else:
        exec_tools = list(tools)

    react_executor = create_react_agent(
        model,
        exec_tools,
        prompt=SystemMessage(
            content=(
                "You are an execution agent. Use the available tools to "
                "complete the step you are given. Call as many tools as needed. "
                "After gathering data, summarize your findings."
            )
        ),
    )

    # ---- planner ----
    def planner(state: PlanExecState) -> dict[str, Any]:
        plan_prompt = SystemMessage(
            content=(
                "You are a planning agent. Given a task, produce a JSON array of steps "
                "to complete it. Each step should be a short action string. "
                "Return ONLY a JSON array of strings, no other text. "
                "Keep the plan to 4-5 steps.\n\n"
                "Available tool servers:\n"
                f"{tools_summary}"
            )
        )
        response = model.invoke([plan_prompt, HumanMessage(content=state["task"])])
        if record_usage is not None:
            record_usage(getattr(response, "usage_metadata", None))
        text = _message_text(response.content).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            steps_list = json.loads(text)
        except json.JSONDecodeError:
            steps_list = [text]
        return {
            "plan": [PlanStep(step=s, done=False) for s in steps_list],
            "current_step_idx": 0,
        }

    # ---- executor ----
    def executor(state: PlanExecState) -> dict[str, Any]:
        idx = state["current_step_idx"]
        plan = state["plan"]
        if idx >= len(plan):
            return {}
        step = plan[idx]
        prior = state.get("step_results", [])
        context = (
            ("Results from prior steps:\n" + "\n".join(prior) + "\n\n") if prior else ""
        )
        instruction = f"{context}Execute step {idx + 1}/{len(plan)}: {step['step']}"
        result = react_executor.invoke(
            {"messages": [HumanMessage(content=instruction)]}
        )
        if record_usage is not None:
            for message in result.get("messages", []):
                record_usage(getattr(message, "usage_metadata", None))
        final_text = ""
        for m in reversed(result.get("messages", [])):
            if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                final_text = _message_text(m.content)
                break
        return {"step_results": [f"Step {idx + 1}: {final_text}"]}

    # ---- reviewer ----
    def reviewer(state: PlanExecState) -> dict[str, Any]:
        plan = list(state["plan"])
        idx = state["current_step_idx"]
        if idx < len(plan):
            plan[idx] = PlanStep(step=plan[idx]["step"], done=True)
        return {"plan": plan, "current_step_idx": idx + 1}

    def should_continue(state: PlanExecState) -> str:
        return (
            "executor" if state["current_step_idx"] < len(state["plan"]) else "compiler"
        )

    # ---- compiler ----
    def compiler(state: PlanExecState) -> dict[str, Any]:
        results_text = "\n\n".join(state.get("step_results", []))
        response = model.invoke(
            [
                SystemMessage(content="You are a report compiler."),
                HumanMessage(
                    content=(
                        f"Original task:\n{state['task']}\n\n"
                        f"Completed step results:\n{results_text}\n\n"
                        "Produce the final response. Follow the task's formatting instructions exactly."
                    )
                ),
            ]
        )
        if record_usage is not None:
            record_usage(getattr(response, "usage_metadata", None))
        return {"final_output": _message_text(response.content)}

    # ---- graph ----
    graph = StateGraph(PlanExecState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("reviewer", reviewer)
    graph.add_node("compiler", compiler)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges(
        "reviewer", should_continue, {"executor": "executor", "compiler": "compiler"}
    )
    graph.add_edge("compiler", END)
    return graph
