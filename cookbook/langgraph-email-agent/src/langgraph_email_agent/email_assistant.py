"""Standalone graph-based LangGraph email assistant."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import import_module
from textwrap import dedent
from typing import Any
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langgraph_email_agent.types import AssistantRecorder
from langgraph_email_agent.types import AssistantRunResult
from langgraph_email_agent.types import NullRecorder

try:
    create_agent = import_module("langchain.agents").create_agent
except ModuleNotFoundError:
    create_agent = import_module("langgraph.prebuilt").create_react_agent


@dataclass(frozen=True)
class AssistantWorkflow:
    """An opinionated built-in workflow the assistant can apply."""

    name: str
    description: str
    instructions: tuple[str, ...]


class AssistantState(TypedDict, total=False):
    """Mutable graph state for one assistant run."""

    task: str
    requested_sections: list[str]
    mailbox_owner: str
    inbox_research: str
    sales_research: str
    counterparty_account: str
    counterparty_ticker: str
    counterparty_research: str
    inbox_triage: str
    todo_list: str
    sales_call_brief: str
    meeting_prep_packet: str
    final_output: str
    validation_error: str | None
    revision_count: int


def _stringify_observation(observation: object) -> str:
    if isinstance(observation, str):
        return observation
    try:
        return json.dumps(observation, indent=2, sort_keys=True, default=str)
    except TypeError:
        return str(observation)


def _message_text(message: object) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            item.get("text", "") for item in content if isinstance(item, dict)
        ]
        return "\n".join(part for part in text_parts if part).strip()
    return str(content)


def _extract_final_output(result: object) -> str:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            return _message_text(messages[-1]).strip()
    return str(result)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Expected a JSON object, got: {text}")


def _strip_repeated_headings(text: str) -> str:
    lines = text.strip().splitlines()
    cleaned = [line for line in lines if not line.strip().startswith("## ")]
    return "\n".join(cleaned).strip()


class LangGraphEmailAssistant:
    """Run a graph-based email assistant over any LangChain-compatible chat model."""

    def __init__(
        self,
        model: BaseChatModel,
        tool_recursion_limit: int = 8,
        workflow_recursion_limit: int = 16,
        workflows: list[AssistantWorkflow] | None = None,
        mailbox_owner: str | None = None,
    ) -> None:
        self._model = model
        self._tool_recursion_limit = tool_recursion_limit
        self._workflow_recursion_limit = workflow_recursion_limit
        self._workflows = workflows or self._default_workflows()
        self._mailbox_owner = mailbox_owner

    @staticmethod
    def _default_workflows() -> list[AssistantWorkflow]:
        return [
            AssistantWorkflow(
                name="inbox-triage",
                description="Rank email threads by urgency, business impact, and deadline risk.",
                instructions=(
                    "Separate urgent work from background noise such as newsletters or FYI threads.",
                    "Prioritize executive asks, customer risk, revenue impact, and deadlines over general updates.",
                    "When you present a triage summary, cite the concrete trigger such as a due date, blocker, or stakeholder.",
                ),
            ),
            AssistantWorkflow(
                name="todo-builder",
                description="Convert important email threads into a short, prioritized action plan.",
                instructions=(
                    "Express todos as concrete next actions, not vague themes.",
                    "Preserve any dates, owners, and meeting lengths mentioned in the source emails.",
                    "Prefer a compact list with explicit priorities such as [P1], [P2], and [P3].",
                ),
            ),
            AssistantWorkflow(
                name="sales-call-brief",
                description="Normalize sales or renewal threads into a concise deal brief.",
                instructions=(
                    "For sales-related threads, identify the account, stage, key pains, blockers, and next committed step.",
                    "Call out commercial or procurement deadlines when they appear.",
                    "Keep the brief operational so a sales lead could act on it immediately.",
                ),
            ),
            AssistantWorkflow(
                name="meeting-prep-packet",
                description="Prepare a concise external meeting brief from inbox threads plus public company context.",
                instructions=(
                    "Identify the concrete objective for the meeting and the business stakes if it goes poorly.",
                    "Propose a focused agenda, open questions, and the specific prep items needed before the call.",
                    "When the counterparty is public, include a relevant financial context fact grounded in tool output.",
                ),
            ),
        ]

    def workflow_names(self) -> list[str]:
        """Return the names of the built-in graph workflows."""
        return [workflow.name for workflow in self._workflows]

    def _build_system_prompt(self) -> str:
        workflow_blocks = []
        for workflow in self._workflows:
            instruction_lines = "\n".join(f"- {line}" for line in workflow.instructions)
            workflow_blocks.append(
                f"{workflow.name}: {workflow.description}\n{instruction_lines}"
            )
        workflows_text = "\n\n".join(workflow_blocks)
        mailbox_scope = ""
        if self._mailbox_owner:
            mailbox_scope = (
                "Only inspect or summarize messages whose To field includes "
                f"{self._mailbox_owner}. When searching email, prefer starting with "
                f"search_emails(kind='to', query='{self._mailbox_owner}')."
            )
        return dedent(
            f"""
            You are a rigorous email operations assistant.

            Use tools before making factual claims about inbox contents.
            Treat tool results as the source of truth.
            Read enough email to support deadlines, stakeholders, and recommendations.
            Ignore low-signal noise unless the task explicitly asks for it.
            {mailbox_scope}

            Built-in workflows:
            {workflows_text}
            """
        ).strip()

    def _section_heading(self, section: str) -> str:
        return {
            "inbox_triage": "## Inbox Triage",
            "todo_list": "## Todo List",
            "sales_call_brief": "## Sales Call Brief",
            "meeting_prep_packet": "## Meeting Prep Packet",
        }[section]

    def _infer_requested_sections(self, task: str) -> list[str]:
        normalized = task.lower()
        requested: list[str] = []
        matches = {
            "inbox_triage": (
                "## inbox triage" in normalized
                or "inbox triage" in normalized
                or "triage" in normalized
            ),
            "todo_list": "## todo list" in normalized or "todo" in normalized,
            "sales_call_brief": (
                "## sales call brief" in normalized
                or "sales call brief" in normalized
                or "renewal" in normalized
                or "sales" in normalized
                or "counterparty" in normalized
            ),
            "meeting_prep_packet": (
                "## meeting prep packet" in normalized
                or "meeting prep packet" in normalized
                or "meeting prep" in normalized
                or "prep packet" in normalized
            ),
        }
        for section in (
            "inbox_triage",
            "todo_list",
            "sales_call_brief",
            "meeting_prep_packet",
        ):
            if matches[section]:
                requested.append(section)
        if not requested:
            requested = ["inbox_triage", "todo_list"]
        return requested

    def _run_tool_research(
        self,
        prompt: str,
        tools: list[BaseTool],
    ) -> str:
        if not tools:
            raise ValueError("LangGraphEmailAssistant requires at least one tool")
        graph = create_agent(self._model, tools)
        result = graph.invoke(
            {
                "messages": [
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": prompt},
                ]
            },
            config={"recursion_limit": self._tool_recursion_limit},
        )
        return _extract_final_output(result).strip()

    def _run_text_step(self, system_prompt: str, user_prompt: str) -> str:
        response = self._model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return _message_text(response).strip()

    def _build_graph(
        self,
        tools: list[BaseTool],
    ) -> CompiledStateGraph[AssistantState, None, AssistantState, AssistantState]:
        def plan_request(state: AssistantState) -> AssistantState:
            task = state["task"]
            requested_sections = self._infer_requested_sections(task)
            return {
                "requested_sections": requested_sections,
                "mailbox_owner": self._mailbox_owner or "",
                "revision_count": 0,
                "validation_error": None,
            }

        def research_inbox(state: AssistantState) -> AssistantState:
            prompt = dedent(
                f"""
                Inspect the inbox and gather only the information needed for this task:
                {state["task"]}

                Restrict your inbox work to messages sent to this mailbox owner:
                {state.get("mailbox_owner", "") or "the primary mailbox referenced by the task"}

                Return concise research notes covering:
                - the most urgent threads
                - deadlines, stakeholders, and blockers
                - which threads are low priority noise
                """
            ).strip()
            return {"inbox_research": self._run_tool_research(prompt, tools)}

        def research_sales(state: AssistantState) -> AssistantState:
            requested_sections = state.get("requested_sections", [])
            if (
                "sales_call_brief" not in requested_sections
                and "meeting_prep_packet" not in requested_sections
            ):
                return {}
            prompt = dedent(
                f"""
                Focus only on the sales, renewal, or external meeting thread relevant to this task:
                {state["task"]}

                Restrict your inbox work to messages sent to this mailbox owner:
                {state.get("mailbox_owner", "") or "the primary mailbox referenced by the task"}

                Read the necessary emails and return concise notes with:
                - account name
                - deal stage
                - key pains
                - blockers
                - deadlines
                - next committed step
                - participants or stakeholders relevant to the meeting
                - the most important discussion topics to prepare for
                """
            ).strip()
            return {"sales_research": self._run_tool_research(prompt, tools)}

        def infer_counterparty_target(state: AssistantState) -> AssistantState:
            requested_sections = state.get("requested_sections", [])
            if (
                "sales_call_brief" not in requested_sections
                and "meeting_prep_packet" not in requested_sections
            ):
                return {}
            user_prompt = dedent(
                f"""
                Infer the public-company counterparty for the sales or renewal thread.

                Task:
                {state["task"]}

                Sales research:
                {state.get("sales_research", "")}

                Return strict JSON with exactly these keys:
                {{
                  "account": "<company name>",
                  "ticker": "<public ticker symbol>"
                }}
                """
            ).strip()
            target = _extract_json_object(
                self._run_text_step(self._build_system_prompt(), user_prompt)
            )
            account = str(target.get("account", "")).strip()
            ticker = str(target.get("ticker", "")).strip().upper()
            if not account or not ticker:
                raise ValueError("Could not infer counterparty account and ticker")
            return {
                "counterparty_account": account,
                "counterparty_ticker": ticker,
            }

        def research_counterparty(state: AssistantState) -> AssistantState:
            requested_sections = state.get("requested_sections", [])
            if (
                "sales_call_brief" not in requested_sections
                and "meeting_prep_packet" not in requested_sections
            ):
                return {}
            prompt = dedent(
                f"""
                Look up the public-company counterparty for this sales thread.

                Account: {state.get("counterparty_account", "")}
                Ticker: {state.get("counterparty_ticker", "")}

                Use the available company-research tools and return concise notes with:
                - official company name
                - ticker
                - latest reported revenue or comparable top-line scale metric
                - fiscal year context if available
                - one useful public-company context fact relevant to this meeting or deal
                """
            ).strip()
            return {"counterparty_research": self._run_tool_research(prompt, tools)}

        def compose_inbox_triage(state: AssistantState) -> AssistantState:
            if "inbox_triage" not in state.get("requested_sections", []):
                return {}
            user_prompt = dedent(
                f"""
                Task:
                {state["task"]}

                Inbox research:
                {state.get("inbox_research", "")}

                Write only the body for a section headed `## Inbox Triage`.
                Use exactly three markdown bullets.
                Each bullet should identify the thread and explain why it matters.
                Do not repeat the heading.
                """
            ).strip()
            return {
                "inbox_triage": self._run_text_step(
                    self._build_system_prompt(), user_prompt
                )
            }

        def compose_todo_list(state: AssistantState) -> AssistantState:
            if "todo_list" not in state.get("requested_sections", []):
                return {}
            user_prompt = dedent(
                f"""
                Task:
                {state["task"]}

                Inbox research:
                {state.get("inbox_research", "")}

                Write only the body for a section headed `## Todo List`.
                Include at least four concrete markdown bullets.
                Format each bullet as `- [P1] ...`, `- [P2] ...`, or `- [P3] ...`.
                Preserve deadlines and owners when present.
                Do not repeat the heading.
                """
            ).strip()
            return {
                "todo_list": self._run_text_step(
                    self._build_system_prompt(), user_prompt
                )
            }

        def compose_sales_call_brief(state: AssistantState) -> AssistantState:
            if "sales_call_brief" not in state.get("requested_sections", []):
                return {}
            user_prompt = dedent(
                f"""
                Task:
                {state["task"]}

                Inbox research:
                {state.get("inbox_research", "")}

                Sales research:
                {state.get("sales_research", "")}

                Counterparty research:
                {state.get("counterparty_research", "")}

                Write only the body for a section headed `## Sales Call Brief`.
                Use exactly these labels:
                Account:
                Ticker:
                Stage:
                Key Pains:
                Blockers:
                Next Step:
                Public Company Context:
                Do not repeat the heading.
                Do not include prefatory notes, tool commentary, or self-referential text.
                If a fact cannot be established, use `Unknown` after the relevant label.
                """
            ).strip()
            return {
                "sales_call_brief": self._run_text_step(
                    self._build_system_prompt(), user_prompt
                )
            }

        def compose_meeting_prep_packet(state: AssistantState) -> AssistantState:
            if "meeting_prep_packet" not in state.get("requested_sections", []):
                return {}
            user_prompt = dedent(
                f"""
                Task:
                {state["task"]}

                Inbox research:
                {state.get("inbox_research", "")}

                Meeting or sales research:
                {state.get("sales_research", "")}

                Counterparty research:
                {state.get("counterparty_research", "")}

                Write only the body for a section headed `## Meeting Prep Packet`.
                Use exactly these labels:
                Meeting:
                Objective:
                Agenda:
                Stakeholders:
                Risks:
                Open Questions:
                Next Step:
                Public Company Context:
                Do not repeat the heading.
                Do not include prefatory notes, tool commentary, or self-referential text.
                Keep the agenda and open questions concise but specific.
                If a fact cannot be established, use `Unknown` after the relevant label.
                """
            ).strip()
            return {
                "meeting_prep_packet": self._run_text_step(
                    self._build_system_prompt(), user_prompt
                )
            }

        def compile_final_response(state: AssistantState) -> AssistantState:
            parts: list[str] = []
            for section in state.get("requested_sections", []):
                body = _strip_repeated_headings(str(state.get(section, "")).strip())
                if not body:
                    continue
                parts.append(f"{self._section_heading(section)}\n{body}")
            return {"final_output": "\n\n".join(parts).strip()}

        def validate_final_response(state: AssistantState) -> AssistantState:
            final_output = state.get("final_output", "").strip()
            if not final_output:
                return {"validation_error": "Final output is empty."}
            headings = re.findall(r"(?m)^## .+$", final_output)
            expected_headings = [
                self._section_heading(section)
                for section in state.get("requested_sections", [])
            ]
            if headings != expected_headings:
                return {
                    "validation_error": (
                        f"Expected headings {expected_headings}, found {headings}."
                    )
                }
            for section in state.get("requested_sections", []):
                heading = self._section_heading(section)
                if heading not in final_output:
                    return {"validation_error": f"Missing required heading: {heading}"}
            if (
                "todo_list" in state.get("requested_sections", [])
                and "[P1]" not in final_output
            ):
                return {
                    "validation_error": "Todo list must contain at least one [P1] action."
                }
            if "sales_call_brief" in state.get("requested_sections", []):
                required_labels = [
                    "Account:",
                    "Ticker:",
                    "Stage:",
                    "Key Pains:",
                    "Blockers:",
                    "Next Step:",
                    "Public Company Context:",
                ]
                for label in required_labels:
                    if label not in final_output:
                        return {
                            "validation_error": f"Sales call brief is missing label: {label}"
                        }
                sales_section = final_output.split("## Sales Call Brief", maxsplit=1)[
                    -1
                ]
                if "unable to access" in sales_section.lower():
                    return {
                        "validation_error": (
                            "Sales call brief should not include tool-access commentary."
                        )
                    }
            if "meeting_prep_packet" in state.get("requested_sections", []):
                required_labels = [
                    "Meeting:",
                    "Objective:",
                    "Agenda:",
                    "Stakeholders:",
                    "Risks:",
                    "Open Questions:",
                    "Next Step:",
                    "Public Company Context:",
                ]
                for label in required_labels:
                    if label not in final_output:
                        return {
                            "validation_error": (
                                f"Meeting prep packet is missing label: {label}"
                            )
                        }
                meeting_section = final_output.split(
                    "## Meeting Prep Packet", maxsplit=1
                )[-1]
                if "unable to access" in meeting_section.lower():
                    return {
                        "validation_error": (
                            "Meeting prep packet should not include tool-access commentary."
                        )
                    }
            return {"validation_error": None}

        def revise_final_response(state: AssistantState) -> AssistantState:
            revision_count = state.get("revision_count", 0) + 1
            user_prompt = dedent(
                f"""
                Revise the response so it satisfies the task and validation error.

                Task:
                {state["task"]}

                Validation error:
                {state.get("validation_error", "")}

                Current response:
                {state.get("final_output", "")}

                Inbox research:
                {state.get("inbox_research", "")}

                Sales research:
                {state.get("sales_research", "")}

                Counterparty research:
                {state.get("counterparty_research", "")}

                Preserve the exact section headings once each.
                Remove any repeated headings or tool-access commentary.
                """
            ).strip()
            return {
                "final_output": self._run_text_step(
                    self._build_system_prompt(), user_prompt
                ),
                "revision_count": revision_count,
            }

        def after_validation(state: AssistantState) -> str:
            if state.get("validation_error") and state.get("revision_count", 0) < 1:
                return "revise_final_response"
            return END

        graph: StateGraph[AssistantState, None, AssistantState, AssistantState] = (
            StateGraph(
                AssistantState,
                input_schema=AssistantState,
                output_schema=AssistantState,
            )
        )
        graph.add_node("plan_request", plan_request)
        graph.add_node("research_inbox", research_inbox)
        graph.add_node("research_sales", research_sales)
        graph.add_node("infer_counterparty_target", infer_counterparty_target)
        graph.add_node("research_counterparty", research_counterparty)
        graph.add_node("compose_inbox_triage", compose_inbox_triage)
        graph.add_node("compose_todo_list", compose_todo_list)
        graph.add_node("compose_sales_call_brief", compose_sales_call_brief)
        graph.add_node("compose_meeting_prep_packet", compose_meeting_prep_packet)
        graph.add_node("compile_final_response", compile_final_response)
        graph.add_node("validate_final_response", validate_final_response)
        graph.add_node("revise_final_response", revise_final_response)
        graph.add_edge(START, "plan_request")
        graph.add_edge("plan_request", "research_inbox")
        graph.add_edge("research_inbox", "research_sales")
        graph.add_edge("research_sales", "infer_counterparty_target")
        graph.add_edge("infer_counterparty_target", "research_counterparty")
        graph.add_edge("research_counterparty", "compose_inbox_triage")
        graph.add_edge("compose_inbox_triage", "compose_todo_list")
        graph.add_edge("compose_todo_list", "compose_sales_call_brief")
        graph.add_edge("compose_sales_call_brief", "compose_meeting_prep_packet")
        graph.add_edge("compose_meeting_prep_packet", "compile_final_response")
        graph.add_edge("compile_final_response", "validate_final_response")
        graph.add_conditional_edges(
            "validate_final_response",
            after_validation,
            {"revise_final_response": "revise_final_response", END: END},
        )
        graph.add_edge("revise_final_response", "validate_final_response")
        return graph.compile()

    def run(
        self,
        task: str,
        tools: list[BaseTool],
        recorder: AssistantRecorder | None = None,
    ) -> AssistantRunResult:
        """Execute the task through an explicit LangGraph workflow."""
        active_recorder = recorder or NullRecorder()
        active_recorder.on_user_message(task)
        graph: CompiledStateGraph[
            AssistantState, None, AssistantState, AssistantState
        ] = self._build_graph(tools)
        result = graph.invoke(
            {"task": task},
            config={"recursion_limit": self._workflow_recursion_limit},
        )
        final_output = str(result.get("final_output", "")).strip()
        if final_output:
            active_recorder.on_assistant_message(final_output)
        return AssistantRunResult(final_output=final_output)
