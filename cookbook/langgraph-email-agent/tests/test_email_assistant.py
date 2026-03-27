# ruff: noqa: D100, D101, D102, D103, S101

from __future__ import annotations

from typing import Any
from typing import cast

import pytest

from langgraph_email_agent import email_assistant as email_assistant_module
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph_email_agent.email_assistant import LangGraphEmailAssistant
from langgraph_email_agent.email_assistant import _extract_json_object
from langgraph_email_agent.email_assistant import _strip_repeated_headings


class _StubModel:
    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "_StubModel":
        _ = tools, kwargs
        return self


def test_email_assistant_system_prompt_exposes_built_in_workflows() -> None:
    assistant = LangGraphEmailAssistant(
        model=cast(BaseChatModel, _StubModel()),
        mailbox_owner="agent@weaverenterprises.com",
    )

    prompt = assistant._build_system_prompt()

    assert "Built-in workflows:" in prompt
    assert "inbox-triage" in prompt
    assert "todo-builder" in prompt
    assert "sales-call-brief" in prompt
    assert "meeting-prep-packet" in prompt
    assert "Use tools before making factual claims about inbox contents." in prompt
    assert "agent@weaverenterprises.com" in prompt


def test_email_assistant_exposes_default_workflow_names() -> None:
    assistant = LangGraphEmailAssistant(model=cast(BaseChatModel, _StubModel()))

    assert assistant.workflow_names() == [
        "inbox-triage",
        "todo-builder",
        "sales-call-brief",
        "meeting-prep-packet",
    ]


def test_infer_requested_sections_supports_meeting_prep_packet() -> None:
    assistant = LangGraphEmailAssistant(model=cast(BaseChatModel, _StubModel()))

    requested = assistant._infer_requested_sections(
        "Return exactly two sections: ## Todo List and ## Meeting Prep Packet."
    )

    assert requested == ["todo_list", "meeting_prep_packet"]


def test_extract_json_object_supports_wrapped_json() -> None:
    parsed = _extract_json_object('Result:\n{"account":"Microsoft","ticker":"MSFT"}')

    assert parsed == {"account": "Microsoft", "ticker": "MSFT"}


def test_strip_repeated_headings_removes_embedded_markdown_headers() -> None:
    cleaned = _strip_repeated_headings(
        "Tool note\n## Sales Call Brief\nAccount: Microsoft\nTicker: MSFT"
    )

    assert cleaned == "Tool note\nAccount: Microsoft\nTicker: MSFT"


def test_email_assistant_uses_distinct_tool_and_workflow_recursion_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, dict[str, int]] = {}

    class _InnerGraph:
        def invoke(
            self, payload: dict[str, Any], config: dict[str, int]
        ) -> dict[str, Any]:
            _ = payload
            seen["tool"] = config
            return {"messages": [type("_Message", (), {"content": "research"})()]}

    class _OuterGraph:
        def invoke(
            self, payload: dict[str, Any], config: dict[str, int]
        ) -> dict[str, Any]:
            _ = payload
            seen["workflow"] = config
            return {"final_output": "done"}

    assistant = LangGraphEmailAssistant(
        model=cast(BaseChatModel, _StubModel()),
        tool_recursion_limit=3,
        workflow_recursion_limit=17,
    )

    monkeypatch.setattr(
        email_assistant_module,
        "create_agent",
        lambda _model, _tools: _InnerGraph(),
    )
    monkeypatch.setattr(assistant, "_build_graph", lambda _tools: _OuterGraph())
    tools = [cast(BaseTool, object())]

    assert assistant._run_tool_research("inspect inbox", tools) == "research"
    assert assistant.run("triage inbox", tools).final_output == "done"
    assert seen["tool"] == {"recursion_limit": 3}
    assert seen["workflow"] == {"recursion_limit": 17}
