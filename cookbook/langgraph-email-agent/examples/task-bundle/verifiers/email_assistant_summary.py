"""Programmatic verifier for the LangGraph email assistant cookbook task."""

from __future__ import annotations

import re
from typing import Any
from typing import Protocol


class _RunArtifactsLike(Protocol):
    final_observation: str | None
    messages: list[dict[str, Any]]


def _tool_messages(run_artifacts: _RunArtifactsLike) -> list[dict[str, Any]]:
    tool_messages: list[dict[str, Any]] = []
    for message in run_artifacts.messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, dict):
            tool_messages.append(content)
    return tool_messages


def verify(run_artifacts: _RunArtifactsLike) -> tuple[bool, str]:
    """Check that the response uses the required structure and facts."""
    final_output = (run_artifacts.final_observation or "").strip()
    if not final_output:
        return False, "The agent did not produce a final response."

    headings = re.findall(r"(?m)^## .+$", final_output)
    expected_headings = [
        "## Inbox Triage",
        "## Todo List",
        "## Sales Call Brief",
    ]
    if headings != expected_headings:
        return False, f"Expected headings {expected_headings}, found {headings}."

    tool_messages = _tool_messages(run_artifacts)
    email_call_count = sum(
        1 for message in tool_messages if message.get("tool_server") == "email-env"
    )
    if email_call_count < 2:
        return False, f"Expected at least 2 email tool calls, found {email_call_count}."
    if not any(
        message.get("tool_server") == "edgar-mcp"
        and message.get("tool_name") == "get_company_info"
        for message in tool_messages
    ):
        return (
            False,
            "The run did not record the required MCP SEC EDGAR company lookup.",
        )

    normalized_output = final_output.lower()
    fact_checks = {
        "Q2 hiring plan deadline": (
            "q2 hiring plan" in normalized_output and "thursday" in normalized_output
        ),
        "board recruiting funnel update": (
            "board update" in normalized_output and "recruit" in normalized_output
        ),
        "finance prep before Friday review": (
            "finance" in normalized_output and "friday" in normalized_output
        ),
        "Northstar customer risk": "northstar" in normalized_output
        and "training" in normalized_output,
        "Microsoft account reference": "microsoft" in normalized_output,
        "Microsoft ticker reference": "msft" in normalized_output,
        "Microsoft commercial deadline": "wednesday" in normalized_output
        and "quote" in normalized_output,
        "Microsoft security blocker": "sso" in normalized_output
        or "soc 2" in normalized_output,
        "Microsoft legal blocker": "legal" in normalized_output
        and "redline" in normalized_output,
        "Public company context": "public company context:" in normalized_output,
    }
    missing_facts = [name for name, passed in fact_checks.items() if not passed]
    if missing_facts:
        return False, f"Response is missing expected facts: {', '.join(missing_facts)}"

    todo_lines = re.findall(r"(?m)^- \[(P[123])\] .+$", final_output)
    if len(todo_lines) < 4:
        return False, "The todo list must contain at least four prioritized bullets."
    if "P1" not in todo_lines:
        return False, "The todo list must include at least one [P1] action."

    sales_fields = [
        "Account:",
        "Ticker:",
        "Stage:",
        "Key Pains:",
        "Blockers:",
        "Next Step:",
        "Public Company Context:",
    ]
    missing_fields = [field for field in sales_fields if field not in final_output]
    if missing_fields:
        return False, f"Sales call brief is missing fields: {', '.join(missing_fields)}"

    return (
        True,
        "Response captured the required inbox priorities, sales brief, and SEC EDGAR lookup.",
    )
