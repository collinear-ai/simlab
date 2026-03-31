"""Programmatic verifier for the meeting prep packet cookbook task."""

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
    """Check that the response includes a distinct meeting prep packet."""
    final_output = (run_artifacts.final_observation or "").strip()
    if not final_output:
        return False, "The agent did not produce a final response."

    headings = re.findall(r"(?m)^## .+$", final_output)
    expected_headings = [
        "## Todo List",
        "## Meeting Prep Packet",
    ]
    if headings != expected_headings:
        return False, f"Expected headings {expected_headings}, found {headings}."

    tool_messages = _tool_messages(run_artifacts)
    email_call_count = sum(
        1 for message in tool_messages if message.get("tool_server") == "email-env"
    )
    if email_call_count < 1:
        return False, "Expected at least 1 email tool call."
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
        "Adobe counterparty reference": "adobe" in normalized_output,
        "Ticker reference": "adbe" in normalized_output,
        "Meeting timing": "11am" in normalized_output or "11 am" in normalized_output,
        "Premium support commercial topic": "premium support" in normalized_output,
        "SSO security blocker": "sso" in normalized_output,
        "SOC 2 blocker": "soc 2" in normalized_output,
        "Audit rights legal issue": "audit" in normalized_output,
        "Finance discount guardrail": "8 percent" in normalized_output
        or "8%" in normalized_output,
        "Revenue or scale context": "revenue" in normalized_output
        or "annual" in normalized_output
        or "fiscal" in normalized_output,
    }
    missing_facts = [name for name, passed in fact_checks.items() if not passed]
    if missing_facts:
        return False, f"Response is missing expected facts: {', '.join(missing_facts)}"

    todo_lines = re.findall(r"(?m)^- \[(P[123])\] .+$", final_output)
    if len(todo_lines) < 4:
        return False, "The todo list must contain at least four prioritized bullets."
    if "P1" not in todo_lines:
        return False, "The todo list must include at least one [P1] action."

    packet_fields = [
        "Meeting:",
        "Objective:",
        "Agenda:",
        "Stakeholders:",
        "Risks:",
        "Open Questions:",
        "Next Step:",
        "Public Company Context:",
    ]
    missing_fields = [field for field in packet_fields if field not in final_output]
    if missing_fields:
        return (
            False,
            f"Meeting prep packet is missing fields: {', '.join(missing_fields)}",
        )

    return (
        True,
        "Response captured the meeting prep packet structure, email context, and SEC EDGAR research.",
    )
