"""Programmatic verifier for the deal risk assessment cookbook task."""

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
    """Check that the response includes a structured deal risk assessment."""
    final_output = (run_artifacts.final_observation or "").strip()
    if not final_output:
        return False, "The agent did not produce a final response."

    # --- Heading structure ---
    headings = re.findall(r"(?m)^## .+$", final_output)
    expected_headings = [
        "## Deal Risk Assessment",
        "## Recommended Actions",
    ]
    if headings != expected_headings:
        return False, f"Expected headings {expected_headings}, found {headings}."

    # --- Tool usage ---
    tool_messages = _tool_messages(run_artifacts)
    email_call_count = sum(
        1 for message in tool_messages if message.get("tool_server") == "email-env"
    )
    if email_call_count < 2:
        return False, f"Expected at least 2 email tool calls, found {email_call_count}."
    if not any(
        message.get("tool_server") == "edgar-mcp" and message.get("tool_name") == "get_company_info"
        for message in tool_messages
    ):
        return False, "The run did not record the required MCP SEC EDGAR company lookup."

    # --- Fact checks ---
    normalized_output = final_output.lower()
    fact_checks = {
        "Salesforce account reference": "salesforce" in normalized_output,
        "CRM ticker reference": "crm" in normalized_output,
        "Deal value reference": "2.4" in normalized_output or "2,400" in normalized_output,
        "April 15 deadline": "april 15" in normalized_output or "april15" in normalized_output,
        "Discount issue": "12%" in final_output or "12 percent" in normalized_output,
        "Data residency blocker": "data residency" in normalized_output
        or "eu data" in normalized_output,
        "Incident response SLA": "incident response" in normalized_output
        or "24-hour" in normalized_output
        or "24 hour" in normalized_output,
        "Competitor risk ServiceNow": "servicenow" in normalized_output,
        "NPS decline": "nps" in normalized_output or "net promoter" in normalized_output,
        "Public company context": "public company context:" in normalized_output,
    }
    missing_facts = [name for name, passed in fact_checks.items() if not passed]
    if missing_facts:
        return False, f"Response is missing expected facts: {', '.join(missing_facts)}"

    # --- Risk factors ---
    risk_section = final_output.split("## Recommended Actions")[0]
    risk_bullets = re.findall(r"(?m)^[-*] .+$", risk_section)
    if len(risk_bullets) < 3:
        return (
            False,
            f"Risk Factors must have at least 3 bullets, found {len(risk_bullets)}.",
        )

    # --- Required fields ---
    assessment_fields = [
        "Account:",
        "Ticker:",
        "Deal Value:",
        "Renewal Deadline:",
        "Risk Level:",
        "Risk Factors:",
        "Public Company Context:",
    ]
    missing_fields = [field for field in assessment_fields if field not in final_output]
    if missing_fields:
        return (
            False,
            f"Deal risk assessment is missing fields: {', '.join(missing_fields)}",
        )

    # --- Recommended actions ---
    actions_section = final_output.split("## Recommended Actions")[-1]
    todo_lines = re.findall(r"(?m)^- \[(P[123])\] .+$", actions_section)
    if len(todo_lines) < 4:
        return False, "Recommended Actions must contain at least four prioritized bullets."
    if "P1" not in todo_lines:
        return False, "Recommended Actions must include at least one [P1] action."

    return (
        True,
        "Response captured deal risks, email context, SEC EDGAR data, and structured actions.",
    )
