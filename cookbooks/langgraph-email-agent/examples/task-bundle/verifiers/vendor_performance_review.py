"""Programmatic verifier for the vendor performance review task."""

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
    """Check that the response includes a structured vendor performance report."""
    final_output = (run_artifacts.final_observation or "").strip()
    if not final_output:
        return False, "The agent did not produce a final response."

    # --- Heading structure ---
    headings = re.findall(r"(?m)^## .+$", final_output)
    expected_headings = [
        "## Vendor Performance Report",
        "## Action Items",
    ]
    if headings != expected_headings:
        return False, f"Expected headings {expected_headings}, found {headings}."

    # --- Tool usage: must use both email and EDGAR ---
    tool_messages = _tool_messages(run_artifacts)
    email_calls = sum(1 for m in tool_messages if m.get("tool_server") == "email-env")
    edgar_calls = sum(
        1
        for m in tool_messages
        if m.get("tool_server") == "edgar-mcp" and m.get("tool_name") == "get_company_info"
    )
    if email_calls < 2:
        return False, f"Expected at least 2 email tool calls, found {email_calls}."
    if edgar_calls < 1:
        return False, "Expected at least 1 SEC EDGAR get_company_info call."

    # --- Fact checks ---
    normalized = final_output.lower()
    fact_checks = {
        "IBM vendor reference": "ibm" in normalized,
        "Oracle vendor reference": "oracle" in normalized,
        "CloudFlare vendor reference": "cloudflare" in normalized,
        "IBM delivery delay": "delay" in normalized or "late" in normalized,
        "Oracle invoice dispute": "dispute" in normalized
        or "overage" in normalized
        or "185" in final_output,
        "CloudFlare compliance gap": "soc 2" in normalized or "compliance" in normalized,
        "CloudFlare auto-renewal": "auto-renew" in normalized
        or "72,000" in final_output
        or "72000" in final_output
        or "15%" in final_output,
        "Wednesday deadline": "wednesday" in normalized,
        "IBM ticker": "ibm" in normalized,
        "Oracle ticker": "orcl" in normalized,
        "Public company context present": "public company context:" in normalized,
    }
    missing_facts = [name for name, passed in fact_checks.items() if not passed]
    if missing_facts:
        return False, f"Response is missing expected facts: {', '.join(missing_facts)}"

    # --- Required fields ---
    required_fields = [
        "Vendor:",
        "Issue Summary:",
        "Risk Rating:",
    ]
    missing_fields = [f for f in required_fields if f not in final_output]
    if missing_fields:
        return False, f"Report is missing fields: {', '.join(missing_fields)}"

    # --- Action items ---
    actions_section = final_output.split("## Action Items")[-1]
    todo_lines = re.findall(r"(?m)^- \[(P[123])\] .+$", actions_section)
    if len(todo_lines) < 4:
        return (
            False,
            f"Action Items must have at least 4 prioritized bullets, found {len(todo_lines)}.",
        )
    if "P1" not in todo_lines:
        return False, "Action Items must include at least one [P1] action."

    return (
        True,
        "Response captured vendor performance data from emails "
        "and SEC EDGAR with structured actions.",
    )
