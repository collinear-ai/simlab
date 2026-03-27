"""Programmatic verifier for the vendor ERP review task."""

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
    """Check that the response uses both email and ERP data."""
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

    # --- Tool usage: must use both email and ERP ---
    tool_messages = _tool_messages(run_artifacts)
    email_calls = sum(1 for m in tool_messages if m.get("tool_server") == "email-env")
    erp_calls = sum(1 for m in tool_messages if m.get("tool_server") == "erp-env")
    if email_calls < 2:
        return False, f"Expected at least 2 email tool calls, found {email_calls}."
    if erp_calls < 2:
        return False, f"Expected at least 2 ERP tool calls, found {erp_calls}."

    # --- Fact checks ---
    normalized = final_output.lower()
    fact_checks = {
        "NovaTech reference": "novatech" in normalized,
        "GlobalParts reference": "globalparts" in normalized or "global parts" in normalized,
        "Meridian reference": "meridian" in normalized,
        "NovaTech delay issue": "delay" in normalized or "late" in normalized,
        "Meridian quality issue": "quality" in normalized
        or "defect" in normalized
        or "failed" in normalized
        or "qa" in normalized,
        "GlobalParts payment issue": "overdue" in normalized or "payment" in normalized,
    }
    missing_facts = [name for name, passed in fact_checks.items() if not passed]
    if missing_facts:
        return False, f"Response is missing expected facts: {', '.join(missing_facts)}"

    # --- ERP Supplier IDs must appear (proves ERP was used) ---
    supplier_ids_found = re.findall(r"SUP-\d{3}", final_output)
    if len(supplier_ids_found) < 2:
        return (
            False,
            f"Report must include at least 2 ERP Supplier IDs (SUP-xxx), "
            f"found {len(supplier_ids_found)}.",
        )

    # --- Required fields ---
    required_fields = ["Supplier:", "Supplier ID:", "Issue Summary:", "Risk Rating:"]
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
        "Response captured vendor data from both emails and ERP with structured actions.",
    )
