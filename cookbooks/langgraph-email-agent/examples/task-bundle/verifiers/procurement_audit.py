"""Programmatic verifier for the procurement audit task.

This is deliberately strict — it checks for specific ERP data points
that can only be found by chaining multiple tool calls.
"""

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
    """Strict verification requiring specific ERP data points."""
    final_output = (run_artifacts.final_observation or "").strip()
    if not final_output:
        return False, "The agent did not produce a final response."

    # --- Heading structure ---
    headings = re.findall(r"(?m)^## .+$", final_output)
    expected = ["## Findings", "## Financial Summary", "## Remediation Plan"]
    if headings != expected:
        return False, f"Expected headings {expected}, found {headings}."

    # --- Tool usage breadth ---
    tool_messages = _tool_messages(run_artifacts)
    email_calls = sum(1 for m in tool_messages if m.get("tool_server") == "email-env")
    erp_calls = sum(1 for m in tool_messages if m.get("tool_server") == "erp-env")
    if email_calls < 2:
        return False, f"Expected at least 2 email tool calls, found {email_calls}."
    if erp_calls < 5:
        return False, f"Expected at least 5 ERP tool calls, found {erp_calls}."

    # --- ERP tool diversity (must use multiple ERP tool types) ---
    erp_tool_names = {
        m.get("tool_name") for m in tool_messages if m.get("tool_server") == "erp-env"
    }
    required_erp_tools = {"search_invoices", "get_order"}
    missing_tools = required_erp_tools - erp_tool_names
    if missing_tools:
        return False, f"Must use ERP tools: {missing_tools}. Used: {erp_tool_names}."

    normalized = final_output.lower()

    # --- Specific order IDs that must appear (proves ERP lookup) ---
    order_checks = {
        "ORD-3101": "ORD-3101" in final_output,
        "ORD-3106": "ORD-3106" in final_output,
    }
    missing_orders = [k for k, v in order_checks.items() if not v]
    if missing_orders:
        return False, f"Missing order references: {', '.join(missing_orders)}"

    # --- Invoice IDs ---
    invoice_checks = {
        "INV-5002": "INV-5002" in final_output,
        "INV-5006": "INV-5006" in final_output,
    }
    missing_invoices = [k for k, v in invoice_checks.items() if not v]
    if missing_invoices:
        return False, f"Missing invoice references: {', '.join(missing_invoices)}"

    # --- Supplier IDs (proves supplier lookup) ---
    if not re.search(r"SUP-\d{3}", final_output):
        return False, "Must include at least one ERP Supplier ID."

    # --- PO references ---
    po_checks = {
        "PO-7002": "PO-7002" in final_output,
        "PO-7004": "PO-7004" in final_output,
    }
    missing_pos = [k for k, v in po_checks.items() if not v]
    if missing_pos:
        return False, f"Missing PO references: {', '.join(missing_pos)}"

    # --- Customer names (proves order→customer lookup) ---
    customer_checks = {
        "Pinnacle": "pinnacle" in normalized,
        "Cascade": "cascade" in normalized,
    }
    missing_customers = [k for k, v in customer_checks.items() if not v]
    if missing_customers:
        return False, f"Missing customer references: {', '.join(missing_customers)}"

    # --- Financial Summary fields ---
    fin_fields = [
        "Total Overdue Amount:",
        "Total At-Risk Revenue:",
        "Suppliers With Open POs:",
    ]
    missing_fin = [f for f in fin_fields if f not in final_output]
    if missing_fin:
        return False, f"Financial Summary missing fields: {', '.join(missing_fin)}"

    # --- Financial Summary must contain actual dollar amounts ---
    fin_section = final_output.split("## Remediation Plan")[0].split("## Financial Summary")[-1]
    dollar_amounts = re.findall(r"\$[\d,]+\.?\d*", fin_section)
    if len(dollar_amounts) < 2:
        return (
            False,
            f"Financial Summary must include at least 2 dollar amounts, "
            f"found {len(dollar_amounts)}.",
        )

    # --- Supplier names in open POs section ---
    if not ("novatech" in normalized and "meridian" in normalized):
        return False, "Suppliers With Open POs must mention NovaTech and Meridian."

    # --- Remediation plan ---
    remediation = final_output.split("## Remediation Plan")[-1]
    todo_lines = re.findall(r"(?m)^- \[(P[123])\] .+$", remediation)
    if len(todo_lines) < 5:
        return (
            False,
            f"Remediation Plan must have at least 5 prioritized bullets, found {len(todo_lines)}.",
        )
    if "P1" not in todo_lines:
        return False, "Remediation Plan must include at least one [P1] action."

    # --- Findings must have numbered items with required fields ---
    findings = final_output.split("## Financial Summary")[0].split("## Findings")[-1]
    has_order_field = "Order:" in findings
    has_finding_field = "Finding:" in findings
    if not has_order_field or not has_finding_field:
        return False, "Findings must include 'Order:' and 'Finding:' fields."

    return (
        True,
        "Procurement audit verified: correct ERP data, "
        "cross-referenced findings, and structured remediation.",
    )
