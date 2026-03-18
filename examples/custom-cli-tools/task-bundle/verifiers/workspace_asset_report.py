"""Verifier for the custom CLI tools example task."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any
from typing import Protocol


class _RunArtifactsLike(Protocol):
    tool_server_url: str | None

    def server_url(self, name: str) -> str | None: ...


def _call_tool(tool_server_url: str, tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps({"action": {"tool_name": tool_name, "parameters": parameters}}).encode()
    request = urllib.request.Request(  # noqa: S310
        f"{tool_server_url}/step",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def verify(run_artifacts: _RunArtifactsLike) -> tuple[bool, str]:
    """Verify the generated workspace report contains the expected facts."""
    tool_server_url = run_artifacts.server_url("coding-env") or run_artifacts.tool_server_url
    if not tool_server_url:
        return False, "coding-env tool server URL was not available to the verifier."

    report_result = _call_tool(
        tool_server_url,
        "read_file",
        {"path": "workspace_report.md"},
    )
    observation = report_result.get("observation", {})
    if observation.get("is_error"):
        return False, f"workspace_report.md missing: {observation.get('text', '')}"

    report_text = str(observation.get("text", "")).lower()
    checks = {
        "q4 revenue $12.5m": (
            "$12.5m" in report_text
            and re.search(r"(q4.*revenue|revenue.*q4)", report_text) is not None
        ),
        "18% growth": "18%" in report_text,
        "total headcount 120": (
            "120" in report_text
            and re.search(r"(headcount.*120|120.*headcount)", report_text) is not None
        ),
        "engineering 45": re.search(r"engineering\D*45", report_text) is not None,
        "sales 30": re.search(r"sales\D*30", report_text) is not None,
        "operations 20": re.search(r"operations\D*20", report_text) is not None,
        "finance 15": re.search(r"finance\D*15", report_text) is not None,
        "people 10": re.search(r"people\D*10", report_text) is not None,
    }
    missing = [name for name, present in checks.items() if not present]
    if missing:
        return False, f"workspace_report.md is missing expected facts: {', '.join(missing)}"

    return True, "workspace_report.md contains the expected PDF and spreadsheet facts"
