"""Verifier for the example custom coding task."""

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
    """Verify the example task wrote a plausible workspace summary."""
    tool_server_url = run_artifacts.server_url("coding-env") or run_artifacts.tool_server_url
    if not tool_server_url:
        return False, "coding-env tool server URL was not available to the verifier."

    summary_result = _call_tool(
        tool_server_url,
        "read_file",
        {"path": "workspace_summary.md"},
    )
    observation = summary_result.get("observation", {})
    if observation.get("is_error"):
        return False, f"workspace_summary.md missing: {observation.get('text', '')}"

    summary_text = str(observation.get("text", "")).lower()
    if len(summary_text.strip()) < 40:
        return (
            False,
            "workspace_summary.md exists but is too short to be a useful summary.",
        )

    expected_tools = ("read_file", "write_file", "list_dir", "run_command")
    mentioned_tools = [
        tool_name for tool_name in expected_tools if re.search(rf"\b{tool_name}\b", summary_text)
    ]
    if len(mentioned_tools) < 2:
        return (
            False,
            "workspace_summary.md should mention at least two available tools such as "
            "read_file, write_file, list_dir, or run_command.",
        )

    return True, "workspace_summary.md describes the available tools."
