"""Post-run accounting logic for rollout artifacts.

Pure data extraction — no CLI output or click dependency.
Used by ``cli/tasks.py`` and potentially ``parallel_daytona.py``
to build run summary data before rendering.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class RunSummaryData:
    """Extracted summary data from a completed rollout."""

    steps: int
    error: str | None
    tool_counts: dict[str, int]
    npc_msg_count: int
    final_observation: str


def count_tool_calls(artifacts: Any) -> dict[str, int]:  # noqa: ANN401
    """Count tool calls per server from run artifacts.

    Uses the first-class ``ToolCall.tool_server`` field when available
    (populated by ``RunArtifacts.record_tool_call``).  Falls back to
    parsing tool names from messages for agents that only record messages.
    """
    counts: Counter[str] = Counter()

    # Primary path: use ToolCall.tool_server from artifacts.tool_calls.
    tool_calls = getattr(artifacts, "tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            server = getattr(tc, "tool_server", "")
            if server:
                counts[server] += 1
        return dict(counts)

    # Fallback: parse server prefix from tool names in messages.
    for msg in getattr(artifacts, "messages", []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, dict):
            for tc in content.get("tool_calls", []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "") if isinstance(fn, dict) else ""
                if name:
                    server = name.split("__")[0] if "__" in name else name
                    counts[server] += 1
    return dict(counts)


def extract_run_summary(artifacts: Any) -> RunSummaryData:  # noqa: ANN401
    """Extract summary data from run artifacts."""
    metadata = getattr(artifacts, "metadata", {}) or {}
    return RunSummaryData(
        steps=artifacts.steps_taken,
        error=artifacts.error,
        tool_counts=count_tool_calls(artifacts),
        npc_msg_count=metadata.get("npc_chat_message_count", 0),
        final_observation=getattr(artifacts, "final_observation", None) or "",
    )
