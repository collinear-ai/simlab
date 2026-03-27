"""Framework-neutral event sinks for writing agent activity into SimLab artifacts."""

from __future__ import annotations

import json
from typing import Any
from typing import Protocol

from simlab.agents.base import RunArtifacts
from simlab.agents.base import ToolCall
from simlab.agents.base import ToolCallResult


class ToolResultLike(Protocol):
    """Structural shape for tool execution results across agent frameworks."""

    observation: Any
    is_error: bool


class ToolEventRecorder(Protocol):
    """Framework-neutral interface for recording agent activity."""

    def on_user_message(self, content: str) -> None:
        """Record a user-originated message."""

    def on_tool_invocation(
        self,
        tool_call_id: str,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
        result: ToolResultLike,
    ) -> None:
        """Record one tool invocation and its result."""

    def on_assistant_tool_call(
        self,
        tool_call_id: str,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> None:
        """Record one assistant-originated tool call request."""

    def on_assistant_message(self, content: str) -> None:
        """Record an assistant-originated message."""


class RunArtifactsRecorder:
    """Write framework-neutral events into SimLab ``RunArtifacts``."""

    def __init__(self, context: RunArtifacts) -> None:
        """Bind the recorder to the run artifact container for one execution."""
        self._context = context

    def on_user_message(self, content: str) -> None:
        """Persist a user-originated message to the run transcript."""
        self._context.record_message("user", content)

    def on_tool_invocation(
        self,
        tool_call_id: str,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
        result: ToolResultLike,
    ) -> None:
        """Persist one tool call, result payload, and synthetic tool message."""
        self._context.record_tool_call(
            ToolCall(
                tool_server=tool_server,
                tool_name=tool_name,
                parameters=parameters,
            ),
            ToolCallResult(
                observation=result.observation,
                is_error=result.is_error,
            ),
        )
        self._context.record_message(
            "tool",
            build_artifact_tool_message_content(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result=result,
            ),
        )

    def on_assistant_tool_call(
        self,
        tool_call_id: str,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> None:
        """Persist one assistant tool call using the reference-agent transcript shape."""
        self._context.record_message(
            "assistant",
            build_artifact_assistant_tool_call_content(
                tool_call_id=tool_call_id,
                tool_server=tool_server,
                tool_name=tool_name,
                parameters=parameters,
            ),
        )

    def on_assistant_message(self, content: str) -> None:
        """Persist an assistant-originated message to the run transcript."""
        self._context.record_message("assistant", content)


def build_artifact_assistant_tool_call_content(
    *,
    tool_call_id: str,
    tool_server: str,
    tool_name: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Build the assistant transcript payload for one requested tool call."""
    return {
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": f"{tool_server}__{tool_name}",
                    "arguments": json.dumps(parameters, sort_keys=True),
                },
            }
        ],
    }


def build_artifact_tool_message_content(
    *,
    tool_call_id: str,
    tool_name: str,
    result: ToolResultLike,
) -> dict[str, Any]:
    """Build the normalized tool-message payload stored in run artifacts."""
    summary: str | None = None
    obs = result.observation
    if isinstance(obs, dict):
        text = obs.get("text")
        if isinstance(text, str):
            summary = text
        nested = obs.get("observation")
        if summary is None and isinstance(nested, dict):
            nested_text = nested.get("text")
            if isinstance(nested_text, str):
                summary = nested_text
    elif isinstance(obs, str):
        summary = obs

    payload: dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "is_error": result.is_error,
    }
    if summary:
        payload["summary"] = summary
    return payload
