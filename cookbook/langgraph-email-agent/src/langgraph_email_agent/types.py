"""Shared types for the standalone assistant and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol

from simlab.agents.adapters.artifacts import ToolResultLike


@dataclass(frozen=True)
class ToolExecutionResult:
    """Transport-agnostic tool call result."""

    observation: Any
    is_error: bool = False


@dataclass(frozen=True)
class AssistantRunResult:
    """Final assistant response."""

    final_output: str


class AssistantRecorder(Protocol):
    """Structured event sink for assistant execution."""

    def on_user_message(self, content: str) -> None:
        """Record the user/task instruction."""

    def on_tool_invocation(
        self,
        tool_call_id: str,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
        result: ToolResultLike,
    ) -> None:
        """Record a tool call and result."""

    def on_assistant_tool_call(
        self,
        tool_call_id: str,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> None:
        """Record one assistant tool call request."""

    def on_assistant_message(self, content: str) -> None:
        """Record the final assistant response."""


class NullRecorder:
    """No-op recorder for standalone usage."""

    def on_user_message(self, content: str) -> None:
        """Ignore the user message."""
        _ = content

    def on_tool_invocation(
        self,
        tool_call_id: str,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
        result: ToolResultLike,
    ) -> None:
        """Ignore the tool call."""
        _ = tool_call_id, tool_server, tool_name, parameters, result

    def on_assistant_tool_call(
        self,
        tool_call_id: str,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> None:
        """Ignore the assistant tool call."""
        _ = tool_call_id, tool_server, tool_name, parameters

    def on_assistant_message(self, content: str) -> None:
        """Ignore the assistant message."""
        _ = content
