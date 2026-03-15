"""Base agent/environment contracts and run artifact models."""

from __future__ import annotations

import json
from abc import ABC
from abc import abstractmethod
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation made by an agent."""

    tool_server: str
    tool_name: str
    parameters: dict[str, Any]


@dataclass
class ToolCallResult:
    """Result payload returned by the tool server."""

    observation: Any
    is_error: bool = False


@dataclass
class RunArtifacts:
    """Serializable run artifact for external agent executions."""

    version: str = "0.1"
    task_id: str = ""
    task: str = ""
    model: str | None = None
    provider: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolCallResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    final_observation: str | None = None
    error: str | None = None
    steps_taken: int = 0
    max_steps: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def record_message(self, role: str, content: str | dict[str, object]) -> None:
        """Append a message (role + content) to this run's message list."""
        self.messages.append({"role": role, "content": content})

    def record_tool_call(self, call: ToolCall, result: ToolCallResult) -> None:
        """Record a tool call and its result, incrementing steps taken."""
        self.tool_calls.append(call)
        self.tool_results.append(result)
        self.steps_taken += 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this run's artifacts."""
        data = asdict(self)
        data["tool_calls"] = [asdict(x) for x in self.tool_calls]
        data["tool_results"] = [asdict(x) for x in self.tool_results]
        return data

    def dump(self, path: Path) -> None:
        """Write artifacts to a JSON file at the given path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


class BaseEnvironment(ABC):
    """Environment abstraction with tool listing and invocation."""

    @abstractmethod
    def list_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        """Return tool definitions for one server or all servers."""

    @abstractmethod
    def call_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        """Invoke a tool on a target tool server."""

    @property
    @abstractmethod
    def tool_servers(self) -> dict[str, str]:
        """Mapping of server name to base URL."""


class BaseAgent(ABC):
    """External agent contract."""

    @staticmethod
    def name() -> str:
        """Return the name of the agent."""
        return "base-agent"

    def version(self) -> str | None:
        """Return the version of the agent."""
        return None

    @abstractmethod
    def setup(self, environment: BaseEnvironment) -> None:
        """Initialize the agent before run."""

    @abstractmethod
    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
    ) -> None:
        """Execute the instruction and populate RunArtifacts."""
