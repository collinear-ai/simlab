"""Base agent/environment contracts and run artifact models."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import warnings
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Coroutine
from concurrent.futures import Future
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class ToolNamespace:
    """A named tool namespace exposed through a concrete transport."""

    name: str
    transport: str
    endpoint: str | None = None


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
    on_step: Callable[[int], None] | None = field(default=None, repr=False, compare=False)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def record_message(self, role: str, content: str | dict[str, object]) -> None:
        """Append a message (role + content) to this run's message list."""
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def record_tool_call(self, call: ToolCall, result: ToolCallResult) -> None:
        """Record a tool call and its result, incrementing steps taken."""
        self.tool_calls.append(call)
        self.tool_results.append(result)
        self.steps_taken += 1
        on_step = self.on_step
        if on_step is None:
            return
        try:
            on_step(self.steps_taken)
        except Exception:
            logger.debug("on_step callback raised", exc_info=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this run's artifacts."""
        data = asdict(self)
        data.pop("on_step", None)
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

    def list_tool_namespaces(self) -> list[ToolNamespace]:
        """Return available tool namespaces.

        Subclasses should override this. The default implementation adapts the
        deprecated ``tool_servers`` property for backwards compatibility.
        """
        return [
            ToolNamespace(name=name, transport="unknown", endpoint=endpoint)
            for name, endpoint in self.tool_servers.items()
        ]

    def list_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        """Return tool definitions through the deprecated sync compatibility path.

        Deprecated: use ``await alist_tools()`` instead. Will be removed in 0.4.0.
        """
        warnings.warn(
            "BaseEnvironment.list_tools() is deprecated; use await alist_tools() instead. "
            "This compatibility method will be removed in 0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _run_async_compat(self.alist_tools(tool_server))

    def call_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        """Invoke a tool through the deprecated sync compatibility path.

        Deprecated: use ``await acall_tool()`` instead. Will be removed in 0.4.0.
        """
        warnings.warn(
            "BaseEnvironment.call_tool() is deprecated; use await acall_tool() instead. "
            "This compatibility method will be removed in 0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _run_async_compat(self.acall_tool(tool_server, tool_name, parameters))

    @abstractmethod
    async def alist_tools(self, tool_server: str | None = None) -> list[dict[str, Any]]:
        """Return tool definitions for one server or all servers."""

    @abstractmethod
    async def acall_tool(
        self,
        tool_server: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        """Invoke a tool on a target tool server."""

    @property
    def tool_servers(self) -> dict[str, str]:
        """Deprecated mapping of namespace name to endpoint string.

        Deprecated: use ``list_tool_namespaces()`` instead. Will be removed in 0.4.0.
        """
        warnings.warn(
            "BaseEnvironment.tool_servers is deprecated; use list_tool_namespaces() instead. "
            "This compatibility property will be removed in 0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            namespace.name: namespace.endpoint or "" for namespace in self.list_tool_namespaces()
        }


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


def _run_async_compat(coro: Coroutine[object, object, T]) -> T:
    """Run an async environment compatibility wrapper from sync code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Future[T] = Future()

    def _target() -> None:
        try:
            result.set_result(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - exercised through wrapper behavior
            result.set_exception(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    return result.result()
