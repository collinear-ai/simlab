"""OpenAI Agents SDK tool adapters for SimLab environments."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from simlab.agents.adapters.artifacts import ToolEventRecorder
from simlab.agents.adapters.core import ToolDescriptor
from simlab.agents.adapters.core import build_tool_dispatch
from simlab.agents.adapters.core import list_tool_descriptors
from simlab.agents.adapters.core import normalize_openai_tool_schema
from simlab.agents.adapters.core import stringify_observation
from simlab.agents.base import BaseEnvironment
from simlab.agents.base import ToolCallResult

_function_tool_class: Any
try:
    from agents import FunctionTool
except (
    ImportError,
    ModuleNotFoundError,
):  # pragma: no cover - exercised when optional deps are absent
    _function_tool_class = None
else:  # pragma: no cover - exercised when optional deps are present
    _function_tool_class = FunctionTool


@dataclass
class _ToolRuntime:
    """Runtime state shared across SDK tool wrappers."""

    environment: BaseEnvironment
    recorder: ToolEventRecorder | None = None
    counter: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    def next_call_id(self) -> str:
        """Return a deterministic tool call id for artifact recording."""
        return f"call_{next(self.counter)}"


def _parse_parameters(raw_args: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _build_openai_agents_tool(
    descriptor: ToolDescriptor,
    runtime: _ToolRuntime,
) -> object:
    async def _invoke(_ctx: object, args: str) -> str:
        parameters = _parse_parameters(args)
        tool_call_id = runtime.next_call_id()
        if runtime.recorder is not None:
            runtime.recorder.on_assistant_tool_call(
                tool_call_id,
                descriptor.tool_server,
                descriptor.tool_name,
                parameters,
            )
        result = await runtime.environment.acall_tool(
            descriptor.tool_server,
            descriptor.tool_name,
            parameters,
        )
        if runtime.recorder is not None:
            runtime.recorder.on_tool_invocation(
                tool_call_id,
                descriptor.tool_server,
                descriptor.tool_name,
                parameters,
                ToolCallResult(
                    observation=result.observation,
                    is_error=result.is_error,
                ),
            )
        return stringify_observation(result.observation)

    return _function_tool_class(
        name=descriptor.wire_name,
        description=descriptor.description,
        params_json_schema=normalize_openai_tool_schema(descriptor.input_schema),
        on_invoke_tool=_invoke,
        # These schemas come from external SimLab/MCP tool definitions rather than the
        # SDK's own Pydantic function-schema path. The SDK strict rewriter can change
        # required/optional semantics for pass-through schemas, so we preserve the
        # upstream contract here and only apply our own targeted normalization above.
        strict_json_schema=False,
    )


def build_openai_agents_tools(
    environment: BaseEnvironment,
    *,
    recorder: ToolEventRecorder | None = None,
) -> list[Any]:
    """Build OpenAI Agents SDK ``FunctionTool`` wrappers around the SimLab environment."""
    if _function_tool_class is None:
        msg = "openai-agents is required to build OpenAI Agents SDK tools"
        raise ModuleNotFoundError(msg)

    runtime = _ToolRuntime(environment=environment, recorder=recorder)
    descriptors = list_tool_descriptors(environment)
    build_tool_dispatch(descriptors)
    return [_build_openai_agents_tool(descriptor, runtime) for descriptor in descriptors]
