"""Google ADK tool adapters for SimLab environments."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import cast

from simlab.agents.adapters.artifacts import ToolEventRecorder
from simlab.agents.adapters.core import ToolDescriptor
from simlab.agents.adapters.core import build_tool_dispatch
from simlab.agents.adapters.core import list_tool_descriptors
from simlab.agents.adapters.core import stringify_observation
from simlab.agents.base import BaseEnvironment
from simlab.agents.base import ToolCallResult

adk_base_tool_class: Any
genai_function_declaration_class: Any
try:
    from google.adk.tools.base_tool import BaseTool as AdkBaseTool
    from google.genai import types as genai_types
except (
    ImportError,
    ModuleNotFoundError,
):  # pragma: no cover - exercised when optional deps are absent
    adk_base_tool_class = None
    genai_function_declaration_class = None
else:  # pragma: no cover - exercised when optional deps are present
    adk_base_tool_class = AdkBaseTool
    genai_function_declaration_class = genai_types.FunctionDeclaration


@dataclass
class ToolRuntime:
    """Runtime state shared across ADK tool wrappers."""

    environment: BaseEnvironment
    recorder: ToolEventRecorder | None = None
    counter: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    def next_call_id(self) -> str:
        """Return a deterministic tool call id for artifact recording."""
        return f"call_{next(self.counter)}"


def create_simlab_google_adk_tool_class(
    base_tool_class: type[Any],
    function_declaration_class: type[Any],
) -> type[Any]:
    """Return an ADK BaseTool wrapper class for SimLab tool descriptors."""

    class SimLabGoogleAdkTool(base_tool_class):  # type: ignore[misc]
        """ADK tool wrapper that dispatches calls into a SimLab environment."""

        descriptor: ToolDescriptor
        runtime: ToolRuntime

        def __init__(
            self,
            descriptor: ToolDescriptor,
            runtime: ToolRuntime,
        ) -> None:
            super().__init__(
                name=descriptor.wire_name,
                description=descriptor.description,
            )
            self.descriptor = descriptor
            self.runtime = runtime

        def _get_declaration(self) -> object:
            """Return a Google GenAI function declaration for the tool."""
            return function_declaration_class(
                name=self.name,
                description=self.description,
                parameters_json_schema=self.descriptor.input_schema,
            )

        async def run_async(  # type: ignore[override]
            self,
            *,
            args: dict[str, Any],
            tool_context: object,
        ) -> dict[str, Any]:
            _ = tool_context
            parameters = args if isinstance(args, dict) else {}
            tool_call_id = self.runtime.next_call_id()
            if self.runtime.recorder is not None:
                self.runtime.recorder.on_assistant_tool_call(
                    tool_call_id,
                    self.descriptor.tool_server,
                    self.descriptor.tool_name,
                    parameters,
                )
            result = await self.runtime.environment.acall_tool(
                self.descriptor.tool_server,
                self.descriptor.tool_name,
                cast("dict[str, Any]", parameters),
            )
            if self.runtime.recorder is not None:
                self.runtime.recorder.on_tool_invocation(
                    tool_call_id,
                    self.descriptor.tool_server,
                    self.descriptor.tool_name,
                    parameters,
                    ToolCallResult(
                        observation=result.observation,
                        is_error=result.is_error,
                    ),
                )
            observation = result.observation
            if isinstance(observation, dict):
                return observation
            if isinstance(observation, (str, int, float, bool, list)) or observation is None:
                return {"result": observation}
            return {"result": stringify_observation(observation)}

    return SimLabGoogleAdkTool


def build_google_adk_tools(
    environment: BaseEnvironment,
    *,
    recorder: ToolEventRecorder | None = None,
) -> list[Any]:
    """Build Google ADK ``BaseTool`` wrappers around the SimLab environment."""
    if adk_base_tool_class is None or genai_function_declaration_class is None:
        msg = "google-adk is required to build Google ADK tools"
        raise ModuleNotFoundError(msg)

    runtime = ToolRuntime(environment=environment, recorder=recorder)
    descriptors = list_tool_descriptors(environment)
    build_tool_dispatch(descriptors)
    tool_class = create_simlab_google_adk_tool_class(
        adk_base_tool_class,
        genai_function_declaration_class,
    )
    return [tool_class(descriptor, runtime) for descriptor in descriptors]
