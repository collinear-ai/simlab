"""LangChain/LangGraph tool adapters for SimLab environments."""

from __future__ import annotations

from itertools import count
from typing import Any
from typing import cast

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

from simlab.agents.adapters.artifacts import ToolEventRecorder
from simlab.agents.adapters.core import ToolDescriptor
from simlab.agents.adapters.core import _run_async_compat
from simlab.agents.adapters.core import alist_tool_descriptors
from simlab.agents.adapters.core import build_tool_dispatch
from simlab.agents.adapters.core import stringify_observation
from simlab.agents.base import BaseEnvironment
from simlab.agents.base import ToolCallResult

_structured_tool_cls: Any
try:
    import langchain_core.tools as _langchain_tools
except ModuleNotFoundError:  # pragma: no cover - exercised when optional deps are absent
    _structured_tool_cls = None
else:  # pragma: no branch - simple optional dependency wiring
    _structured_tool_cls = _langchain_tools.StructuredTool


def _schema_to_model(descriptor: ToolDescriptor) -> type[BaseModel]:
    raw_schema = descriptor.input_schema
    properties = raw_schema.get("properties", {})
    required = set(raw_schema.get("required", []))
    fields: dict[str, tuple[type[Any], Any]] = {}
    for name, raw in properties.items():
        if not isinstance(raw, dict):
            continue
        default = ... if name in required else None
        fields[name] = (
            Any,
            Field(default=default, description=raw.get("description", "")),
        )
    return create_model(
        f"{descriptor.tool_server}_{descriptor.tool_name}_Args",
        **cast("dict[str, Any]", fields),
    )


async def abuild_langchain_tools(
    environment: BaseEnvironment,
    *,
    recorder: ToolEventRecorder | None = None,
) -> list[Any]:
    """Build LangChain ``BaseTool`` wrappers around the SimLab environment."""
    if _structured_tool_cls is None:
        msg = "langchain-core is required to build LangChain tools"
        raise ModuleNotFoundError(msg)

    tools: list[Any] = []
    call_counter = count(1)
    descriptors = await alist_tool_descriptors(environment)
    build_tool_dispatch(descriptors)
    for descriptor in descriptors:
        args_model = _schema_to_model(descriptor)

        async def ainvoke_tool(
            _tool_server: str = descriptor.tool_server,
            _tool_name: str = descriptor.tool_name,
            **kwargs: object,
        ) -> str:
            payload = cast("dict[str, Any]", dict(kwargs))
            if recorder is not None:
                tool_call_id = f"call_{next(call_counter)}"
                recorder.on_assistant_tool_call(
                    tool_call_id,
                    _tool_server,
                    _tool_name,
                    payload,
                )
            result = await environment.acall_tool(_tool_server, _tool_name, payload)
            if recorder is not None:
                recorder.on_tool_invocation(
                    tool_call_id,
                    _tool_server,
                    _tool_name,
                    payload,
                    ToolCallResult(
                        observation=result.observation,
                        is_error=result.is_error,
                    ),
                )
            return stringify_observation(result.observation)

        def invoke_tool(
            _tool_server: str = descriptor.tool_server,
            _tool_name: str = descriptor.tool_name,
            **kwargs: object,
        ) -> str:
            return _run_async_compat(
                ainvoke_tool(
                    _tool_server=_tool_server,
                    _tool_name=_tool_name,
                    **kwargs,
                )
            )

        tools.append(
            _structured_tool_cls.from_function(
                func=invoke_tool,
                coroutine=ainvoke_tool,
                name=descriptor.wire_name,
                description=descriptor.description,
                args_schema=args_model,
            )
        )
    return tools


def build_langchain_tools(
    environment: BaseEnvironment,
    *,
    recorder: ToolEventRecorder | None = None,
) -> list[Any]:
    """Build LangChain ``BaseTool`` wrappers from sync code."""
    return _run_async_compat(abuild_langchain_tools(environment, recorder=recorder))
