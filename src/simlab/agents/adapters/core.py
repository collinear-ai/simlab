"""Framework-neutral tool descriptors and dispatch helpers."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any
from typing import TypeVar

from simlab.agents.base import BaseEnvironment

T = TypeVar("T")


@dataclass(frozen=True)
class ToolDescriptor:
    """Normalized description of one tool exposed by a ``BaseEnvironment``."""

    tool_server: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    transport: str | None = None

    @property
    def wire_name(self) -> str:
        """Return the stable framework-facing tool name."""
        return f"{self.tool_server}__{self.tool_name}"


def stringify_observation(observation: object) -> str:
    """Return a string form suitable for framework tool outputs."""
    if isinstance(observation, str):
        return observation
    try:
        return json.dumps(observation, indent=2, sort_keys=True, default=str)
    except TypeError:
        return str(observation)


def normalize_openai_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a tool schema normalized for OpenAI-compatible tools."""
    normalized = _normalize_openai_schema_node(schema, schema_node=True)
    if not isinstance(normalized, dict):
        return {"type": "object"}
    if "type" not in normalized:
        normalized["type"] = "object"
    return normalized


async def alist_tool_descriptors(environment: BaseEnvironment) -> list[ToolDescriptor]:
    """Return normalized tool descriptors for all visible environment tools."""
    descriptors: list[ToolDescriptor] = []
    for tool in await environment.alist_tools():
        tool_server = tool.get("tool_server")
        tool_name = tool.get("name")
        if not tool_server or not tool_name:
            continue
        raw_schema = tool.get("input_schema")
        input_schema = raw_schema if isinstance(raw_schema, dict) else {"type": "object"}
        descriptors.append(
            ToolDescriptor(
                tool_server=str(tool_server),
                tool_name=str(tool_name),
                description=str(tool.get("description", "")),
                input_schema=input_schema,
                transport=(
                    str(tool["transport"]) if isinstance(tool.get("transport"), str) else None
                ),
            )
        )
    return descriptors


def list_tool_descriptors(environment: BaseEnvironment) -> list[ToolDescriptor]:
    """Return normalized tool descriptors from sync code without using deprecated APIs."""
    return _run_async_compat(alist_tool_descriptors(environment))


def build_tool_dispatch(
    descriptors: list[ToolDescriptor],
) -> dict[str, tuple[str, str]]:
    """Build a wire-name dispatch map, rejecting ambiguous duplicates."""
    dispatch: dict[str, tuple[str, str]] = {}
    for descriptor in descriptors:
        if descriptor.wire_name in dispatch:
            existing_server, existing_tool = dispatch[descriptor.wire_name]
            raise RuntimeError(
                "Duplicate tool wire name detected: "
                f"{descriptor.wire_name} is provided by both "
                f"{existing_server}.{existing_tool} and "
                f"{descriptor.tool_server}.{descriptor.tool_name}"
            )
        dispatch[descriptor.wire_name] = (
            descriptor.tool_server,
            descriptor.tool_name,
        )
    return dispatch


_SCHEMA_MAPPING_KEYWORDS = {
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
}
_SCHEMA_LIST_KEYWORDS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_VALUE_KEYWORDS = {
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedProperties",
}


def _clone_json_value(node: object) -> object:
    """Return a deep copy of plain JSON-like values without schema rewriting."""
    if isinstance(node, list):
        return [_clone_json_value(item) for item in node]
    if isinstance(node, dict):
        return {key: _clone_json_value(value) for key, value in node.items()}
    return node


def _normalize_openai_schema_node(node: object, *, schema_node: bool) -> object:
    """Recursively normalize nested JSON Schema nodes for OpenAI compatibility."""
    if isinstance(node, list):
        if not schema_node:
            return [_clone_json_value(item) for item in node]
        return [_normalize_openai_schema_node(item, schema_node=True) for item in node]
    if not isinstance(node, dict):
        return node
    if schema_node and not node:
        return {"type": "object"}
    if not schema_node:
        return _clone_json_value(node)

    normalized: dict[str, object] = {}
    for key, value in node.items():
        if key in _SCHEMA_MAPPING_KEYWORDS and isinstance(value, dict):
            normalized[key] = {
                child_key: _normalize_openai_schema_node(child_value, schema_node=True)
                for child_key, child_value in value.items()
            }
            continue
        if key in _SCHEMA_LIST_KEYWORDS and isinstance(value, list):
            normalized[key] = [
                _normalize_openai_schema_node(item, schema_node=True) for item in value
            ]
            continue
        if key in _SCHEMA_VALUE_KEYWORDS and isinstance(value, dict):
            normalized[key] = _normalize_openai_schema_node(value, schema_node=True)
            continue
        normalized[key] = _clone_json_value(value)
    if "type" not in normalized:
        if "properties" in normalized:
            normalized["type"] = "object"
        elif "items" in normalized:
            normalized["type"] = "array"
    return normalized


def _run_async_compat(coro: Coroutine[object, object, T]) -> T:
    """Run an async helper from sync adapter code."""
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
