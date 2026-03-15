"""Baked-in reference agent implementation."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from simlab.agents.base import BaseAgent
from simlab.agents.base import BaseEnvironment
from simlab.agents.base import RunArtifacts
from simlab.agents.base import ToolCall
from simlab.agents.base import ToolCallResult


class ReferenceAgent(BaseAgent):
    """OpenAI SDK tool-calling agent used when --agent-import-path is omitted."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialize the reference agent."""
        self._api_key = api_key
        self._base_url = base_url

    @staticmethod
    def name() -> str:
        """Return the agent name used when no custom agent is specified."""
        return "reference-agent"

    def setup(self, environment: BaseEnvironment) -> None:
        """No-op setup; environment is used during run()."""
        _ = environment

    def run(self, instruction: str, environment: BaseEnvironment, context: RunArtifacts) -> None:
        """Execute the instruction via OpenAI tool-calling and record results in context."""
        if not self._api_key:
            raise ValueError("ReferenceAgent requires an OpenAI API key")

        model = context.model or "gpt-4o-mini"
        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        openai_tools, dispatch = _build_openai_tools(environment)
        context.metadata["reference_agent_tools"] = {
            "count": len(openai_tools),
            "names": [t["function"]["name"] for t in openai_tools],
        }
        messages: list[dict[str, Any]] = []
        context.record_message("user", instruction)
        messages.append(
            {
                "role": "system",
                "content": (
                    "You have direct tool access via function calls. "
                    "Do not ask the user for system access, links, or credentials. "
                    "Use available tools to complete the task end-to-end."
                ),
            }
        )
        messages.append({"role": "user", "content": instruction})
        context.metadata["reference_agent"] = True

        max_steps = context.max_steps or 30
        for _ in range(max_steps):
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                tools=openai_tools or None,  # type: ignore[arg-type]
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if tool_calls:
                # Only function tool calls have .function; narrow for type checker
                serialized_tool_calls = []
                message_tool_calls = []
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
                    serialized_tool_calls.append(
                        {
                            "id": tc.id,
                            "name": fn.name,
                            "arguments": _parse_tool_arguments(fn.arguments),
                        }
                    )
                    message_tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": fn.name, "arguments": fn.arguments},
                        }
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": message_tool_calls,
                    }
                )
                context.record_message(
                    "assistant",
                    {
                        "content": message.content or "",
                        "tool_calls": serialized_tool_calls,
                    },
                )
                for tool_call in tool_calls:
                    fn = getattr(tool_call, "function", None)
                    if fn is None:
                        continue
                    result = _execute_tool_call(
                        tool_call_id=tool_call.id,
                        tool_name=fn.name,
                        raw_args=fn.arguments,
                        dispatch=dispatch,
                        environment=environment,
                        context=context,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": _to_tool_content(result),
                        }
                    )
                    context.record_message(
                        "tool",
                        _artifact_tool_message_content(
                            tool_call_id=tool_call.id,
                            tool_name=fn.name,
                            result=result,
                        ),
                    )
                continue

            assistant_text = message.content or ""
            context.record_message("assistant", assistant_text)
            context.final_observation = assistant_text
            return

        context.error = "Max steps reached without final assistant response"


def _build_openai_tools(
    environment: BaseEnvironment,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    openai_tools: list[dict[str, Any]] = []
    dispatch: dict[str, tuple[str, str]] = {}
    for tool in environment.list_tools():
        server = tool.get("tool_server")
        name = tool.get("name")
        if not server or not name:
            continue
        wire_name = f"{server}__{name}"
        dispatch[wire_name] = (server, name)
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": wire_name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
        )
    return openai_tools, dispatch


def _execute_tool_call(
    *,
    tool_call_id: str,
    tool_name: str,
    raw_args: str,
    dispatch: dict[str, tuple[str, str]],
    environment: BaseEnvironment,
    context: RunArtifacts,
) -> ToolCallResult:
    if tool_name not in dispatch:
        return ToolCallResult(
            observation=f"Unknown tool call requested by model: {tool_name}",
            is_error=True,
        )
    server, actual_name = dispatch[tool_name]
    try:
        params = json.loads(raw_args) if raw_args else {}
        if not isinstance(params, dict):
            params = {}
    except json.JSONDecodeError:
        params = {}
    call = ToolCall(tool_server=server, tool_name=actual_name, parameters=params)
    result = environment.call_tool(server, actual_name, params)
    context.record_tool_call(call, result)
    _ = tool_call_id
    return result


def _to_tool_content(payload: ToolCallResult) -> str:
    if isinstance(payload.observation, str):
        return payload.observation
    try:
        return json.dumps(payload.observation)
    except TypeError:
        return str(payload.observation)


def _parse_tool_arguments(raw_args: str) -> dict[str, Any] | str:
    try:
        parsed = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return raw_args
    return parsed if isinstance(parsed, dict) else raw_args


def _artifact_tool_message_content(
    *,
    tool_call_id: str,
    tool_name: str,
    result: ToolCallResult,
) -> dict[str, Any]:
    """Compact tool message for artifacts; full payload lives in tool_results."""
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
