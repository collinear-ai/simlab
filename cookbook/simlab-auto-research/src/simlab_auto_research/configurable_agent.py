"""Configurable agent that reads its system prompt from a file at runtime.

Modeled on SimLab's built-in ReferenceAgent. The only difference is that the
system prompt is loaded from a file (default: ``system-prompt.md``) at the
start of every ``run()`` call instead of being hardcoded.  This lets an outer
agent modify the prompt between experiments.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from simlab.agents.adapters import alist_tool_descriptors
from simlab.agents.adapters import build_artifact_tool_message_content
from simlab.agents.adapters import build_tool_dispatch
from simlab.agents.base import BaseAgent
from simlab.agents.base import BaseEnvironment
from simlab.agents.base import RunArtifacts
from simlab.agents.base import ToolCall
from simlab.agents.base import ToolCallResult
from simlab.agents.base import _run_async_compat

DEFAULT_SYSTEM_PROMPT_PATH = "system-prompt.md"


class ConfigurableAgent(BaseAgent):
    """LiteLLM tool-calling agent whose system prompt is loaded from a file.

    Reads ``system-prompt.md`` (or the path in ``SYSTEM_PROMPT_PATH``) at the
    start of every ``run()`` call.  Everything else mirrors ReferenceAgent.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize from env vars (no-arg construction required by SimLab loader)."""
        self._provider = provider or os.getenv("SIMLAB_AGENT_PROVIDER", "openai")
        self._api_key = (
            api_key or os.getenv("SIMLAB_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self._base_url = base_url or os.getenv("SIMLAB_AGENT_BASE_URL")
        self._model = model or os.getenv("SIMLAB_AGENT_MODEL", "gpt-4o-mini")

    @staticmethod
    def name() -> str:
        """Return the agent name."""
        return "configurable-agent"

    def setup(self, environment: BaseEnvironment) -> None:
        """No-op setup; environment is used during run()."""
        _ = environment

    def _load_system_prompt(self) -> str:
        """Read the system prompt from the configured file path."""
        path = Path(os.getenv("SYSTEM_PROMPT_PATH", DEFAULT_SYSTEM_PROMPT_PATH))
        if not path.exists():
            raise FileNotFoundError(
                f"System prompt file not found: {path}. "
                f"Create it or set SYSTEM_PROMPT_PATH to the correct location."
            )
        return path.read_text(encoding="utf-8").strip()

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Execute the instruction using the file-based system prompt."""
        import litellm  # noqa: PLC0415

        if not self._api_key:
            raise ValueError(
                "ConfigurableAgent requires an API key (set SIMLAB_AGENT_API_KEY or OPENAI_API_KEY)"
            )

        system_prompt = self._load_system_prompt()

        raw_model = self._model
        litellm_model = raw_model
        if self._provider and not raw_model.startswith(f"{self._provider}/"):
            litellm_model = f"{self._provider}/{raw_model}"

        openai_tools, dispatch = _run_async_compat(_abuild_openai_tools(environment))
        context.metadata["configurable_agent"] = {
            "system_prompt_path": os.getenv(
                "SYSTEM_PROMPT_PATH", DEFAULT_SYSTEM_PROMPT_PATH
            ),
            "system_prompt_length": len(system_prompt),
            "tool_count": len(openai_tools),
            "tool_names": [t["function"]["name"] for t in openai_tools],
        }

        messages: list[dict[str, Any]] = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instruction})
        context.record_message("user", instruction)
        context.metadata["litellm_model"] = litellm_model

        max_steps = context.max_steps or 30
        for _ in range(max_steps):
            if stop_event is not None and stop_event.is_set():
                context.error = "Cancelled"
                return

            response = litellm.completion(
                model=litellm_model,
                messages=messages,
                tools=openai_tools or None,
                api_key=self._api_key,
                base_url=self._base_url,
            )
            message = response.choices[0].message  # type: ignore[union-attr]
            tool_calls = message.tool_calls or []

            if tool_calls:
                message_tool_calls = []
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
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
                        "tool_calls": message_tool_calls,
                    },
                )
                for tool_call in tool_calls:
                    fn = getattr(tool_call, "function", None)
                    if fn is None:
                        continue
                    result = _run_async_compat(
                        _aexecute_tool_call(
                            tool_call_id=tool_call.id,
                            tool_name=fn.name,
                            raw_args=fn.arguments,
                            dispatch=dispatch,
                            environment=environment,
                            context=context,
                        )
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
                        build_artifact_tool_message_content(
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


# ---------------------------------------------------------------------------
# Private helpers (mirrored from simlab.agents.reference_agent)
# ---------------------------------------------------------------------------


async def _abuild_openai_tools(
    environment: BaseEnvironment,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    descriptors = await alist_tool_descriptors(environment)
    dispatch = build_tool_dispatch(descriptors)
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": descriptor.wire_name,
                "description": descriptor.description,
                "parameters": descriptor.input_schema,
            },
        }
        for descriptor in descriptors
    ]
    return openai_tools, dispatch


async def _aexecute_tool_call(
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
    result = await environment.acall_tool(server, actual_name, params)
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
