#!/usr/bin/env python3
r"""SimLab agent backed by a Tinker sampling client.

Compatible with ``simlab tasks run --agent-import-path``.  Configures itself
from environment variables since simlab instantiates agents with no arguments.

Environment variables:
    TINKER_API_KEY          - Required
    TINKER_BASE_MODEL       - e.g. "Qwen/Qwen3-4B-Instruct-2507"
    TINKER_RENDERER         - e.g. "qwen3_instruct"
    TINKER_CHECKPOINT       - Full tinker:// path to checkpoint (omit for base model)
    TINKER_MAX_TOKENS       - Max tokens per generation (default: 4096)

Usage:
    # Base model:
    export TINKER_BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
    export TINKER_RENDERER=qwen3_instruct
    simlab tasks run --env my-env --task <id> --tasks-dir ./eval-tasks \
        --agent-import-path cookbook.tinker-sft.scripts.eval_agent:TinkerEvalAgent

    # Fine-tuned checkpoint:
    export TINKER_CHECKPOINT="tinker://...:train:0/weights/my-checkpoint"
    simlab tasks run ...
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

import tinker
from simlab.agents.base import BaseAgent
from simlab.agents.base import BaseEnvironment
from simlab.agents.base import RunArtifacts
from simlab.agents.base import ToolCall
from simlab.agents.base import ToolCallResult
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.supervised.types import get_tokenizer

# Tolerant tool-call parser. Accepts:
#   <tool_call>{...}</tool_call>      (Qwen3 official format)
#   <function_call>{...}</function_call>  (tinker_cookbook format)
# And both "arguments" and "args" as the keyword.
_TOOL_CALL_RE = re.compile(
    r"<(tool_call|function_call)>\s*(\{.*?\})\s*</\1>",
    re.DOTALL,
)


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Return a list of {name, args} dicts parsed from assistant text."""
    out: list[dict[str, Any]] = []
    for _tag, body in _TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments", obj.get("args", {}))
        if not isinstance(name, str) or not isinstance(args, dict):
            continue
        out.append({"name": name, "args": args})
    return out


class TinkerEvalAgent(BaseAgent):
    """SimLab agent that samples from a Tinker model (base or fine-tuned)."""

    def __init__(self) -> None:
        """Initialise from environment variables."""
        base_model = os.environ.get("TINKER_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        renderer_name = os.environ.get("TINKER_RENDERER", "qwen3_instruct")
        checkpoint = os.environ.get("TINKER_CHECKPOINT", "")
        self._max_tokens = int(os.environ.get("TINKER_MAX_TOKENS", "4096"))

        tokenizer = get_tokenizer(base_model)
        self._renderer = get_renderer(renderer_name, tokenizer)
        self._tokenizer = tokenizer

        sc = tinker.ServiceClient()
        if checkpoint:
            # checkpoint is a sampler path returned by save_weights_for_sampler:
            # tinker://<run-id>:train:<step>/sampler_weights/<name>
            self._client = sc.create_sampling_client(model_path=checkpoint)
        else:
            self._client = sc.create_sampling_client(base_model=base_model)

    @staticmethod
    def name() -> str:
        """Return agent name."""
        return "tinker-eval-agent"

    def setup(self, environment: BaseEnvironment) -> None:
        """No-op; config is handled in __init__."""

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Execute the agent loop: sample, parse tool calls, dispatch."""
        # Build tool specs from environment
        tool_specs = []
        dispatch = {}
        for tool in environment.list_tools():
            server = tool.get("tool_server")
            name = tool.get("name")
            if not server or not name:
                continue
            wire_name = f"{server}__{name}"
            dispatch[wire_name] = (server, name)
            tool_specs.append(
                {
                    "name": wire_name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                }
            )

        # Build system prompt: instructions + tool listing + call format.
        # Training data uses <tool_call>{"name":..., "arguments":...}</tool_call>.
        tools_listing = "\n".join(
            f"- {t['name']}: {t['description']}\n  parameters: {json.dumps(t['parameters'])}"
            for t in tool_specs
        )
        system_prompt = (
            "You have direct tool access via function calls. "
            "Do not ask the user for system access, links, or credentials. "
            "Use available tools to complete the task end-to-end.\n\n"
            "To call a tool, emit one or more blocks of exactly this form:\n"
            '<tool_call>\n{"name": "<tool_name>", "arguments": {<json args>}}\n</tool_call>\n\n'
            "Available tools:\n" + tools_listing
        )

        stop_sequences = self._renderer.get_stop_sequences()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        context.record_message("user", instruction)

        max_steps = context.max_steps or 30

        for step in range(max_steps):
            if stop_event and stop_event.is_set():
                context.error = "Cancelled"
                return

            prompt = self._renderer.build_generation_prompt(messages)
            try:
                result = self._client.sample(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        stop=stop_sequences,
                        max_tokens=self._max_tokens,
                        temperature=0.0,
                    ),
                ).result()
            except Exception as exc:
                if "exceeds the model's context window" in str(exc):
                    context.error = "context_window_exceeded"
                    return
                raise

            response_tokens = result.sequences[0].tokens
            # Decode tokens ourselves so we don't depend on tinker_cookbook's
            # hard-coded <function_call> regex. Strip the <|im_end|> stop token if present.
            text = self._tokenizer.decode(response_tokens)
            text = text.split("<|im_end|>", 1)[0]
            tool_calls = _extract_tool_calls(text)
            message = {"role": "assistant", "content": text}

            if tool_calls:
                messages.append(message)
                serialized = []
                for i, tc in enumerate(tool_calls):
                    fn_name = tc.get("name", "")
                    args = tc.get("args", {}) or {}
                    serialized.append(
                        {
                            "id": f"call_{step}_{i}",
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": json.dumps(args),
                            },
                        }
                    )
                context.record_message(
                    "assistant",
                    {
                        "content": message.get("content", ""),
                        "tool_calls": serialized,
                    },
                )

                for i, tc in enumerate(tool_calls):
                    fn_name = tc.get("name", "")
                    args = tc.get("args", {}) or {}

                    if fn_name in dispatch:
                        server, actual_name = dispatch[fn_name]
                        tool_result = environment.call_tool(server, actual_name, args)
                    else:
                        server, actual_name = "unknown", fn_name
                        tool_result = ToolCallResult(
                            observation=f"Unknown tool: {fn_name}", is_error=True
                        )

                    context.record_tool_call(
                        ToolCall(tool_server=server, tool_name=actual_name, parameters=args),
                        tool_result,
                    )

                    obs = tool_result.observation
                    content_str = json.dumps(obs) if not isinstance(obs, str) else obs
                    messages.append(
                        {
                            "role": "tool",
                            "content": content_str,
                            "tool_call_id": f"call_{step}_{i}",
                            "name": fn_name,
                        }
                    )
                continue

            # No tool calls — final response
            text = message.get("content", "")
            context.record_message("assistant", text)
            context.final_observation = text
            return

        context.error = "Max steps reached"
