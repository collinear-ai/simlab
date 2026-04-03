"""Custom Claude Agent SDK app code used by this cookbook."""

from __future__ import annotations

import json
import os
from typing import Any

from claude_agent_sdk import AssistantMessage
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import TextBlock
from claude_agent_sdk import query

DEFAULT_MODEL = "claude-sonnet-4-20250514"
SIMLAB_SENTINEL_MODEL_VALUES = {"custom-agent"}
DEFAULT_INSTRUCTIONS = (
    "Use the available tools to complete the user's request end-to-end. "
    "Base factual claims on tool output and keep the final response concise."
)
DEFAULT_MAX_TURNS = 24


def resolve_model(model: str | None = None) -> str:
    """Resolve the model for the custom agent."""
    candidate = (model or "").strip()
    if candidate in SIMLAB_SENTINEL_MODEL_VALUES:
        candidate = ""
    return candidate or os.getenv("CLAUDE_AGENT_SDK_MODEL", "").strip() or DEFAULT_MODEL


def resolve_instructions(instructions: str | None = None) -> str:
    """Resolve the cookbook-owned default instructions."""
    return (
        (instructions or "").strip()
        or os.getenv("CLAUDE_AGENT_SDK_INSTRUCTIONS", "").strip()
        or DEFAULT_INSTRUCTIONS
    )


def resolve_max_turns(max_turns: int | None = None) -> int:
    """Resolve the max turn budget for one SDK run."""
    if max_turns is not None:
        return max_turns
    raw_value = os.getenv("CLAUDE_AGENT_SDK_MAX_TURNS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_TURNS
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_MAX_TURNS


def build_custom_agent(
    *,
    mcp_servers: dict[str, Any],
    allowed_tools: list[str],
    model: str | None = None,
    instructions: str | None = None,
    max_turns: int | None = None,
) -> ClaudeAgentOptions:
    """Build ClaudeAgentOptions for a SimLab-integrated agent."""
    return ClaudeAgentOptions(
        model=resolve_model(model),
        system_prompt=resolve_instructions(instructions),
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
        permission_mode="bypassPermissions",
        max_turns=resolve_max_turns(max_turns),
    )


def stringify_final_output(output: object) -> str:
    """Return a stable string representation for SimLab final output."""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, indent=2, sort_keys=True, default=str)
    except TypeError:
        return str(output)


async def run_custom_agent(
    *,
    instruction: str,
    mcp_servers: dict[str, Any],
    allowed_tools: list[str],
    model: str | None = None,
    instructions: str | None = None,
    max_turns: int | None = None,
) -> str:
    """Run the custom Claude Agent SDK app and return the final output text."""
    options = build_custom_agent(
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
        model=model,
        instructions=instructions,
        max_turns=max_turns,
    )
    final_text = ""
    async for message in query(prompt=instruction, options=options):
        # ResultMessage is the terminal message — always handle it, even
        # when result is None (structured_output or errors may be set).
        if isinstance(message, ResultMessage):
            if message.is_error:
                error_detail = (
                    message.result
                    or getattr(message, "errors", None)
                    or "unknown error"
                )
                raise RuntimeError(
                    f"Claude Agent SDK run failed: {stringify_final_output(error_detail)}"
                )
            if message.result is not None:
                final_text = stringify_final_output(message.result)
            elif getattr(message, "structured_output", None) is not None:
                final_text = stringify_final_output(message.structured_output)
            # else: keep prior assistant text as final_text
            break

        # Accumulate assistant text from AssistantMessage content blocks.
        if isinstance(message, AssistantMessage):
            text_parts = [
                block.text
                for block in (message.content or [])
                if isinstance(block, TextBlock) and block.text
            ]
            if text_parts:
                final_text = "\n".join(text_parts)

    return final_text
