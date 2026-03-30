"""Custom OpenAI Agents SDK app code used by this cookbook."""

from __future__ import annotations

import json
import os
from typing import Any

from agents import Agent
from agents import Runner
from agents.result import RunResult
from agents.tool import Tool

DEFAULT_MODEL = "gpt-4o-mini"
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
    return (
        candidate or os.getenv("OPENAI_AGENTS_SDK_MODEL", "").strip() or DEFAULT_MODEL
    )


def resolve_instructions(instructions: str | None = None) -> str:
    """Resolve the cookbook-owned default instructions."""
    return (
        (instructions or "").strip()
        or os.getenv("OPENAI_AGENTS_SDK_INSTRUCTIONS", "").strip()
        or DEFAULT_INSTRUCTIONS
    )


def resolve_max_turns(max_turns: int | None = None) -> int:
    """Resolve the max turn budget for one SDK run."""
    if max_turns is not None:
        return max_turns
    raw_value = os.getenv("OPENAI_AGENTS_SDK_MAX_TURNS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_TURNS
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_MAX_TURNS


def build_custom_agent(
    *,
    tools: list[Tool],
    model: str | None = None,
    instructions: str | None = None,
) -> Agent[Any]:
    """Build the custom OpenAI Agents SDK agent."""
    return Agent(
        name="SimLab Custom Agent",
        model=resolve_model(model),
        instructions=resolve_instructions(instructions),
        tools=tools,
    )


def stringify_final_output(output: object) -> str:
    """Return a stable string representation for SimLab final output."""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, indent=2, sort_keys=True, default=str)
    except TypeError:
        return str(output)


def run_custom_agent(
    *,
    instruction: str,
    tools: list[Tool],
    model: str | None = None,
    instructions: str | None = None,
    max_turns: int | None = None,
) -> RunResult:
    """Run the custom OpenAI Agents SDK app."""
    agent = build_custom_agent(
        tools=tools,
        model=model,
        instructions=instructions,
    )
    return Runner.run_sync(
        agent,
        instruction,
        max_turns=resolve_max_turns(max_turns),
    )
