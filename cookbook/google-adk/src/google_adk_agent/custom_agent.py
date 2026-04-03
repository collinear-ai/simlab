"""Custom Google ADK agent code used by this cookbook."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.runners import RunConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash"
SIMLAB_SENTINEL_MODEL_VALUES = {"custom-agent"}
DEFAULT_MAX_LLM_CALLS = 30
DEFAULT_INSTRUCTIONS = (
    "Use the available tools to complete the user's request end-to-end. "
    "Base factual claims on tool output. "
    "Keep the final response concise."
)


def resolve_model(model: str | None = None) -> str:
    """Resolve the model for the custom ADK agent."""
    candidate = (model or "").strip()
    if candidate in SIMLAB_SENTINEL_MODEL_VALUES:
        candidate = ""
    return candidate or os.getenv("GOOGLE_ADK_MODEL", "").strip() or DEFAULT_MODEL


def resolve_instructions(instructions: str | None = None) -> str:
    """Resolve the cookbook owned default instructions."""
    return (
        (instructions or "").strip()
        or os.getenv("GOOGLE_ADK_INSTRUCTIONS", "").strip()
        or DEFAULT_INSTRUCTIONS
    )


def resolve_user_id(user_id: str | None = None) -> str:
    """Resolve the user id for the ADK session."""
    return (
        (user_id or "").strip()
        or os.getenv("GOOGLE_ADK_USER_ID", "").strip()
        or "simlab"
    )


def resolve_session_id(session_id: str | None = None) -> str:
    """Resolve the session id for the ADK session."""
    return (
        (session_id or "").strip()
        or os.getenv("GOOGLE_ADK_SESSION_ID", "").strip()
        or "simlab-session"
    )


def resolve_max_llm_calls(max_llm_calls: int | None = None) -> int:
    """Resolve the max LLM call budget for one ADK run."""
    if max_llm_calls is not None:
        return max_llm_calls
    raw_value = os.getenv("GOOGLE_ADK_MAX_LLM_CALLS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_LLM_CALLS
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_MAX_LLM_CALLS


def build_custom_agent(
    *,
    tools: list[Callable[..., Any] | BaseTool | BaseToolset],
    model: str | None = None,
    instructions: str | None = None,
) -> LlmAgent:
    """Build the custom Google ADK agent."""
    return LlmAgent(
        name="simlab_google_adk_agent",
        model=resolve_model(model),
        instruction=resolve_instructions(instructions),
        tools=tools,
    )


def extract_text_from_event(event: object) -> str:
    """Extract the visible text from an ADK event payload."""
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None)
    if not parts:
        return ""
    chunks: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            chunks.append(text)
    return "".join(chunks)


def is_final_response_event(event: object) -> bool:
    """Return True when the ADK event represents the final response."""
    attribute = getattr(event, "is_final_response", None)
    if callable(attribute):
        try:
            return bool(attribute())
        except Exception:
            return False
    return attribute is True


def run_custom_agent(
    *,
    instruction: str,
    tools: list[Callable[..., Any] | BaseTool | BaseToolset],
    model: str | None = None,
    instructions: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    max_llm_calls: int | None = None,
) -> str:
    """Run the custom Google ADK agent and return the final text response."""
    agent = build_custom_agent(
        tools=tools,
        model=model,
        instructions=instructions,
    )
    runner = Runner(
        app_name="simlab-google-adk",
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )
    message = types.Content(
        role="user",
        parts=[types.Part(text=instruction)],
    )
    run_config = RunConfig(max_llm_calls=resolve_max_llm_calls(max_llm_calls))
    final_text = ""
    for event in runner.run(
        user_id=resolve_user_id(user_id),
        session_id=resolve_session_id(session_id),
        new_message=message,
        run_config=run_config,
    ):
        candidate = extract_text_from_event(event).strip()
        if candidate:
            final_text = candidate
        if is_final_response_event(event):
            break
    return final_text
