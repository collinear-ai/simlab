"""Shared adapter primitives for integrating external agent frameworks with SimLab."""

from simlab.agents.adapters.artifacts import RunArtifactsRecorder
from simlab.agents.adapters.artifacts import ToolEventRecorder
from simlab.agents.adapters.artifacts import build_artifact_assistant_tool_call_content
from simlab.agents.adapters.artifacts import build_artifact_tool_message_content
from simlab.agents.adapters.claude_agent import build_claude_agent_tools
from simlab.agents.adapters.core import ToolDescriptor
from simlab.agents.adapters.core import alist_tool_descriptors
from simlab.agents.adapters.core import build_tool_dispatch
from simlab.agents.adapters.core import list_tool_descriptors
from simlab.agents.adapters.core import normalize_openai_tool_schema
from simlab.agents.adapters.core import stringify_observation
from simlab.agents.adapters.google_adk import build_google_adk_tools
from simlab.agents.adapters.openai_agents import build_openai_agents_tools

__all__ = [
    "RunArtifactsRecorder",
    "ToolDescriptor",
    "ToolEventRecorder",
    "alist_tool_descriptors",
    "build_artifact_assistant_tool_call_content",
    "build_artifact_tool_message_content",
    "build_claude_agent_tools",
    "build_google_adk_tools",
    "build_openai_agents_tools",
    "build_tool_dispatch",
    "list_tool_descriptors",
    "normalize_openai_tool_schema",
    "stringify_observation",
]
