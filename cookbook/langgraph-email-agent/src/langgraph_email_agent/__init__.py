"""Standalone LangGraph email assistant and SimLab adapter."""

from langgraph_email_agent.email_assistant import LangGraphEmailAssistant
from langgraph_email_agent.model_factory import build_chat_model_from_env
from langgraph_email_agent.simlab_adapter import SimLabLangGraphEmailAgent

__all__ = [
    "LangGraphEmailAssistant",
    "SimLabLangGraphEmailAgent",
    "build_chat_model_from_env",
]
