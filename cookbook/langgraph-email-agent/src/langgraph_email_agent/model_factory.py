"""Cookbook-owned model configuration."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

DEFAULT_BACKEND = "openai-compatible"


def build_chat_model_from_env() -> BaseChatModel:
    """Build the LangChain chat model configured for this cookbook."""
    backend = (
        os.getenv("LANGGRAPH_EMAIL_BACKEND", DEFAULT_BACKEND).strip() or DEFAULT_BACKEND
    )
    if backend != DEFAULT_BACKEND:
        raise ValueError(f"Unsupported backend: {backend}")

    model = os.getenv("LANGGRAPH_EMAIL_MODEL", "").strip()
    if not model:
        raise ValueError("LANGGRAPH_EMAIL_MODEL is required")

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": float(os.getenv("LANGGRAPH_EMAIL_TEMPERATURE", "0")),
    }
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        kwargs["api_key"] = api_key
    base_url = os.getenv("LANGGRAPH_EMAIL_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)
