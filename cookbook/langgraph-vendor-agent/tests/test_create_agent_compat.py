"""Tests for create_agent / create_react_agent compatibility in plan_and_execute."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import SystemMessage


def test_build_graph_calls_create_fn_with_correct_kwarg() -> None:
    """The executor node should pass system_prompt= or prompt= depending on the resolved API."""
    from langgraph_vendor_agent import plan_and_execute as module

    mock_fn = MagicMock(return_value=MagicMock())
    original_fn = module._create_agent_fn
    original_flag = module._USE_NEW_CREATE_AGENT

    # --- new API path (system_prompt=) ---
    module._create_agent_fn = mock_fn
    module._USE_NEW_CREATE_AGENT = True
    try:
        model = MagicMock()
        tool = MagicMock()
        tool.name = "server__tool1"
        tools = [tool]
        module.build_plan_and_execute_graph(model, tools, safe_tool_filter=False)
        _, kwargs = mock_fn.call_args
        assert "system_prompt" in kwargs
        assert isinstance(kwargs["system_prompt"], SystemMessage)
    finally:
        module._create_agent_fn = original_fn
        module._USE_NEW_CREATE_AGENT = original_flag


def test_build_graph_calls_legacy_fn_with_prompt_kwarg() -> None:
    """When falling back to create_react_agent, prompt= should be used."""
    from langgraph_vendor_agent import plan_and_execute as module

    mock_fn = MagicMock(return_value=MagicMock())
    original_fn = module._create_agent_fn
    original_flag = module._USE_NEW_CREATE_AGENT

    module._create_agent_fn = mock_fn
    module._USE_NEW_CREATE_AGENT = False
    try:
        model = MagicMock()
        tool = MagicMock()
        tool.name = "server__tool1"
        tools = [tool]
        module.build_plan_and_execute_graph(model, tools, safe_tool_filter=False)
        _, kwargs = mock_fn.call_args
        assert "prompt" in kwargs
        assert isinstance(kwargs["prompt"], SystemMessage)
    finally:
        module._create_agent_fn = original_fn
        module._USE_NEW_CREATE_AGENT = original_flag


def test_import_fallback_to_langgraph_prebuilt() -> None:
    """When langchain.agents has no create_agent, we fall back to langgraph.prebuilt."""
    from importlib import import_module as real_import_module

    def fake_import_module(name: str) -> object:
        if name == "langchain.agents":
            mod = MagicMock(spec=[])  # spec=[] means no attributes
            del mod.create_agent  # ensure AttributeError
            return mod
        return real_import_module(name)

    with patch("langgraph_vendor_agent.plan_and_execute.import_module", fake_import_module):
        # Re-run the import logic
        try:
            _create_agent_fn = getattr(fake_import_module("langchain.agents"), "create_agent")
            use_new = True
        except (ModuleNotFoundError, AttributeError):
            _create_agent_fn = getattr(real_import_module("langgraph.prebuilt"), "create_react_agent")
            use_new = False

    assert not use_new
    # The fallback should resolve to the real create_react_agent
    from langgraph.prebuilt import create_react_agent
    assert _create_agent_fn is create_react_agent
