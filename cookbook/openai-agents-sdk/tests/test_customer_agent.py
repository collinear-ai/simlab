# ruff: noqa: D100, D101, D102, D103, S101

from __future__ import annotations

from typing import Protocol
from typing import cast

from agents.tool import Tool
import pytest

from openai_agents_sdk import custom_agent


class ResultLike(Protocol):
    final_output: object


def test_build_custom_agent_uses_resolved_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_AGENTS_SDK_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_AGENTS_SDK_INSTRUCTIONS", "Follow tools.")
    monkeypatch.setattr(custom_agent, "Agent", FakeAgent)

    custom_agent.build_custom_agent(tools=cast(list[Tool], [object()]))

    assert captured["name"] == "SimLab Custom Agent"
    assert captured["model"] == "gpt-test"
    assert captured["instructions"] == "Follow tools."
    assert len(cast(list[object], captured["tools"])) == 1


def test_run_custom_agent_uses_runner_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, **kwargs: object) -> None:
            captured["agent_kwargs"] = kwargs

    class FakeRunner:
        @staticmethod
        def run_sync(agent: object, instruction: str, *, max_turns: int) -> object:
            captured["agent"] = agent
            captured["instruction"] = instruction
            captured["max_turns"] = max_turns
            return type("Result", (), {"final_output": "done"})()

    monkeypatch.setattr(custom_agent, "Agent", FakeAgent)
    monkeypatch.setattr(custom_agent, "Runner", FakeRunner)

    result = cast(
        ResultLike,
        custom_agent.run_custom_agent(
            instruction="do the task",
            tools=cast(list[Tool], [object()]),
            model="gpt-runner",
            max_turns=7,
        ),
    )

    assert result.final_output == "done"
    assert captured["instruction"] == "do the task"
    assert captured["max_turns"] == 7
    assert cast(dict[str, object], captured["agent_kwargs"])["model"] == "gpt-runner"


def test_resolve_model_ignores_simlab_custom_agent_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_AGENTS_SDK_MODEL", raising=False)

    assert custom_agent.resolve_model("custom-agent") == custom_agent.DEFAULT_MODEL


def test_stringify_final_output_handles_structured_values() -> None:
    assert custom_agent.stringify_final_output({"ok": True}) == '{\n  "ok": true\n}'
