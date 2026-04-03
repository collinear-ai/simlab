from __future__ import annotations

import asyncio
from typing import Any
from typing import Protocol
from typing import cast

import pytest
from simlab.agents.adapters import google_adk

from tests.adapter_contract import ToolAdapterHarness
from tests.adapter_contract import assert_exposes_demo_tools
from tests.adapter_contract import assert_records_tool_invocation
from tests.adapter_contract import assert_rejects_duplicate_wire_names


class FakeBaseTool:
    def __init__(self, *, name: str, description: str, **_: object) -> None:
        self.name = name
        self.description = description


class FakeFunctionDeclaration:
    def __init__(
        self,
        *,
        name: str,
        description: str | None = None,
        parameters_json_schema: dict[str, Any] | None = None,
        **_: object,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters_json_schema = parameters_json_schema


class GoogleAdkToolLike(Protocol):
    name: str

    async def run_async(self, *, args: dict[str, Any], tool_context: object) -> object: ...


class GoogleAdkHarness(ToolAdapterHarness):
    def build_tools(self, environment, *, recorder=None) -> list[object]:  # noqa: ANN001
        return google_adk.build_google_adk_tools(environment, recorder=recorder)

    def tool_name(self, tool: object) -> str:
        return cast(GoogleAdkToolLike, tool).name

    def invoke(self, tool: object, payload: dict[str, Any]) -> object:
        return asyncio.run(cast(GoogleAdkToolLike, tool).run_async(args=payload, tool_context=None))


def test_build_google_adk_tools_exposes_http_and_mcp_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(google_adk, "adk_base_tool_class", FakeBaseTool)
    monkeypatch.setattr(google_adk, "genai_function_declaration_class", FakeFunctionDeclaration)
    assert_exposes_demo_tools(GoogleAdkHarness())


def test_google_adk_tool_invocation_records_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(google_adk, "adk_base_tool_class", FakeBaseTool)
    monkeypatch.setattr(google_adk, "genai_function_declaration_class", FakeFunctionDeclaration)
    assert_records_tool_invocation(GoogleAdkHarness())


def test_google_adk_tools_reject_duplicate_wire_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(google_adk, "adk_base_tool_class", FakeBaseTool)
    monkeypatch.setattr(google_adk, "genai_function_declaration_class", FakeFunctionDeclaration)
    assert_rejects_duplicate_wire_names(GoogleAdkHarness())
