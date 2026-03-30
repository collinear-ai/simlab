"""Tests for the NPC chat tool server, session lifecycle, and HTTP wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests
from simlab.npc_chat.activation import NpcChatSession
from simlab.npc_chat.activation import _extract_chat_npcs
from simlab.npc_chat.http_server import NpcChatServerHandle
from simlab.npc_chat.server import NpcChatProfile
from simlab.npc_chat.server import NpcChatServer


def _npc(
    name: str = "Test",
    role: str = "tester",
    traits: str = "",
    context: str = "",
    quirks: str = "",
) -> NpcChatProfile:
    return NpcChatProfile(name=name, role=role, traits=traits, context=context, quirks=quirks)


@pytest.fixture
def sarah() -> NpcChatProfile:
    return _npc(
        name="Sarah Chen",
        role="frustrated_customer",
        traits="impatient, detail-oriented",
        context="Waiting 3 days for a $299 refund.",
        quirks="Asks to speak to a manager if not resolved quickly",
    )


@pytest.fixture
def mike() -> NpcChatProfile:
    return _npc(
        name="Mike Johnson",
        role="new_customer",
        traits="friendly, confused",
        context="First time user.",
    )


@pytest.fixture
def server(sarah: NpcChatProfile, mike: NpcChatProfile) -> NpcChatServer:
    return NpcChatServer([sarah, mike], model="gpt-4o-mini", api_key="fake-key")


def _mock_llm_response(text: str = "Mock response") -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = text
    return mock_resp


# ---------------------------------------------------------------------------
# _extract_chat_npcs
# ---------------------------------------------------------------------------


class TestExtractChatNpcs:
    def test_skips_seed_only_npcs(self) -> None:
        npcs = [{"id": "emily_davis", "secret": "prefers mornings"}]
        assert _extract_chat_npcs(npcs) == []

    def test_extracts_npcs_with_traits(self) -> None:
        npcs = [
            {"id": "sarah_chen", "name": "Sarah Chen", "role": "customer", "traits": "impatient"},
        ]
        result = _extract_chat_npcs(npcs)
        assert len(result) == 1
        assert result[0].name == "Sarah Chen"

    def test_extracts_npcs_with_context(self) -> None:
        npcs = [{"id": "mike", "name": "Mike", "role": "user", "context": "first time user"}]
        result = _extract_chat_npcs(npcs)
        assert len(result) == 1

    def test_extracts_npcs_with_quirks(self) -> None:
        npcs = [{"id": "joe", "name": "Joe", "role": "manager", "quirks": "terse"}]
        result = _extract_chat_npcs(npcs)
        assert len(result) == 1

    def test_fallback_name_from_id(self) -> None:
        npcs = [{"id": "sarah_chen", "role": "customer", "traits": "impatient"}]
        result = _extract_chat_npcs(npcs)
        assert len(result) == 1
        assert result[0].name == "Sarah Chen"

    def test_returns_validated_profiles(self) -> None:
        npcs = [{"name": "Test", "role": "tester", "traits": "friendly"}]
        result = _extract_chat_npcs(npcs)
        assert len(result) == 1
        assert isinstance(result[0], NpcChatProfile)

    def test_mixed_npcs(self) -> None:
        npcs = [
            {"id": "emily_davis", "secret": "prefers mornings"},
            {"id": "sarah_chen", "name": "Sarah Chen", "role": "customer", "traits": "impatient"},
        ]
        result = _extract_chat_npcs(npcs)
        assert len(result) == 1
        assert result[0].name == "Sarah Chen"


# ---------------------------------------------------------------------------
# NpcChatServer — list_tools
# ---------------------------------------------------------------------------


class TestListTools:
    def test_returns_two_tools(self, server: NpcChatServer) -> None:
        assert len(server.list_tools()) == 2

    def test_tool_names(self, server: NpcChatServer) -> None:
        names = [t["name"] for t in server.list_tools()]
        assert "npc_send_message" in names
        assert "npc_list_contacts" in names

    def test_send_message_is_mutating(self, server: NpcChatServer) -> None:
        send = next(t for t in server.list_tools() if t["name"] == "npc_send_message")
        assert send["mutates_state"] is True

    def test_no_simulated_language(self, server: NpcChatServer) -> None:
        send = next(t for t in server.list_tools() if t["name"] == "npc_send_message")
        assert "simulated" not in send["description"].lower()
        assert "in-character" not in send["description"].lower()


# ---------------------------------------------------------------------------
# NpcChatServer — call_tool (npc_list_contacts)
# ---------------------------------------------------------------------------


class TestListContacts:
    def test_returns_all_npcs(self, server: NpcChatServer) -> None:
        is_err, _text, data = server.call_tool("npc_list_contacts", {})
        assert not is_err
        assert len(data) == 2

    def test_npc_fields(self, server: NpcChatServer) -> None:
        _, _, data = server.call_tool("npc_list_contacts", {})
        names = {p["name"] for p in data}
        assert names == {"Sarah Chen", "Mike Johnson"}


# ---------------------------------------------------------------------------
# NpcChatServer — _resolve_npc
# ---------------------------------------------------------------------------


class TestResolveNpc:
    def test_exact_match(self, server: NpcChatServer) -> None:
        assert server._resolve_npc("Sarah Chen") is not None

    def test_case_insensitive(self, server: NpcChatServer) -> None:
        assert server._resolve_npc("sarah chen") is not None

    def test_first_name_only_does_not_match(self, server: NpcChatServer) -> None:
        assert server._resolve_npc("Sarah") is None

    def test_unknown_returns_none(self, server: NpcChatServer) -> None:
        assert server._resolve_npc("Unknown") is None

    def test_empty_returns_none(self, server: NpcChatServer) -> None:
        assert server._resolve_npc("") is None


# ---------------------------------------------------------------------------
# NpcChatServer — _build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_contains_name_and_role(self, sarah: NpcChatProfile) -> None:
        prompt = NpcChatServer._build_system_prompt(sarah)
        assert "Sarah Chen" in prompt
        assert "frustrated_customer" in prompt

    def test_no_ai_mention(self, sarah: NpcChatProfile) -> None:
        prompt = NpcChatServer._build_system_prompt(sarah)
        assert "Do not break character" in prompt


# ---------------------------------------------------------------------------
# NpcChatServer — call_tool (npc_send_message)
# ---------------------------------------------------------------------------


class TestSendMessage:
    def test_success(self, server: NpcChatServer) -> None:
        with patch("litellm.completion", return_value=_mock_llm_response("I need my refund!")):
            is_err, _text, data = server.call_tool(
                "npc_send_message", {"to": "Sarah Chen", "message": "Hi"}
            )
        assert not is_err
        assert data["from"] == "Sarah Chen"
        assert data["message"] == "I need my refund!"

    def test_unknown_npc_error(self, server: NpcChatServer) -> None:
        is_err, text, _ = server.call_tool("npc_send_message", {"to": "Nobody", "message": "Hi"})
        assert is_err
        assert "Unknown contact" in text

    def test_missing_to_error(self, server: NpcChatServer) -> None:
        is_err, _text, _ = server.call_tool("npc_send_message", {"to": "", "message": "Hi"})
        assert is_err

    def test_missing_message_error(self, server: NpcChatServer) -> None:
        is_err, _text, _ = server.call_tool("npc_send_message", {"to": "Sarah Chen", "message": ""})
        assert is_err

    def test_history_accumulates(self, server: NpcChatServer) -> None:
        with patch("litellm.completion", return_value=_mock_llm_response("Reply 1")):
            server.call_tool("npc_send_message", {"to": "Sarah Chen", "message": "Msg 1"})
        with patch("litellm.completion", return_value=_mock_llm_response("Reply 2")):
            server.call_tool("npc_send_message", {"to": "Sarah Chen", "message": "Msg 2"})

        transcripts = server.get_transcripts()
        assert len(transcripts["Sarah Chen"]["messages"]) == 4

    def test_llm_error_returns_error(self, server: NpcChatServer) -> None:
        with patch("litellm.completion", side_effect=RuntimeError("LLM down")):
            is_err, text, _ = server.call_tool(
                "npc_send_message", {"to": "Sarah Chen", "message": "Hi"}
            )
        assert is_err
        assert "NPC chat error" in text

    def test_llm_error_does_not_corrupt_history(self, server: NpcChatServer) -> None:
        with patch("litellm.completion", side_effect=RuntimeError("LLM down")):
            server.call_tool("npc_send_message", {"to": "Sarah Chen", "message": "Hi"})
        assert len(server.get_transcripts()["Sarah Chen"]["messages"]) == 0


# ---------------------------------------------------------------------------
# NpcChatServer — transcripts / validation
# ---------------------------------------------------------------------------


class TestTranscriptsAndValidation:
    def test_empty_server_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one NPC"):
            NpcChatServer([], model="gpt-4o-mini", api_key="fake")


# ---------------------------------------------------------------------------
# NpcChatSession — from_task_data
# ---------------------------------------------------------------------------


class TestNpcChatSessionFromTaskData:
    def test_returns_none_for_empty_task(self) -> None:
        assert NpcChatSession.from_task_data({}) is None

    def test_returns_none_for_seed_only_npcs(self) -> None:
        task = {"npcs": [{"id": "emily_davis", "secret": "prefers mornings"}]}
        assert NpcChatSession.from_task_data(task) is None

    def test_creates_session_for_chat_npcs(self) -> None:
        task = {
            "npcs": [
                {"id": "sarah", "name": "Sarah", "role": "customer", "traits": "impatient"},
            ]
        }
        session = NpcChatSession.from_task_data(task, model="gpt-4o-mini", api_key="fake")
        assert session is not None

    def test_env_var_overrides_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIMLAB_NPC_CHAT_MODEL", "custom-model")
        task = {"npcs": [{"name": "Test", "role": "tester", "traits": "friendly"}]}
        session = NpcChatSession.from_task_data(task, model="gpt-4o-mini", api_key="fake")
        assert session is not None
        assert session._server._model == "custom-model"


# ---------------------------------------------------------------------------
# NpcChatSession — transcript saving
# ---------------------------------------------------------------------------


class TestNpcChatSessionTranscripts:
    def test_save_creates_json(self, tmp_path: Path) -> None:
        session = NpcChatSession(
            [_npc(traits="friendly")],
            model="gpt-4o-mini",
            api_key="fake",
        )
        session.save_transcripts(tmp_path)
        assert (tmp_path / "npc_chat_transcripts.json").exists()

    def test_no_md_file_created(self, tmp_path: Path) -> None:
        session = NpcChatSession(
            [_npc(traits="friendly")],
            model="gpt-4o-mini",
            api_key="fake",
        )
        session.save_transcripts(tmp_path)
        assert not (tmp_path / "npc_chat_transcripts.md").exists()

    def test_attach_to_artifacts(self) -> None:
        class FakeArtifacts:
            def __init__(self) -> None:
                self.metadata: dict[str, Any] = {}

        session = NpcChatSession(
            [_npc(traits="friendly")],
            model="gpt-4o-mini",
            api_key="fake",
        )
        artifacts = FakeArtifacts()
        session.attach_to_artifacts(artifacts)
        assert "npc_chat_transcripts" in artifacts.metadata
        assert artifacts.metadata["npc_chat_message_count"] == 0


# ---------------------------------------------------------------------------
# NpcChatServerHandle — start / stop
# ---------------------------------------------------------------------------


def _make_handle() -> NpcChatServerHandle:
    server = NpcChatServer([_npc(traits="test")], model="gpt-4o-mini", api_key="fake")
    return NpcChatServerHandle(server)


_has_npc_extra = True
try:
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
except ImportError:
    _has_npc_extra = False


@pytest.mark.skipif(not _has_npc_extra, reason="requires simlab[npc] extra (fastapi + uvicorn)")
class TestNpcChatServerHandle:
    def test_start_returns_url(self) -> None:
        handle = _make_handle()
        url = handle.start()
        try:
            assert url.startswith("http://127.0.0.1:")
        finally:
            handle.stop()

    def test_health_endpoint(self) -> None:
        handle = _make_handle()
        url = handle.start()
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            assert resp.status_code == 200
        finally:
            handle.stop()

    def test_tools_endpoint(self) -> None:
        handle = _make_handle()
        url = handle.start()
        try:
            resp = requests.get(f"{url}/tools", timeout=5)
            tools = resp.json()["tools"]
            assert len(tools) == 2
        finally:
            handle.stop()

    def test_stop_shuts_down(self) -> None:
        handle = _make_handle()
        url = handle.start()
        handle.stop()
        with pytest.raises(requests.ConnectionError):
            requests.get(f"{url}/health", timeout=2)

    def test_type_check(self) -> None:
        with pytest.raises(TypeError, match="Expected NpcChatServer"):
            NpcChatServerHandle("not a server")  # type: ignore[arg-type]
