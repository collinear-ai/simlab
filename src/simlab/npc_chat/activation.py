"""NPC chat tool server lifecycle management for task runs.

Provides ``NpcChatSession`` — a self-contained helper that both
``tasks.py`` and ``parallel_daytona.py`` use to start/stop the
NPC chat tool server and save conversation transcripts.

NPCs that have chat personality fields (name, role, traits, context, quirks)
in the task JSON's ``npcs`` list are automatically detected and used
to start the npc-chat tool.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from simlab.config import load_global_config
from simlab.npc_chat.http_server import NpcChatServerHandle
from simlab.npc_chat.server import NpcChatProfile
from simlab.npc_chat.server import NpcChatServer

logger = logging.getLogger(__name__)

# NPC entries with any of these fields are considered chat-capable.
_CHAT_PERSONALITY_FIELDS = {"traits", "context", "quirks"}


def _extract_chat_npcs(npcs: list[dict[str, Any]]) -> list[NpcChatProfile]:
    """Filter and validate NPC entries that have chat personality fields.

    An NPC entry like ``{"id": "emily_davis", "secret": "..."}`` is a
    traditional seed-data NPC.  One like ``{"id": "sarah_chen", "name":
    "Sarah Chen", "role": "frustrated_customer", "traits": "impatient"}``
    has chat personality and will power the npc-chat tool.

    Returns validated ``NpcChatProfile`` instances.
    """
    result: list[NpcChatProfile] = []
    for npc in npcs:
        if not isinstance(npc, dict):
            continue
        if not any(npc.get(field) for field in _CHAT_PERSONALITY_FIELDS):
            continue
        # Ensure name exists — fall back to id if missing.
        data = dict(npc)
        if not data.get("name") and data.get("id"):
            data["name"] = str(data["id"]).replace("_", " ").title()
        try:
            result.append(NpcChatProfile.model_validate(data))
        except ValidationError as exc:
            logger.warning("Skipping invalid NPC chat profile %s: %s", data.get("name", "?"), exc)
    return result


class NpcChatSession:
    """Manages the NPC chat tool server lifecycle for a single rollout.

    Usage::

        session = NpcChatSession.from_task_data(task_data, model=..., api_key=...)
        if session is not None:
            url = session.start()
            tool_endpoints["npc-chat"] = url
            try:
                # ... run agent ...
                pass
            finally:
                session.attach_to_artifacts(artifacts)
                session.save_transcripts(output_dir)
                session.stop()
    """

    def __init__(  # noqa: D107
        self,
        profiles: list[NpcChatProfile],
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._server = NpcChatServer(
            profiles,
            model=model,
            api_key=api_key,
            base_url=base_url,
            provider=provider,
        )
        self._handle = NpcChatServerHandle(self._server)
        self._url: str | None = None

    @property
    def url(self) -> str:
        """Base URL of the running NPC chat tool server."""
        if self._url is None:
            raise RuntimeError("NpcChatSession not started; call start() first")
        return self._url

    def start(self) -> str:
        """Start the NPC chat tool server and return its URL."""
        self._url = self._handle.start()
        return self._url

    def stop(self) -> None:
        """Stop the NPC chat tool server."""
        self._handle.stop()

    def get_transcripts(self) -> dict[str, Any]:
        """Return conversation transcripts from the NPC chat server."""
        return self._server.get_transcripts()

    def attach_to_artifacts(self, artifacts: Any) -> None:  # noqa: ANN401
        """Attach NPC chat transcripts to RunArtifacts.metadata."""
        transcripts = self.get_transcripts()
        if not transcripts:
            return
        if not hasattr(artifacts, "metadata"):
            return
        artifacts.metadata["npc_chat_transcripts"] = transcripts
        total_messages = sum(len(data.get("messages", [])) for data in transcripts.values())
        artifacts.metadata["npc_chat_message_count"] = total_messages

    def save_transcripts(self, output_dir: Path) -> None:
        """Save conversation transcripts as JSON."""
        transcripts = self.get_transcripts()
        if not transcripts:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "npc_chat_transcripts.json"
        json_path.write_text(
            json.dumps(transcripts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Saved NPC chat transcripts to %s", output_dir)

    @staticmethod
    def from_task_data(
        task_data: dict[str, Any],
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
    ) -> NpcChatSession | None:
        """Create an NpcChatSession if the task has NPCs with chat personality.

        Scans the ``npcs`` list in ``task_data`` for entries that have
        personality fields (traits, context, quirks).  Returns ``None``
        if no chat-capable NPCs are found.

        Credential resolution (highest priority wins):
        1. ``[npc_chat]`` section in config file (via GlobalConfig)
        2. ``SIMLAB_NPC_CHAT_*`` env vars (via GlobalConfig env overlay)
        3. Explicit parameters (from agent settings)
        4. LiteLLM provider env vars (OPENAI_API_KEY, etc.) at call time
        """
        all_npcs = task_data.get("npcs") or []
        chat_npcs = _extract_chat_npcs(all_npcs)
        if not chat_npcs:
            return None

        cfg = load_global_config()
        resolved_model = cfg.npc_chat_model or model or "gpt-4o-mini"
        resolved_api_key = cfg.npc_chat_api_key or api_key
        resolved_base_url = cfg.npc_chat_base_url or base_url
        resolved_provider = cfg.npc_chat_provider or provider

        if not resolved_api_key:
            logger.info(
                "No explicit NPC chat API key found; LiteLLM will attempt "
                "provider-specific env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)"
            )

        return NpcChatSession(
            chat_npcs,
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            provider=resolved_provider,
        )
