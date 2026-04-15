"""Core NPC chat simulation engine.

Each NPC with personality fields (name, role, traits, context, quirks)
gets an LLM-backed chat persona.  Conversation history is maintained
per-NPC within a rollout and resets on ``reset()``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)

# Default maximum messages kept per NPC to bound context window usage.
_DEFAULT_MAX_HISTORY = 40  # 20 turns (user + assistant pairs)


class NpcChatProfile(BaseModel):
    """Chat personality configuration for an NPC.

    These fields are optional extensions on the standard NPC entry in task JSON.
    When present, the npc-chat tool auto-activates for that NPC.
    """

    name: str = Field(..., description="Display name, e.g. 'Sarah Chen'")
    role: str = Field(..., description="Role or archetype, e.g. 'frustrated_customer'")
    traits: str = Field("", description="Personality traits, e.g. 'impatient, detail-oriented'")
    context: str = Field("", description="Situational background for this NPC")
    quirks: str = Field("", description="Communication quirks or habits")


class NpcChatServer:
    """In-process NPC chat tool server backed by LiteLLM.

    Exposes two tools via the standard ``list_tools`` / ``call_tool`` contract:

    * ``npc_send_message`` — send a message to a named NPC and get
      a synchronous in-character response.
    * ``npc_list_contacts`` — list available NPCs and their roles.
    """

    def __init__(  # noqa: D107
        self,
        npcs: list[NpcChatProfile],
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        max_history: int = _DEFAULT_MAX_HISTORY,
    ) -> None:
        if not npcs:
            raise ValueError("At least one NPC with chat personality is required")

        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._provider = provider
        self._max_history = max_history

        self._npcs: list[NpcChatProfile] = list(npcs)
        self._by_name: dict[str, NpcChatProfile] = {npc.name.strip().lower(): npc for npc in npcs}

        self._conversations: dict[str, list[dict[str, str]]] = {
            npc.name.strip().lower(): [] for npc in npcs
        }

    # ------------------------------------------------------------------
    # Public API (tool server contract)
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions in the standard RL-Gym format."""
        npc_names = [npc.name for npc in self._npcs]
        return [
            {
                "name": "npc_send_message",
                "description": (
                    "Send a message to a person and receive their response. "
                    "Available contacts: " + ", ".join(npc_names) + "."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Name of the person to message. One of: "
                            + ", ".join(npc_names),
                        },
                        "message": {
                            "type": "string",
                            "description": "The message to send.",
                        },
                    },
                    "required": ["to", "message"],
                },
                "mutates_state": True,
            },
            {
                "name": "npc_list_contacts",
                "description": "List available contacts and their roles.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str, Any | None]:
        """Execute a tool and return ``(is_error, text, structured_content)``."""
        if name == "npc_send_message":
            return self._handle_send_message(arguments)
        if name == "npc_list_contacts":
            return self._handle_list_contacts()
        return True, f"Unknown tool: {name}", None

    def get_transcripts(self) -> dict[str, Any]:
        """Return full conversation transcripts keyed by NPC name."""
        result: dict[str, Any] = {}
        for npc in self._npcs:
            key = npc.name.strip().lower()
            messages = list(self._conversations.get(key, []))
            result[npc.name] = {
                "role": npc.role,
                "messages": messages,
            }
        return result

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_send_message(self, arguments: dict[str, Any]) -> tuple[bool, str, Any | None]:
        to = str(arguments.get("to", "")).strip()
        message = str(arguments.get("message", "")).strip()

        if not to:
            return True, "Missing required parameter: 'to'", None
        if not message:
            return True, "Missing required parameter: 'message'", None

        npc = self._resolve_npc(to)
        if npc is None:
            available = ", ".join(n.name for n in self._npcs)
            return True, f"Unknown contact: '{to}'. Available: {available}", None

        key = npc.name.strip().lower()
        history = self._conversations[key]

        history.append({"role": "user", "content": message})

        system_prompt = self._build_system_prompt(npc)
        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        truncated = history[-self._max_history :]
        llm_messages.extend(truncated)

        try:
            response_text = self._call_llm(llm_messages)
        except Exception as exc:
            logger.exception("NPC LLM call failed for %s", npc.name)
            if history and history[-1].get("role") == "user":
                history.pop()
            return True, f"NPC chat error: {exc}", None

        history.append({"role": "assistant", "content": response_text})

        payload = {
            "from": npc.name,
            "role": npc.role,
            "message": response_text,
        }
        return False, json.dumps(payload, indent=2, ensure_ascii=False), payload

    def _handle_list_contacts(self) -> tuple[bool, str, Any | None]:
        contacts = [{"name": npc.name, "role": npc.role} for npc in self._npcs]
        return (
            False,
            json.dumps(contacts, indent=2, ensure_ascii=False),
            contacts,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_npc(self, name: str) -> NpcChatProfile | None:
        """Resolve an NPC by exact name (case-insensitive)."""
        return self._by_name.get(name.strip().lower())

    @staticmethod
    def _build_system_prompt(npc: NpcChatProfile) -> str:
        """Build the LLM system prompt from an NPC chat profile."""
        parts = [f"You are {npc.name}, a {npc.role}."]
        if npc.traits:
            parts.append(f"Personality: {npc.traits}.")
        if npc.context:
            parts.append(f"Situation: {npc.context}")
        if npc.quirks:
            parts.append(f"Communication style: {npc.quirks}.")
        parts.append(
            "Stay in character at all times. Respond naturally and concisely "
            "to the message. Do not break character or mention that you are an AI."
        )
        return "\n".join(parts)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call LiteLLM and return the response text."""
        import litellm  # noqa: PLC0415

        model = (self._model or "").strip()
        provider = (self._provider or "").strip().lower()
        if provider:
            if not model.startswith(f"{provider}/"):
                model = f"{provider}/{model}"
        elif "/" not in model:
            model = f"openai/{model}"

        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=self._api_key,
            base_url=self._base_url,
        )
        content = response.choices[0].message.content  # type: ignore[union-attr]
        return content or ""
