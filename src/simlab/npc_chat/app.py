"""FastAPI application for the NPC chat tool server.

Provides ``create_app(server)`` — a factory that returns a configured
FastAPI instance wired to a live ``NpcChatServer``.  This keeps route
definitions in a standalone module rather than inline inside a method.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from simlab.npc_chat.server import NpcChatServer


def create_app(server: NpcChatServer) -> FastAPI:
    """Create a FastAPI app backed by the given NPC chat server."""
    app = FastAPI(title="NPC Chat Tool Server")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/tools")
    async def get_tools() -> dict[str, Any]:
        return {"tools": server.list_tools()}

    @app.post("/step")
    async def step(request: dict[str, Any]) -> dict[str, Any]:
        action = request.get("action", {}) or {}
        tool_name = action.get("tool_name") or action.get("kind") or ""
        parameters = action.get("parameters") or action.get("arguments") or {}

        is_error, text, structured = server.call_tool(tool_name, dict(parameters))

        return {
            "observation": {
                "is_error": is_error,
                "text": text,
                "content": [{"type": "text", "text": text}] if text else [],
                "structured_content": structured,
            },
            "reward": None,
            "done": False,
        }

    @app.get("/snapshot")
    async def snapshot() -> dict[str, Any]:
        transcripts = server.get_transcripts()
        lines = ["NPC Chat Transcripts:"]
        for name, data in transcripts.items():
            role = data.get("role", "")
            msgs = data.get("messages", [])
            lines.append(f"\n  [{name} ({role})]:")
            if msgs:
                for msg in msgs[-10:]:
                    speaker = name if msg["role"] == "assistant" else "Agent"
                    text = (msg.get("content") or "")[:80]
                    lines.append(f"    <{speaker}> {text}")
            else:
                lines.append("    (no messages)")
        return {
            "status": "ok",
            "tool_server": "npc-chat",
            "data": transcripts,
            "human_readable": "\n".join(lines),
        }

    return app
