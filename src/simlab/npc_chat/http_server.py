"""HTTP wrapper for NpcChatServer.

Starts a lightweight FastAPI server on a random localhost port so the
NPC chat tool integrates with ``UnifiedToolEnvironment`` via standard
HTTP tool-server endpoints (GET /health, GET /tools, POST /step).
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any

import requests as http_requests

from simlab.npc_chat.server import NpcChatServer

logger = logging.getLogger(__name__)


class NpcChatServerHandle:
    """Manages an NPC chat HTTP server on a random localhost port.

    Wraps ``NpcChatServer`` (pure business logic) in a FastAPI HTTP layer
    so that ``UnifiedToolEnvironment`` can discover and call it.

    Usage::

        handle = NpcChatServerHandle(npc_server)
        url = handle.start()          # "http://127.0.0.1:54321"
        # ... agent uses the URL ...
        handle.stop()
    """

    def __init__(self, npc_server: NpcChatServer) -> None:  # noqa: D107
        if not isinstance(npc_server, NpcChatServer):
            raise TypeError(f"Expected NpcChatServer, got {type(npc_server).__name__}")
        self._server = npc_server
        self._port: int | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None

    @property
    def server(self) -> NpcChatServer:
        """The underlying NpcChatServer instance."""
        return self._server

    @property
    def url(self) -> str:
        """The base URL of the running server."""
        if self._url is None:
            raise RuntimeError("Server not started; call start() first")
        return self._url

    def start(self) -> str:
        """Start the HTTP server in a daemon thread and return its base URL."""
        if self._thread is not None and self._thread.is_alive():
            return self.url

        self._port = _find_free_port()
        self._url = f"http://127.0.0.1:{self._port}"

        try:
            import uvicorn  # noqa: PLC0415

            app = self._build_app()
        except ImportError as exc:
            raise ImportError(
                "NPC chat requires the [npc] extra. Install with: pip install simlab[npc]"
            ) from exc

        def _run() -> None:
            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self._port,  # type: ignore[arg-type]
                log_level="warning",
            )
            server = uvicorn.Server(config)
            self._uvicorn_server = server
            server.run()

        self._thread = threading.Thread(target=_run, name="npc-chat-server", daemon=True)
        self._thread.start()

        self._wait_for_ready()

        logger.info("NPC chat server started at %s", self._url)
        return self._url

    def stop(self) -> None:
        """Stop the HTTP server."""
        uvicorn_server = getattr(self, "_uvicorn_server", None)
        if uvicorn_server is not None:
            uvicorn_server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("NPC chat server stopped")

    def _build_app(self) -> Any:  # noqa: ANN401
        """Build a FastAPI application wrapping the NpcChatServer."""
        from simlab.npc_chat.app import create_app  # noqa: PLC0415

        return create_app(self._server)

    def _wait_for_ready(self, timeout: float = 10.0) -> None:
        """Poll the health endpoint until the server responds."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = http_requests.get(f"{self._url}/health", timeout=2)
                if resp.status_code == 200:
                    return
            except http_requests.ConnectionError:
                pass
            time.sleep(0.1)
        logger.warning("NPC chat server did not become ready within %.0fs", timeout)


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
