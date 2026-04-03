"""Runtime URL rewriting helpers for Harbor-generated environments."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import yaml


def compose_service_host_port(compose_data: dict[str, Any], service_name: str) -> int | None:
    """Extract mapped host port for a docker-compose service, if available."""
    services = compose_data.get("services", {})
    if not isinstance(services, dict):
        return None
    service = services.get(service_name)
    if not isinstance(service, dict):
        return None
    ports = service.get("ports", [])
    if not isinstance(ports, list):
        return None
    for entry in ports:
        if isinstance(entry, str) and ":" in entry:
            host = entry.split(":", 1)[0].strip().strip('"').strip("'")
            if host.isdigit():
                return int(host)
    return None


def rewrite_mcp_config_for_runtime(
    mcp_config: dict[str, Any] | None,
    *,
    env_dir: Path,
    using_daytona: bool,
    daytona_client_factory: Any,  # noqa: ANN401
    daytona_api_key: str | None = None,
) -> dict[str, Any] | None:
    """Rewrite Harbor MCP URLs to the runtime-visible local or Daytona host URLs."""
    if mcp_config is None:
        return None
    rewritten = copy.deepcopy(mcp_config)
    servers = rewritten.get("mcpServers")
    if not isinstance(servers, dict):
        return rewritten
    for entry in servers.values():
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if isinstance(url, str) and url:
            entry["url"] = compose_service_host_url(
                url,
                env_dir=env_dir,
                using_daytona=using_daytona,
                daytona_client_factory=daytona_client_factory,
                daytona_api_key=daytona_api_key,
            )
    return rewritten


def compose_service_host_url(
    raw_url: str,
    *,
    env_dir: Path,
    using_daytona: bool,
    daytona_client_factory: Any,  # noqa: ANN401
    daytona_api_key: str | None = None,
) -> str:
    """Rewrite a compose-service URL to a host-visible local or Daytona preview URL."""
    parsed = urlsplit(raw_url)
    if not parsed.hostname:
        return raw_url

    compose_path = env_dir / "docker-compose.yml"
    if not compose_path.is_file():
        return raw_url
    compose_data = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
    host_port = compose_service_host_port(compose_data, parsed.hostname)
    if host_port is None:
        return raw_url

    if using_daytona:
        state_file = env_dir / "daytona-state.json"
        if not state_file.is_file():
            return raw_url
        state = json.loads(state_file.read_text(encoding="utf-8"))
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return raw_url
        sandbox = daytona_client_factory(daytona_api_key=daytona_api_key).get(sandbox_id)
        preview_parts = urlsplit(sandbox.get_preview_link(host_port).url)
        return urlunsplit(
            (
                preview_parts.scheme or parsed.scheme or "http",
                preview_parts.netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )

    return urlunsplit(
        (
            parsed.scheme or "http",
            f"localhost:{host_port}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )
