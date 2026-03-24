"""Load and validate MCP servers configuration (mcpServers JSON)."""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

from simlab.catalog.registry import ToolRegistry

MCP_SERVERS_FILENAME = "mcp-servers.json"
MCP_SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
MCP_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MCP_GATEWAY_ENV_PREFIX = "SIMLAB_MCP_"


def _normalize_gateway_server_name(server_name: str) -> str:
    """Normalize a server name into the gateway env-var namespace segment."""
    return re.sub(r"[^A-Za-z0-9]+", "_", server_name).upper()


@cache
def _builtin_tool_server_names() -> frozenset[str]:
    """Return built-in tool server names from the catalog registry."""
    registry = ToolRegistry()
    registry.load_all()
    return frozenset(registry.tool_names)


def validate_mcp_servers_config(data: dict[str, Any]) -> None:
    """Validate MCP servers document. Raises ValueError/TypeError on invalid shape."""
    if not isinstance(data, dict):
        raise TypeError("MCP config must be a JSON object")
    servers = data.get("mcpServers")
    if servers is None:
        raise ValueError("MCP config must contain 'mcpServers'")
    if not isinstance(servers, dict):
        raise TypeError("mcpServers must be an object")
    normalized_names: dict[str, str] = {}
    for name, entry in servers.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MCP server name must be a non-empty string")
        if not MCP_SERVER_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"mcpServers.{name}: server names may contain only letters, numbers, '_' and '-'"
            )
        if name in _builtin_tool_server_names():
            raise ValueError(
                f"mcpServers.{name}: name conflicts with a built-in tool server; "
                "choose a different MCP server name"
            )
        normalized_name = _normalize_gateway_server_name(name)
        existing_name = normalized_names.get(normalized_name)
        if existing_name is not None and existing_name != name:
            raise ValueError(
                "mcpServers names must remain distinct after gateway env normalization: "
                f"{existing_name!r} and {name!r}"
            )
        normalized_names[normalized_name] = name
        if not isinstance(entry, dict):
            raise TypeError(f"mcpServers.{name} must be an object")
        has_url = "url" in entry and entry["url"] is not None
        has_command = "command" in entry and entry["command"] is not None
        if has_url and has_command:
            raise ValueError(f"mcpServers.{name}: use either 'url' or 'command', not both")
        if not has_url and not has_command:
            raise ValueError(f"mcpServers.{name}: must have 'url' or 'command'")
        if has_command:
            cmd = entry["command"]
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError(f"mcpServers.{name}.command must be a non-empty string")
            if "args" in entry and not isinstance(entry["args"], list):
                raise ValueError(f"mcpServers.{name}.args must be an array")
            if "env" in entry and not isinstance(entry["env"], dict):
                raise ValueError(f"mcpServers.{name}.env must be an object")
            for key in entry.get("env") or {}:
                if not isinstance(key, str) or not MCP_ENV_KEY_PATTERN.fullmatch(key):
                    raise ValueError(
                        f"mcpServers.{name}.env keys must be valid environment variable names"
                    )
        if has_url:
            url = entry["url"]
            if not isinstance(url, str) or not url.strip():
                raise ValueError(f"mcpServers.{name}.url must be a non-empty string")


def load_mcp_servers_config(path: Path) -> dict[str, Any]:
    """Load and validate MCP servers JSON from a file. Returns the full document."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    validate_mcp_servers_config(data)
    return data


def load_mcp_servers_from_env_dir(env_dir: Path) -> dict[str, Any] | None:
    """Load mcp-servers.json from env dir if present. Returns None if file missing."""
    config_file = env_dir / MCP_SERVERS_FILENAME
    if not config_file.is_file():
        return None
    return load_mcp_servers_config(config_file)


def get_mcp_server_urls(config: dict[str, Any]) -> dict[str, str]:
    """Return map of server name -> url for URL-based MCP servers."""
    servers = config.get("mcpServers") or {}
    return {
        name: str(entry["url"]).strip()
        for name, entry in servers.items()
        if isinstance(entry, dict) and entry.get("url")
    }


def get_mcp_command_servers(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return map of server name -> entry (command, args?, env?) for command-based MCP servers."""
    servers = config.get("mcpServers") or {}
    return {
        name: dict(entry)
        for name, entry in servers.items()
        if isinstance(entry, dict) and entry.get("command")
    }


def normalize_command_server_env(
    command_servers: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Return per-server env defaults normalized to string values."""
    normalized: dict[str, dict[str, str]] = {}
    for server_name, entry in command_servers.items():
        env = entry.get("env")
        if not isinstance(env, dict) or not env:
            continue
        normalized[server_name] = {
            str(key): "" if value is None else str(value) for key, value in env.items()
        }
    return normalized


def get_command_server_env_key_usage(
    command_servers: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Return the number of MCP command servers declaring each env key."""
    normalized = normalize_command_server_env(command_servers)
    counts: Counter[str] = Counter()
    for env in normalized.values():
        counts.update(env.keys())
    return dict(counts)


def build_scoped_gateway_env_var_name(server_name: str, env_key: str) -> str:
    """Return the gateway container env var name used for one server/key pair."""
    normalized_server = _normalize_gateway_server_name(server_name)
    return f"{MCP_GATEWAY_ENV_PREFIX}{normalized_server}__{env_key}"


def build_gateway_container_env(
    command_servers: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return gateway service env mappings and .env defaults for command servers.

    The gateway container only receives server-scoped variables. When an env key
    is unique across all command servers, the scoped variable is sourced from the
    legacy raw env name for backwards compatibility. Shared keys require the
    fully scoped env name in `.env`.
    """
    gateway_env: dict[str, str] = {}
    env_defaults: dict[str, str] = {}
    normalized = normalize_command_server_env(command_servers)
    key_usage = get_command_server_env_key_usage(command_servers)
    for server_name, env in normalized.items():
        for key, default in env.items():
            scoped_name = build_scoped_gateway_env_var_name(server_name, key)
            source_name = key if key_usage.get(key, 0) == 1 else scoped_name
            gateway_env[scoped_name] = f"${{{source_name}}}"
            env_defaults.setdefault(source_name, default)
    return gateway_env, env_defaults


def resolve_gateway_server_env(
    *,
    server_name: str,
    configured_env: dict[str, str],
    container_env: dict[str, str],
    key_usage: dict[str, int],
) -> dict[str, str]:
    """Resolve per-server subprocess env from config defaults and container env."""
    resolved: dict[str, str] = {}
    for key, default in configured_env.items():
        scoped_name = build_scoped_gateway_env_var_name(server_name, key)
        if scoped_name in container_env:
            resolved[key] = container_env[scoped_name]
        elif key_usage.get(key, 0) == 1 and key in container_env:
            resolved[key] = container_env[key]
        else:
            resolved[key] = default
    return resolved


def build_gateway_config_arena_format(command_servers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build gateway config in arena format: servers with transport and single command string."""
    servers = []
    normalized_env = normalize_command_server_env(command_servers)
    for name, entry in command_servers.items():
        cmd = entry.get("command", "")
        args = entry.get("args") or []
        if not isinstance(cmd, str):
            cmd = str(cmd)
        parts = [cmd] + [str(a) for a in args]
        command_str = shlex.join(parts)
        server_config: dict[str, Any] = {
            "name": name,
            "transport": "stdio",
            "command": command_str,
        }
        env = normalized_env.get(name)
        if env:
            server_config["env"] = env
        servers.append(server_config)
    return {"servers": servers}
