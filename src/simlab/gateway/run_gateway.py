"""Minimal MCP gateway: reads arena-format config, runs stdio servers, exposes streamable HTTP."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

_MCP_GATEWAY_ENV_PREFIX = "SIMLAB_MCP_"


def _build_scoped_gateway_env_var_name(server_name: str, env_key: str) -> str:
    """Return the gateway container env var name used for one server/key pair."""
    normalized_server = re.sub(r"[^A-Za-z0-9]+", "_", server_name).upper()
    return f"{_MCP_GATEWAY_ENV_PREFIX}{normalized_server}__{env_key}"


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
        scoped_name = _build_scoped_gateway_env_var_name(server_name, key)
        if scoped_name in container_env:
            resolved[key] = container_env[scoped_name]
        elif key_usage.get(key, 0) == 1 and key in container_env:
            resolved[key] = container_env[key]
        else:
            resolved[key] = default
    return resolved


def load_config() -> str:
    """Load gateway config from CONFIG_PATH or MCP_GATEWAY_CONFIG env."""
    config_path = os.environ.get("CONFIG_PATH")
    if config_path and Path(config_path).is_file():
        return Path(config_path).read_text(encoding="utf-8")
    raw = os.environ.get("MCP_GATEWAY_CONFIG")
    if raw:
        return raw
    raise RuntimeError("Set CONFIG_PATH or MCP_GATEWAY_CONFIG")


def parse_stdio_servers(config_json: str) -> list[tuple[str, str, list[str], dict[str, str]]]:
    """Return (name, command, args, env) for each stdio server (arena format)."""
    config = json.loads(config_json)
    servers = config.get("servers", [])
    if not isinstance(servers, list):
        return []
    result: list[tuple[str, str, list[str], dict[str, str]]] = []
    for entry in servers:
        if not isinstance(entry, dict) or entry.get("transport") != "stdio":
            continue
        command = entry.get("command")
        if not isinstance(command, str) or not command.strip():
            continue
        name = (entry.get("name") or "stdio").strip() or "stdio"
        parts = shlex.split(command)
        if not parts:
            continue
        env = entry.get("env")
        scoped_env = (
            {str(key): "" if value is None else str(value) for key, value in env.items()}
            if isinstance(env, dict)
            else {}
        )
        result.append((name, parts[0], parts[1:], scoped_env))
    return result


def main() -> None:
    """Run the MCP gateway (streamable-http) with config from env."""
    host = os.environ.get("MCP_GATEWAY_HOST", "0.0.0.0")  # noqa: S104
    port = int(os.environ.get("MCP_GATEWAY_PORT", "8080"))
    config_raw = load_config()
    stdio_servers = parse_stdio_servers(config_raw)
    if not stdio_servers:
        raise RuntimeError("No stdio MCP servers in config")

    from fastmcp import FastMCP  # noqa: PLC0415
    from fastmcp.client import StdioTransport  # noqa: PLC0415

    gateway = FastMCP("simlab-mcp-gateway")
    server_env_defaults = {name: env for name, _command, _args, env in stdio_servers}
    key_usage: dict[str, int] = {}
    if server_env_defaults:
        for env in server_env_defaults.values():
            for key in env:
                key_usage[key] = key_usage.get(key, 0) + 1
    for name, command, args, env in stdio_servers:
        resolved_env = resolve_gateway_server_env(
            server_name=name,
            configured_env=env,
            container_env=dict(os.environ),
            key_usage=key_usage,
        )
        transport = StdioTransport(command=command, args=args, env={**os.environ, **resolved_env})
        proxy = FastMCP.as_proxy(transport, name=name)
        gateway.mount(proxy, name)

    print(
        f"[gateway] streamable-http host={host} port={port} servers={len(stdio_servers)}",
        file=sys.stderr,
        flush=True,
    )
    gateway.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
