from __future__ import annotations

import json

from simlab.gateway.run_gateway import parse_stdio_servers
from simlab.gateway.run_gateway import resolve_gateway_server_env
from simlab.mcp_config import build_scoped_gateway_env_var_name


def test_parse_stdio_servers_preserves_scoped_env() -> None:
    config_json = json.dumps(
        {
            "servers": [
                {
                    "name": "weather",
                    "transport": "stdio",
                    "command": "uvx mcp-weather",
                    "env": {"API_KEY": "weather-key"},
                },
                {
                    "name": "docs",
                    "transport": "stdio",
                    "command": "uvx mcp-docs",
                    "env": {"API_KEY": "docs-key"},
                },
            ]
        }
    )

    assert parse_stdio_servers(config_json) == [
        ("weather", "uvx", ["mcp-weather"], {"API_KEY": "weather-key"}),
        ("docs", "uvx", ["mcp-docs"], {"API_KEY": "docs-key"}),
    ]


def test_resolve_gateway_server_env_uses_scoped_override_for_shared_key() -> None:
    scoped_name = build_scoped_gateway_env_var_name("weather", "API_KEY")

    resolved = resolve_gateway_server_env(
        server_name="weather",
        configured_env={"API_KEY": "placeholder"},
        container_env={scoped_name: "weather-secret", "API_KEY": "shared-secret"},
        key_usage={"API_KEY": 2},
    )

    assert resolved == {"API_KEY": "weather-secret"}


def test_resolve_gateway_server_env_uses_raw_override_for_unique_key() -> None:
    resolved = resolve_gateway_server_env(
        server_name="weather",
        configured_env={"ACCUWEATHER_API_KEY": "placeholder"},
        container_env={"ACCUWEATHER_API_KEY": "real-key"},
        key_usage={"ACCUWEATHER_API_KEY": 1},
    )

    assert resolved == {"ACCUWEATHER_API_KEY": "real-key"}
