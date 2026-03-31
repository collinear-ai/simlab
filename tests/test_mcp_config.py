"""Tests for MCP config loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from simlab.mcp_config import build_gateway_config_arena_format
from simlab.mcp_config import build_gateway_container_env
from simlab.mcp_config import build_scoped_gateway_env_var_name
from simlab.mcp_config import get_mcp_command_servers
from simlab.mcp_config import get_mcp_server_urls
from simlab.mcp_config import load_mcp_servers_config
from simlab.mcp_config import load_mcp_servers_from_env_dir
from simlab.mcp_config import resolve_gateway_server_env
from simlab.mcp_config import validate_mcp_server_name_conflicts
from simlab.mcp_config import validate_mcp_servers_config


def test_validate_mcp_servers_accepts_url() -> None:
    validate_mcp_servers_config({"mcpServers": {"notion": {"url": "https://mcp.notion.com/mcp"}}})


def test_validate_mcp_servers_accepts_command() -> None:
    validate_mcp_servers_config(
        {
            "mcpServers": {
                "weather": {
                    "command": "uvx",
                    "args": ["--from", "git+https://example.com/mcp-weather.git", "mcp-weather"],
                    "env": {"API_KEY": ""},
                },
            },
        }
    )


def test_validate_mcp_servers_rejects_invalid_server_name() -> None:
    with pytest.raises(
        ValueError, match="server names may contain only letters, numbers, '_' and '-'"
    ):
        validate_mcp_servers_config(
            {"mcpServers": {"notion prod": {"url": "https://mcp.notion.com/mcp"}}}
        )


def test_validate_mcp_servers_allows_builtin_tool_name_without_env_context() -> None:
    validate_mcp_servers_config({"mcpServers": {"email": {"url": "https://example.com"}}})


def test_validate_mcp_servers_allows_non_catalog_names() -> None:
    validate_mcp_servers_config({"mcpServers": {"email-env": {"url": "https://example.com/mcp"}}})


def test_validate_mcp_server_name_conflicts_rejects_existing_tool_name() -> None:
    with pytest.raises(ValueError, match="conflicts with an existing tool server"):
        validate_mcp_server_name_conflicts(
            {"mcpServers": {"coding": {"url": "https://example.com/mcp"}}},
            existing_tool_names=frozenset({"coding"}),
        )


def test_load_mcp_servers_config_allows_builtin_tool_name_without_env_context(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "mcp.json"
    config_file.write_text(
        json.dumps({"mcpServers": {"coding": {"url": "https://example.com/mcp"}}})
    )

    data = load_mcp_servers_config(config_file)

    assert data["mcpServers"]["coding"]["url"] == "https://example.com/mcp"


def test_validate_mcp_servers_rejects_names_that_collapse_to_same_env_prefix() -> None:
    with pytest.raises(ValueError, match="must remain distinct after gateway env normalization"):
        validate_mcp_servers_config(
            {
                "mcpServers": {
                    "foo-bar": {"command": "uvx"},
                    "foo_bar": {"command": "uvx"},
                }
            }
        )


def test_validate_mcp_servers_rejects_invalid_env_key() -> None:
    with pytest.raises(ValueError, match="env keys must be valid environment variable names"):
        validate_mcp_servers_config(
            {"mcpServers": {"weather": {"command": "uvx", "env": {"api-key": "x"}}}}
        )


def test_validate_mcp_servers_rejects_missing_mcp_servers() -> None:
    with pytest.raises(ValueError, match="mcpServers"):
        validate_mcp_servers_config({})


def test_validate_mcp_servers_rejects_both_url_and_command() -> None:
    with pytest.raises(ValueError, match=r"use either 'url' or 'command', not both"):
        validate_mcp_servers_config(
            {
                "mcpServers": {
                    "x": {"url": "http://a", "command": "npx"},
                },
            }
        )


def test_validate_mcp_servers_rejects_neither_url_nor_command() -> None:
    with pytest.raises(ValueError, match="must have"):
        validate_mcp_servers_config({"mcpServers": {"x": {}}})


def test_load_mcp_servers_config(tmp_path: Path) -> None:
    config_file = tmp_path / "mcp.json"
    config_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "notion": {"url": "https://mcp.notion.com/mcp"},
                    "weather": {"command": "uvx", "args": ["mcp-weather"], "env": {}},
                },
            }
        )
    )
    data = load_mcp_servers_config(config_file)
    assert "mcpServers" in data
    assert "notion" in data["mcpServers"]
    assert "weather" in data["mcpServers"]


def test_get_mcp_server_urls() -> None:
    config = {"mcpServers": {"notion": {"url": "https://mcp.notion.com/mcp"}}}
    assert get_mcp_server_urls(config) == {"notion": "https://mcp.notion.com/mcp"}


def test_get_mcp_command_servers() -> None:
    config = {
        "mcpServers": {
            "weather": {"command": "uvx", "args": ["mcp-weather"], "env": {"KEY": "x"}},
        },
    }
    servers = get_mcp_command_servers(config)
    assert list(servers) == ["weather"]
    assert servers["weather"]["command"] == "uvx"
    assert servers["weather"]["args"] == ["mcp-weather"]
    assert servers["weather"]["env"] == {"KEY": "x"}


def test_build_gateway_config_arena_format() -> None:
    command_servers = {
        "weather": {
            "command": "uvx",
            "args": ["mcp-weather"],
            "env": {"API_KEY": "demo-key"},
        },
    }
    out = build_gateway_config_arena_format(command_servers)
    assert out["servers"] == [
        {
            "name": "weather",
            "transport": "stdio",
            "command": "uvx mcp-weather",
            "env": {"API_KEY": "demo-key"},
        },
    ]


def test_build_gateway_container_env_uses_raw_names_for_unique_keys() -> None:
    command_servers = {
        "weather": {"command": "uvx", "env": {"ACCUWEATHER_API_KEY": "demo-key"}},
    }

    gateway_env, env_defaults = build_gateway_container_env(command_servers)

    assert gateway_env == {
        build_scoped_gateway_env_var_name(
            "weather", "ACCUWEATHER_API_KEY"
        ): "${ACCUWEATHER_API_KEY}",
    }
    assert env_defaults == {"ACCUWEATHER_API_KEY": "demo-key"}


def test_build_gateway_container_env_uses_scoped_names_for_shared_keys() -> None:
    command_servers = {
        "weather": {"command": "uvx", "env": {"API_KEY": "weather-key"}},
        "docs": {"command": "uvx", "env": {"API_KEY": "docs-key"}},
    }

    gateway_env, env_defaults = build_gateway_container_env(command_servers)

    weather_var = build_scoped_gateway_env_var_name("weather", "API_KEY")
    docs_var = build_scoped_gateway_env_var_name("docs", "API_KEY")
    assert gateway_env == {
        weather_var: f"${{{weather_var}}}",
        docs_var: f"${{{docs_var}}}",
    }
    assert env_defaults == {
        weather_var: "weather-key",
        docs_var: "docs-key",
    }


def test_resolve_gateway_server_env_prefers_scoped_override_then_raw_then_config() -> None:
    configured_env = {"API_KEY": "placeholder"}
    key_usage = {"API_KEY": 1}

    assert resolve_gateway_server_env(
        server_name="weather",
        configured_env=configured_env,
        container_env={build_scoped_gateway_env_var_name("weather", "API_KEY"): "scoped-secret"},
        key_usage=key_usage,
    ) == {"API_KEY": "scoped-secret"}
    assert resolve_gateway_server_env(
        server_name="weather",
        configured_env=configured_env,
        container_env={"API_KEY": "raw-secret"},
        key_usage=key_usage,
    ) == {"API_KEY": "raw-secret"}
    assert resolve_gateway_server_env(
        server_name="weather",
        configured_env=configured_env,
        container_env={},
        key_usage=key_usage,
    ) == {"API_KEY": "placeholder"}


def test_resolve_gateway_server_env_does_not_use_raw_override_for_shared_key() -> None:
    assert resolve_gateway_server_env(
        server_name="weather",
        configured_env={"API_KEY": "weather-placeholder"},
        container_env={"API_KEY": "shared-secret"},
        key_usage={"API_KEY": 2},
    ) == {"API_KEY": "weather-placeholder"}


def test_load_mcp_servers_from_env_dir_missing_returns_none(tmp_path: Path) -> None:
    assert load_mcp_servers_from_env_dir(tmp_path) is None


def test_load_mcp_servers_from_env_dir_loads_file(tmp_path: Path) -> None:
    (tmp_path / "mcp-servers.json").write_text(
        json.dumps({"mcpServers": {"notion": {"url": "https://example.com/mcp"}}})
    )
    data = load_mcp_servers_from_env_dir(tmp_path)
    assert data is not None
    assert get_mcp_server_urls(data) == {"notion": "https://example.com/mcp"}


def test_load_mcp_servers_from_env_dir_rejects_tool_name_conflict(tmp_path: Path) -> None:
    (tmp_path / "mcp-servers.json").write_text(
        json.dumps({"mcpServers": {"email": {"url": "https://example.com/mcp"}}})
    )

    with pytest.raises(ValueError, match="conflicts with an existing tool server"):
        load_mcp_servers_from_env_dir(tmp_path)


def test_load_mcp_servers_from_env_dir_rejects_env_local_custom_tool_conflict(
    tmp_path: Path,
) -> None:
    custom_tools_dir = tmp_path / "custom-tools"
    custom_tools_dir.mkdir()
    (custom_tools_dir / "harbor-main.yaml").write_text(
        "\n".join(
            [
                "name: harbor-main",
                "display_name: Harbor Main",
                "description: Harbor task runtime",
                "category: custom",
                "tool_server_url: http://localhost:9000",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "mcp-servers.json").write_text(
        json.dumps({"mcpServers": {"harbor-main": {"url": "https://example.com/mcp"}}})
    )

    with pytest.raises(ValueError, match="conflicts with an existing tool server"):
        load_mcp_servers_from_env_dir(tmp_path)
