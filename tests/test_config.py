"""Tests for global config loading (TOML file + env + CLI overrides)."""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from simlab import config as config_mod
from simlab.config import GlobalConfig
from simlab.config import get_global_config_from_ctx
from simlab.config import load_global_config


def test_load_global_config_no_file_no_env_no_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no config file, no env, and no CLI overrides, all fields are None."""
    no_file = tmp_path / "no-config.toml"
    assert not no_file.exists()
    for key in (
        "SIMLAB_COLLINEAR_API_KEY",
        "COLLINEAR_API_KEY",
        "SIMLAB_CONFIG",
        "SIMLAB_SCENARIO_MANAGER_API_URL",
        "SIMLAB_DAYTONA_API_KEY",
        "DAYTONA_API_KEY",
        "SIMLAB_VERIFIER_MODEL",
        "SIMLAB_VERIFIER_PROVIDER",
        "SIMLAB_VERIFIER_BASE_URL",
        "SIMLAB_VERIFIER_API_KEY",
        "SIMLAB_AGENT_MODEL",
        "SIMLAB_AGENT_PROVIDER",
        "SIMLAB_AGENT_BASE_URL",
        "SIMLAB_AGENT_API_KEY",
        "SIMLAB_ENVIRONMENTS_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = load_global_config(config_file_path=str(no_file), cli_overrides=None)
    assert isinstance(cfg, GlobalConfig)
    assert cfg.collinear_api_key is None
    assert cfg.daytona_api_key is None
    assert cfg.scenario_manager_api_url is None
    assert cfg.verifier_model is None
    assert cfg.agent_api_key is None


def test_load_global_config_from_file_in_tmpdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Config file in tmpdir is loaded and applied."""
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("SIMLAB_DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
collinear_api_key = "file-key"
scenario_manager_api_url = "https://api.example.com"

[daytona]
api_key = "file-daytona"

[verifier]
model = "gpt-4o"
"""
    )
    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)
    assert cfg.collinear_api_key == "file-key"
    assert cfg.daytona_api_key == "file-daytona"
    assert cfg.scenario_manager_api_url == "https://api.example.com"
    assert cfg.verifier_model == "gpt-4o"
    assert cfg.verifier_provider is None
    assert cfg.agent_model is None


def test_load_global_config_partial_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only keys present in the file are set; rest remain None."""
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("SIMLAB_DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    config_file = tmp_path / "partial.toml"
    config_file.write_text('scenario_manager_api_url = "https://api.example.com"\n')
    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)
    assert cfg.collinear_api_key is None
    assert cfg.daytona_api_key is None
    assert cfg.scenario_manager_api_url == "https://api.example.com"


def test_load_global_config_tasks_section(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tasks rollout format is loaded from the tasks config section."""
    monkeypatch.delenv("SIMLAB_TASKS_ROLLOUT_FORMAT", raising=False)
    config_file = tmp_path / "config.toml"
    config_file.write_text('[tasks]\nrollout_format = "ATIF"\n')

    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)

    assert cfg.tasks_rollout_format == "atif"


def test_load_global_config_missing_file_uses_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When config_file_path points to a non-existent file, no file is loaded."""
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("SIMLAB_DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    missing = tmp_path / "does-not-exist.toml"
    assert not missing.exists()
    cfg = load_global_config(config_file_path=str(missing), cli_overrides=None)
    assert cfg.collinear_api_key is None
    assert cfg.daytona_api_key is None


def test_load_global_config_empty_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Empty or invalid TOML file yields no file data (no crash)."""
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    config_file = tmp_path / "empty.toml"
    config_file.write_text("")
    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)
    assert cfg.collinear_api_key is None

    config_file.write_text("not valid toml [[[[")
    cfg2 = load_global_config(config_file_path=str(config_file), cli_overrides=None)
    assert cfg2.collinear_api_key is None


def test_load_global_config_extra_keys_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unknown keys in TOML are ignored (model has extra='ignore')."""
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    config_file = tmp_path / "extra.toml"
    config_file.write_text(
        """
scenario_manager_api_url = "https://api.example.com"
unknown_key = "ignored"
another_unknown = 123
"""
    )
    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)
    assert cfg.scenario_manager_api_url == "https://api.example.com"
    assert not hasattr(cfg, "unknown_key")


def test_load_global_config_env_overlay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Env vars overlay config file."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('collinear_api_key = "file-key"\n[daytona]\napi_key = "file-daytona"\n')
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "env-key")
    monkeypatch.setenv("SIMLAB_DAYTONA_API_KEY", "env-daytona")
    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)
    assert cfg.collinear_api_key == "env-key"
    assert cfg.daytona_api_key == "env-daytona"


def test_load_global_config_simlab_collinear_env_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SIMLAB_COLLINEAR_API_KEY populates the Collinear auth field."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('[daytona]\napi_key = "file-daytona"\n')
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "env-key")

    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)

    assert cfg.collinear_api_key == "env-key"


def test_telemetry_state_path_defaults_to_global_simlab_config_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Telemetry state should default to the shared SimLab config directory."""
    monkeypatch.delenv("SIMLAB_TELEMETRY_STATE_PATH", raising=False)
    monkeypatch.setattr(config_mod, "_DEFAULT_CONFIG_DIR", tmp_path / ".config" / "simlab")

    state_path = config_mod.telemetry_state_path()

    assert state_path == tmp_path / ".config" / "simlab" / "simlab.json"
    assert not state_path.parent.exists()


def test_load_global_config_telemetry_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Telemetry settings load from a dedicated config.toml section."""
    monkeypatch.delenv("SIMLAB_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("SIMLAB_TELEMETRY_STATE_PATH", raising=False)
    config_file = tmp_path / "config.toml"
    telemetry_state_file = tmp_path / "simlab-telemetry.json"
    config_file.write_text(
        f"""
[telemetry]
disabled = true
state_path = "{telemetry_state_file}"
"""
    )

    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)

    assert cfg.telemetry_disabled is True
    assert cfg.telemetry_state_path == str(telemetry_state_file)


def test_load_global_config_daytona_and_agent_sections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Daytona and agent settings load from dedicated config.toml sections."""
    for key in (
        "SIMLAB_DAYTONA_API_KEY",
        "DAYTONA_API_KEY",
        "SIMLAB_AGENT_MODEL",
        "SIMLAB_AGENT_PROVIDER",
        "SIMLAB_AGENT_BASE_URL",
        "SIMLAB_AGENT_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[daytona]
api_key = "section-daytona-key"

[agent]
model = "gpt-4o-mini"
provider = "openai"
base_url = "https://llm.example.com/v1"
api_key = "section-agent-key"
"""
    )

    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)

    assert cfg.daytona_api_key == "section-daytona-key"
    assert cfg.agent_model == "gpt-4o-mini"
    assert cfg.agent_provider == "openai"
    assert cfg.agent_base_url == "https://llm.example.com/v1"
    assert cfg.agent_api_key == "section-agent-key"


def test_load_global_config_verifier_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verifier settings load from a dedicated config.toml section."""
    for key in (
        "SIMLAB_VERIFIER_MODEL",
        "SIMLAB_VERIFIER_PROVIDER",
        "SIMLAB_VERIFIER_BASE_URL",
        "SIMLAB_VERIFIER_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[verifier]
model = "gpt-5"
provider = "openai"
base_url = "https://judge.example.com/v1"
api_key = "section-verifier-key"
"""
    )

    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)

    assert cfg.verifier_model == "gpt-5"
    assert cfg.verifier_provider == "openai"
    assert cfg.verifier_base_url == "https://judge.example.com/v1"
    assert cfg.verifier_api_key == "section-verifier-key"


def test_load_global_config_ignores_flat_sectioned_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sectioned settings must not load from flat config.toml keys."""
    for key in (
        "SIMLAB_DAYTONA_API_KEY",
        "DAYTONA_API_KEY",
        "SIMLAB_AGENT_MODEL",
        "SIMLAB_AGENT_API_KEY",
        "SIMLAB_VERIFIER_MODEL",
        "SIMLAB_DISABLE_TELEMETRY",
    ):
        monkeypatch.delenv(key, raising=False)

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
daytona_api_key = "flat-daytona"
agent_model = "flat-agent-model"
agent_api_key = "flat-agent-key"
verifier_model = "flat-verifier-model"
telemetry_disabled = true
"""
    )

    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)

    assert cfg.daytona_api_key is None
    assert cfg.agent_model is None
    assert cfg.agent_api_key is None
    assert cfg.verifier_model is None
    assert cfg.telemetry_disabled is None


def test_load_global_config_cli_overrides_file_and_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI overrides take precedence over file and env."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('collinear_api_key = "file-key"\n[daytona]\napi_key = "file-daytona"\n')
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "env-key")
    cfg = load_global_config(
        config_file_path=str(config_file),
        cli_overrides={
            "collinear_api_key": "cli-key",
            "daytona_api_key": "cli-daytona-key",
        },
    )
    assert cfg.collinear_api_key == "cli-key"
    assert cfg.daytona_api_key == "cli-daytona-key"


def test_load_global_config_cli_overrides_stripped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI override values are stripped; None and unknown keys are ignored."""
    monkeypatch.delenv("SIMLAB_DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setenv("SIMLAB_CONFIG", str(tmp_path / "missing-config.toml"))
    cfg = load_global_config(
        config_file_path=None,
        cli_overrides={
            "collinear_api_key": "  stripped  ",
            "daytona_api_key": None,
            "unknown_param": "ignored",
        },
    )
    assert cfg.collinear_api_key == "stripped"
    assert cfg.daytona_api_key is None


def test_load_global_config_precedence_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Order: defaults < file < env < CLI."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        'collinear_api_key = "from-file"\n[daytona]\napi_key = "from-file-daytona"\n'
    )
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "from-env")
    monkeypatch.setenv("DAYTONA_API_KEY", "from-env-daytona")
    cfg = load_global_config(
        config_file_path=str(config_file),
        cli_overrides={
            "collinear_api_key": "from-cli",
            "daytona_api_key": "from-cli-daytona",
        },
    )
    assert cfg.collinear_api_key == "from-cli"
    assert cfg.daytona_api_key == "from-cli-daytona"

    cfg_no_cli = load_global_config(config_file_path=str(config_file), cli_overrides=None)
    assert cfg_no_cli.collinear_api_key == "from-env"
    assert cfg_no_cli.daytona_api_key == "from-env-daytona"

    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    cfg_file_only = load_global_config(config_file_path=str(config_file), cli_overrides=None)
    assert cfg_file_only.collinear_api_key == "from-file"
    assert cfg_file_only.daytona_api_key == "from-file-daytona"


def test_load_global_config_daytona_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """SIMLAB_DAYTONA_API_KEY is preferred; DAYTONA_API_KEY is fallback when SIMLAB_ not set."""
    monkeypatch.setenv("SIMLAB_DAYTONA_API_KEY", "simlab-daytona-key")
    monkeypatch.setenv("DAYTONA_API_KEY", "daytona-key")
    cfg = load_global_config(config_file_path=None, cli_overrides=None)
    assert cfg.daytona_api_key == "simlab-daytona-key"
    monkeypatch.delenv("SIMLAB_DAYTONA_API_KEY", raising=False)
    cfg2 = load_global_config(config_file_path=None, cli_overrides=None)
    assert cfg2.daytona_api_key == "daytona-key"


def test_resolve_daytona_api_key_prefers_explicit_then_config_then_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMLAB_DAYTONA_API_KEY", "env-daytona-key")

    assert config_mod.resolve_daytona_api_key("explicit-daytona-key") == "explicit-daytona-key"
    assert (
        config_mod.resolve_daytona_api_key(
            None,
            config=GlobalConfig(daytona_api_key="config-daytona-key"),
        )
        == "config-daytona-key"
    )
    assert config_mod.resolve_daytona_api_key(None) == "env-daytona-key"


def test_resolve_collinear_api_key_config_does_not_fall_back_to_default_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    leaked_config = tmp_path / "leaked.toml"
    leaked_config.write_text('collinear_api_key = "leaked-key"\n')
    monkeypatch.setenv("SIMLAB_CONFIG", str(leaked_config))
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)

    assert config_mod.resolve_collinear_api_key(config=GlobalConfig(collinear_api_key=None)) is None


def test_resolve_daytona_api_key_config_does_not_fall_back_to_default_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    leaked_config = tmp_path / "leaked.toml"
    leaked_config.write_text('[daytona]\napi_key = "leaked-daytona-key"\n')
    monkeypatch.setenv("SIMLAB_CONFIG", str(leaked_config))
    monkeypatch.delenv("SIMLAB_DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)

    assert config_mod.resolve_daytona_api_key(config=GlobalConfig(daytona_api_key=None)) is None


def test_resolve_agent_api_key_falls_back_to_openai_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMLAB_AGENT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback-key")

    assert (
        config_mod.resolve_agent_api_key(
            None,
            provider="openai",
            config=GlobalConfig(agent_api_key=None, agent_provider=None),
        )
        == "openai-fallback-key"
    )


def test_resolve_agent_api_key_does_not_use_openai_env_for_non_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMLAB_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback-key")

    assert (
        config_mod.resolve_agent_api_key(
            None,
            provider="anthropic",
            config=GlobalConfig(agent_api_key=None, agent_provider=None),
        )
        is None
    )


def test_resolve_agent_api_key_config_does_not_fall_back_to_default_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    leaked_config = tmp_path / "leaked.toml"
    leaked_config.write_text(
        """
[agent]
provider = "anthropic"
api_key = "leaked-agent-key"
"""
    )
    monkeypatch.setenv("SIMLAB_CONFIG", str(leaked_config))
    monkeypatch.delenv("SIMLAB_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert (
        config_mod.resolve_agent_api_key(
            None,
            config=GlobalConfig(agent_api_key=None, agent_provider=None),
        )
        is None
    )


def test_resolve_scenario_manager_api_url_config_does_not_fall_back_to_default_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    leaked_config = tmp_path / "leaked.toml"
    leaked_config.write_text('scenario_manager_api_url = "https://leaked.example.com"\n')
    monkeypatch.setenv("SIMLAB_CONFIG", str(leaked_config))
    monkeypatch.delenv("SIMLAB_SCENARIO_MANAGER_API_URL", raising=False)

    assert (
        config_mod.resolve_scenario_manager_api_url(
            config=GlobalConfig(scenario_manager_api_url=None),
        )
        == config_mod.DEFAULT_SCENARIO_MANAGER_API_URL
    )


def test_get_global_config_from_ctx_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_global_config_from_ctx(None) returns default config."""
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("SIMLAB_CONFIG", raising=False)
    monkeypatch.setattr(
        config_mod, "_config_file_path", lambda override=None, must_exist=True: None
    )
    cfg = get_global_config_from_ctx(None)
    assert isinstance(cfg, GlobalConfig)
    assert cfg.collinear_api_key is None


def test_get_global_config_from_ctx_not_click_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_global_config_from_ctx with non-Context returns default config."""
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("SIMLAB_CONFIG", raising=False)
    monkeypatch.setattr(
        config_mod, "_config_file_path", lambda override=None, must_exist=True: None
    )
    cfg = get_global_config_from_ctx("not a context")  # type: ignore[arg-type]
    assert isinstance(cfg, GlobalConfig)
    assert cfg.collinear_api_key is None


def test_get_global_config_from_ctx_uses_root_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """get_global_config_from_ctx uses root context params as CLI overrides."""
    monkeypatch.delenv("SIMLAB_SCENARIO_MANAGER_API_URL", raising=False)
    config_file = tmp_path / "config.toml"
    config_file.write_text('scenario_manager_api_url = "https://api.example.com"\n')
    root = click.Context(click.Command("root"))
    root.params = {"config_file": str(config_file), "collinear_api_key": "cli-key"}
    cfg = get_global_config_from_ctx(root)
    assert cfg.collinear_api_key == "cli-key"
    assert cfg.scenario_manager_api_url == "https://api.example.com"


def test_get_global_config_from_ctx_walks_to_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """get_global_config_from_ctx walks up to root context."""
    monkeypatch.delenv("SIMLAB_SCENARIO_MANAGER_API_URL", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    config_file = tmp_path / "config.toml"
    config_file.write_text('[daytona]\napi_key = "file-only"\n')
    root = click.Context(click.Command("root"))
    root.params = {"config_file": str(config_file)}
    child = click.Context(click.Command("child"), parent=root)
    child.params = {}
    cfg = get_global_config_from_ctx(child)
    assert cfg.daytona_api_key == "file-only"


def test_load_global_config_reads_collinear_api_key_from_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """collinear_api_key in config.toml should load for Collinear auth."""
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    config_file = tmp_path / "config.toml"
    config_file.write_text('collinear_api_key = "file-key"\n', encoding="utf-8")

    cfg = load_global_config(config_file_path=str(config_file), cli_overrides=None)

    assert cfg.collinear_api_key == "file-key"
