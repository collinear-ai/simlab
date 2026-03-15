"""Centralized SimLab configuration: TOML file + env vars + CLI overrides."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any
from typing import Protocol

import click
import yaml
from pydantic import BaseModel
from pydantic import Field

DEFAULT_SCENARIO_MANAGER_API_URL = "https://rl-gym-api.collinear.ai"

SIMLAB_COLLINEAR_API_KEY_ENV_VARS = ("SIMLAB_COLLINEAR_API_KEY",)
SIMLAB_DAYTONA_API_KEY_ENV_VARS = ("SIMLAB_DAYTONA_API_KEY", "DAYTONA_API_KEY")
SIMLAB_SCENARIO_MANAGER_API_URL_ENV_VARS = ("SIMLAB_SCENARIO_MANAGER_API_URL",)
SIMLAB_TELEMETRY_DISABLE_ENV_VARS = ("SIMLAB_DISABLE_TELEMETRY",)
SIMLAB_TELEMETRY_STATE_PATH_ENV_VARS = ("SIMLAB_TELEMETRY_STATE_PATH",)

_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "simlab"
_DEFAULT_CONFIG_NAME = "config.toml"
_ENV_CONFIG_PATH = "SIMLAB_CONFIG"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

_ENV_MAP: dict[str, list[str]] = {
    "collinear_api_key": list(SIMLAB_COLLINEAR_API_KEY_ENV_VARS),
    "daytona_api_key": list(SIMLAB_DAYTONA_API_KEY_ENV_VARS),
    "scenario_manager_api_url": list(SIMLAB_SCENARIO_MANAGER_API_URL_ENV_VARS),
    "environments_dir": ["SIMLAB_ENVIRONMENTS_DIR"],
    "telemetry_disabled": list(SIMLAB_TELEMETRY_DISABLE_ENV_VARS),
    "telemetry_state_path": list(SIMLAB_TELEMETRY_STATE_PATH_ENV_VARS),
    "verifier_model": ["SIMLAB_VERIFIER_MODEL"],
    "verifier_provider": ["SIMLAB_VERIFIER_PROVIDER"],
    "verifier_base_url": ["SIMLAB_VERIFIER_BASE_URL"],
    "verifier_api_key": ["SIMLAB_VERIFIER_API_KEY"],
    "agent_model": ["SIMLAB_AGENT_MODEL"],
    "agent_provider": ["SIMLAB_AGENT_PROVIDER"],
    "agent_base_url": ["SIMLAB_AGENT_BASE_URL"],
    "agent_api_key": ["SIMLAB_AGENT_API_KEY"],
}

_TOP_LEVEL_FILE_FIELDS = {
    "collinear_api_key",
    "scenario_manager_api_url",
    "environments_dir",
}

_SECTION_FIELD_MAP: dict[str, dict[str, str]] = {
    "daytona": {
        "api_key": "daytona_api_key",
    },
    "agent": {
        "model": "agent_model",
        "provider": "agent_provider",
        "base_url": "agent_base_url",
        "api_key": "agent_api_key",
    },
    "verifier": {
        "model": "verifier_model",
        "provider": "verifier_provider",
        "base_url": "verifier_base_url",
        "api_key": "verifier_api_key",
    },
    "telemetry": {
        "disabled": "telemetry_disabled",
        "state_path": "telemetry_state_path",
    },
}


class ScenarioManagerConfigLike(Protocol):
    """Config shape accepted for Scenario Manager URL resolution."""

    scenario_manager_api_url: str | None


class CollinearApiKeyConfigLike(Protocol):
    """Config shape accepted for Collinear API key resolution."""

    collinear_api_key: str | None


class DaytonaApiKeyConfigLike(Protocol):
    """Config shape accepted for Daytona API key resolution."""

    daytona_api_key: str | None


class TelemetryConfigLike(Protocol):
    """Config shape accepted for telemetry settings resolution."""

    telemetry_disabled: bool | None
    telemetry_state_path: str | None


class AgentRuntimeConfigLike(Protocol):
    """Config shape accepted for agent runtime setting resolution."""

    agent_provider: str | None
    agent_api_key: str | None


class GlobalConfig(BaseModel):
    """Global CLI configuration (file + env + CLI). All fields optional."""

    model_config = {"extra": "ignore"}

    collinear_api_key: str | None = Field(
        default=None,
        description="Collinear API key for rl-gym-api",
    )
    daytona_api_key: str | None = Field(default=None, description="Daytona API key")
    scenario_manager_api_url: str | None = Field(
        default=None, description="Scenario Manager API base URL"
    )
    environments_dir: str | None = Field(
        default=None,
        description="Root directory for environments (default: <cwd>/environments)",
    )
    telemetry_disabled: bool | None = Field(
        default=None,
        description="Disable CLI telemetry when true",
    )
    telemetry_state_path: str | None = Field(
        default=None,
        description="Path for persisted CLI telemetry state",
    )
    verifier_model: str | None = None
    verifier_provider: str | None = None
    verifier_base_url: str | None = None
    verifier_api_key: str | None = None
    agent_model: str | None = None
    agent_provider: str | None = None
    agent_base_url: str | None = None
    agent_api_key: str | None = None


def env_var_list(env_vars: tuple[str, ...] | list[str]) -> str:
    """Return a comma-separated list of environment variable names."""
    return ", ".join(env_vars)


def _normalize_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    return None


def _normalize_loaded_value(field: str, value: object) -> object:
    if field == "telemetry_disabled":
        normalized = _normalize_bool(value)
        if normalized is not None:
            return normalized
    if isinstance(value, str):
        return value.strip() or None
    return value


def _first_env_value(env_vars: tuple[str, ...] | list[str]) -> str | None:
    for env_var in env_vars:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    return None


def _config_file_path(override: str | None = None, must_exist: bool = True) -> Path | None:
    """Path to global config file. Override from CLI or SIMLAB_CONFIG."""
    override_path = _normalize_string(override)
    if override_path:
        path = Path(override_path)
        return path if (not must_exist or path.is_file()) else None

    env_path = _normalize_string(os.environ.get(_ENV_CONFIG_PATH))
    if env_path:
        path = Path(env_path)
        return path if (not must_exist or path.is_file()) else None

    default = _DEFAULT_CONFIG_DIR / _DEFAULT_CONFIG_NAME
    return default if (not must_exist or default.is_file()) else None


def _read_toml(path: Path) -> dict[str, Any]:
    """Load TOML into a dict. Returns {} on missing/empty/invalid."""
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _flatten_file_data(file_data: dict[str, Any]) -> dict[str, Any]:
    """Flatten supported config.toml sections into GlobalConfig field names."""
    flattened: dict[str, Any] = {}

    for key in _TOP_LEVEL_FILE_FIELDS:
        if key in file_data and file_data[key] is not None:
            flattened[key] = _normalize_loaded_value(key, file_data[key])

    for section, field_map in _SECTION_FIELD_MAP.items():
        section_data = file_data.get(section)
        if not isinstance(section_data, dict):
            continue
        for source_key, target_key in field_map.items():
            if source_key in section_data and section_data[source_key] is not None:
                flattened[target_key] = _normalize_loaded_value(
                    target_key,
                    section_data[source_key],
                )

    return flattened


def _env_overlay() -> dict[str, Any]:
    """Build overlay dict from environment variables."""
    overlay: dict[str, Any] = {}
    for field, env_vars in _ENV_MAP.items():
        value = _first_env_value(env_vars)
        if value is not None:
            overlay[field] = _normalize_loaded_value(field, value)
    return overlay


def _env_value(field: str) -> object | None:
    """Return a normalized value for a single config field from env vars only."""
    env_vars = _ENV_MAP.get(field)
    if not env_vars:
        return None
    value = _first_env_value(env_vars)
    if value is None:
        return None
    return _normalize_loaded_value(field, value)


def _normalize_overrides(cli_overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only keys that exist on GlobalConfig and normalize values."""
    if not cli_overrides:
        return {}

    allowed = set(GlobalConfig.model_fields)
    normalized: dict[str, Any] = {}
    for key, value in cli_overrides.items():
        if key not in allowed or value is None:
            continue
        final = _normalize_loaded_value(key, value)
        if final is not None:
            normalized[key] = final
    return normalized


def load_global_config(
    config_file_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> GlobalConfig:
    """Load global config: defaults < config file < env < CLI overrides."""
    data: dict[str, Any] = {}

    path = _config_file_path(override=str(config_file_path) if config_file_path else None)
    if path is not None:
        data.update(_flatten_file_data(_read_toml(path)))

    data.update(_env_overlay())
    data.update(_normalize_overrides(cli_overrides))
    return GlobalConfig.model_validate(data)


def get_global_config_from_ctx(ctx: click.Context | None) -> GlobalConfig:
    """Load global config using root CLI params (for use in subcommands)."""
    if ctx is None or not isinstance(ctx, click.Context):
        return load_global_config()

    root: click.Context = ctx
    while getattr(root, "parent", None) is not None:
        parent = root.parent
        if parent is None:
            break
        root = parent

    params = getattr(root, "params", {}) or {}
    obj = getattr(root, "obj", None)
    cli_overrides = params
    if isinstance(obj, dict):
        stored_overrides = obj.get("global_config_overrides")
        if isinstance(stored_overrides, dict):
            cli_overrides = stored_overrides
    return load_global_config(
        config_file_path=cli_overrides.get("config_file"),
        cli_overrides=cli_overrides,
    )


def _global_config(
    ctx: click.Context | None = None,
    config: GlobalConfig | None = None,
) -> GlobalConfig:
    if config is not None:
        return config
    if ctx is not None:
        return get_global_config_from_ctx(ctx)
    return load_global_config()


def resolve_collinear_api_key(
    explicit_api_key: str | None = None,
    *,
    ctx: click.Context | None = None,
    config: CollinearApiKeyConfigLike | None = None,
) -> str | None:
    """Resolve the global Collinear API key used for Scenario Manager APIs."""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()
    if config is not None:
        config_api_key = _normalize_string(getattr(config, "collinear_api_key", None))
        if config_api_key:
            return config_api_key
        return _normalize_string(_env_value("collinear_api_key"))
    return _global_config(ctx=ctx).collinear_api_key


def resolve_daytona_api_key(
    explicit_api_key: str | None = None,
    *,
    ctx: click.Context | None = None,
    config: DaytonaApiKeyConfigLike | None = None,
) -> str | None:
    """Resolve the Daytona API key."""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()
    if config is not None:
        config_api_key = _normalize_string(getattr(config, "daytona_api_key", None))
        if config_api_key:
            return config_api_key
        return _normalize_string(_env_value("daytona_api_key"))
    return _global_config(ctx=ctx).daytona_api_key


def resolve_agent_api_key(
    explicit_api_key: str | None = None,
    *,
    provider: str | None = None,
    ctx: click.Context | None = None,
    config: AgentRuntimeConfigLike | None = None,
) -> str | None:
    """Resolve the reference-agent API key, including provider-specific fallback."""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    global_cfg: AgentRuntimeConfigLike | None = None
    if config is not None:
        config_api_key = _normalize_string(getattr(config, "agent_api_key", None))
        if config_api_key:
            return config_api_key
        env_api_key = _normalize_string(_env_value("agent_api_key"))
        if env_api_key:
            return env_api_key
    else:
        global_cfg = _global_config(ctx=ctx)
        global_api_key = _normalize_string(global_cfg.agent_api_key)
        if global_api_key:
            return global_api_key

    resolved_provider = _normalize_string(provider)
    if resolved_provider is None and config is not None:
        resolved_provider = _normalize_string(getattr(config, "agent_provider", None))
    if resolved_provider is None and global_cfg is not None:
        resolved_provider = _normalize_string(global_cfg.agent_provider)
    if (resolved_provider or "openai") == "openai":
        return _normalize_string(os.environ.get("OPENAI_API_KEY"))

    return None


def resolve_scenario_manager_api_url(
    config_path: Path | None = None,
    config: ScenarioManagerConfigLike | None = None,
    *,
    base_url: str | None = None,
    ctx: click.Context | None = None,
) -> str:
    """Resolve the Scenario Manager API base URL from CLI, env config, global config, or default."""
    if base_url and base_url.strip():
        return base_url.strip().rstrip("/")

    if config is not None:
        config_url = _normalize_string(getattr(config, "scenario_manager_api_url", None))
        if config_url:
            return config_url.rstrip("/")

    if config_path is not None and config_path.is_file():
        raw = config_path.read_text()
        if config_path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(raw) or {}
        elif config_path.suffix == ".json":
            data = json.loads(raw)
        else:
            data = {}
        config_file_url = _normalize_string(data.get("scenario_manager_api_url"))
        if config_file_url:
            return config_file_url.rstrip("/")

    if config is not None or config_path is not None:
        env_url = _normalize_string(_env_value("scenario_manager_api_url"))
        if env_url:
            return env_url.rstrip("/")
        return DEFAULT_SCENARIO_MANAGER_API_URL.rstrip("/")

    global_url = _global_config(ctx=ctx).scenario_manager_api_url
    if global_url:
        return global_url.rstrip("/")
    return DEFAULT_SCENARIO_MANAGER_API_URL.rstrip("/")


def telemetry_disabled(
    ctx: click.Context | None = None,
    config: TelemetryConfigLike | None = None,
) -> bool:
    """Return whether CLI telemetry is disabled by global configuration."""
    if config is not None and getattr(config, "telemetry_disabled", None) is not None:
        return bool(config.telemetry_disabled)
    return bool(_global_config(ctx=ctx).telemetry_disabled)


def telemetry_state_path(
    ctx: click.Context | None = None,
    config: TelemetryConfigLike | None = None,
) -> Path:
    """Return the path for persisted CLI telemetry state."""
    raw_path: str | None = None
    if config is not None:
        raw_path = _normalize_string(getattr(config, "telemetry_state_path", None))
    if raw_path is None:
        raw_path = _global_config(ctx=ctx).telemetry_state_path

    if raw_path:
        return Path(raw_path)

    default = Path.cwd() / "simlab" / "simlab.json"
    default.parent.mkdir(parents=True, exist_ok=True)
    return default


def get_environments_dir(
    ctx: click.Context | None = None,
    base_path: Path | None = None,
) -> Path:
    """Return the root directory for environments."""
    root = base_path or Path.cwd()
    raw = _normalize_string(_global_config(ctx=ctx).environments_dir)
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (root / path).resolve()
    return root / "environments"


def get_env_dir(
    env_name: str,
    ctx: click.Context | None = None,
    base_path: Path | None = None,
) -> Path:
    """Return the directory for the given environment name."""
    return get_environments_dir(ctx=ctx, base_path=base_path) / env_name


def resolve_env_dir(
    env_name: str,
    ctx: click.Context | None = None,
    base_path: Path | None = None,
    *,
    require_env_yaml: bool = True,
) -> Path:
    """Resolve env directory and validate it exists; on failure exit with a clear message."""
    env_dir = get_env_dir(env_name, ctx=ctx, base_path=base_path)
    if not env_dir.is_dir():
        _exit_env_not_found(env_name, env_dir, ctx, base_path)
    if require_env_yaml and not (env_dir / "env.yaml").is_file():
        _exit_env_not_found(env_name, env_dir, ctx, base_path)
    return env_dir


def _exit_env_not_found(
    env_name: str,
    env_dir: Path,
    ctx: click.Context | None,
    base_path: Path | None,
) -> None:
    """Print env-not-found message and exit."""
    roots_dir = get_environments_dir(ctx=ctx, base_path=base_path)
    click.echo(click.style(f"Environment '{env_name}' not found.", fg="red"), err=True)
    click.echo(f"  Looked at: {env_dir}", err=True)
    if not (env_dir / "env.yaml").is_file() and env_dir.is_dir():
        click.echo(click.style("  env.yaml is missing in that directory.", fg="yellow"), err=True)
    click.echo(
        "  Check the name for typos. If you changed directory, set the environments "
        "root with --environments-dir or SIMLAB_ENVIRONMENTS_DIR.",
        err=True,
    )
    if roots_dir.is_dir():
        try:
            subdirs = [d.name for d in roots_dir.iterdir() if d.is_dir()]
            if subdirs:
                click.echo(
                    f"  Available environments: {', '.join(sorted(subdirs)[:10])}"
                    + (" ..." if len(subdirs) > 10 else ""),
                    err=True,
                )
        except OSError:
            pass
    raise SystemExit(1)
