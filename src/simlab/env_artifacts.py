"""Helpers for generated environment artifact freshness."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml

from simlab.composer.engine import ComposeEngine
from simlab.composer.engine import ComposeOutput
from simlab.composer.engine import EnvConfig
from simlab.composer.engine import write_output
from simlab.env_registry import build_registry
from simlab.env_registry import get_custom_tools_dir
from simlab.mcp_config import MCP_SERVERS_FILENAME

GENERATED_STATE_FILENAME = ".simlab-generated.json"
_STATE_VERSION = 1


def load_env_config(env_dir: Path) -> EnvConfig:
    """Load ``env.yaml`` from an environment directory."""
    config_file = env_dir / "env.yaml"
    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    return EnvConfig(**data)


def _tracked_input_paths(env_dir: Path) -> list[Path]:
    """Return the env inputs that affect generated compose artifacts."""
    paths: list[Path] = []
    env_yaml = env_dir / "env.yaml"
    if env_yaml.is_file():
        paths.append(env_yaml)

    mcp_file = env_dir / MCP_SERVERS_FILENAME
    if mcp_file.is_file():
        paths.append(mcp_file)

    custom_tools_dir = get_custom_tools_dir(env_dir)
    if custom_tools_dir.is_dir():
        paths.extend(sorted(path for path in custom_tools_dir.rglob("*.yaml") if path.is_file()))
    return paths


def _file_hash(path: Path) -> str:
    """Return a stable hash of one tracked file."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def compute_generation_inputs(env_dir: Path) -> dict[str, str]:
    """Return a relative-path -> content-hash manifest for tracked env inputs."""
    return {
        path.relative_to(env_dir).as_posix(): _file_hash(path)
        for path in _tracked_input_paths(env_dir)
    }


def _generation_state_path(env_dir: Path) -> Path:
    return env_dir / GENERATED_STATE_FILENAME


def write_generation_state(env_dir: Path) -> None:
    """Persist the tracked input manifest after successful generation."""
    payload = {
        "version": _STATE_VERSION,
        "inputs": compute_generation_inputs(env_dir),
    }
    _generation_state_path(env_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_generation_state(env_dir: Path) -> dict[str, Any] | None:
    state_file = _generation_state_path(env_dir)
    if not state_file.is_file():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version") != _STATE_VERSION:
        return None
    inputs = data.get("inputs")
    if not isinstance(inputs, dict):
        return None
    return data


def _legacy_generation_is_stale(env_dir: Path) -> bool:
    """Fallback freshness heuristic for envs created before generation state existed."""
    tracked_inputs = _tracked_input_paths(env_dir)
    if not tracked_inputs:
        return False

    output_paths = [env_dir / "docker-compose.yml", env_dir / ".env"]
    existing_outputs = [path for path in output_paths if path.is_file()]
    if not existing_outputs:
        return True

    latest_input_mtime = max(path.stat().st_mtime for path in tracked_inputs)
    latest_output_mtime = max(path.stat().st_mtime for path in existing_outputs)
    return latest_input_mtime > latest_output_mtime


def detect_generation_drift(env_dir: Path) -> tuple[bool, list[str]]:
    """Return whether generated env outputs are stale and why."""
    missing_outputs = _missing_generated_outputs(env_dir)
    current_inputs = compute_generation_inputs(env_dir)
    stored_state = _read_generation_state(env_dir)
    if stored_state is None:
        if _legacy_generation_is_stale(env_dir):
            legacy_reasons = ["Generated env files are older than env inputs."]
            legacy_reasons = [*missing_outputs, *legacy_reasons]
            return True, legacy_reasons
        if missing_outputs:
            return True, missing_outputs
        return False, []

    stored_inputs = stored_state["inputs"]
    reasons: list[str] = []
    all_keys = sorted(set(current_inputs) | set(stored_inputs))
    for key in all_keys:
        current_hash = current_inputs.get(key)
        stored_hash = stored_inputs.get(key)
        if current_hash == stored_hash:
            continue
        if stored_hash is None:
            reasons.append(f"Added input: {key}")
        elif current_hash is None:
            reasons.append(f"Removed input: {key}")
        else:
            reasons.append(f"Changed input: {key}")
    reasons = [*missing_outputs, *reasons]
    return bool(reasons), reasons


def _missing_generated_outputs(env_dir: Path) -> list[str]:
    """Return reasons when generated env outputs are missing or incomplete."""
    reasons: list[str] = []
    compose_file = env_dir / "docker-compose.yml"
    env_file = env_dir / ".env"

    if not compose_file.is_file():
        reasons.append("Missing docker-compose.yml.")
        return reasons

    try:
        compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        reasons.append("docker-compose.yml could not be read as YAML.")
        return reasons

    services = compose.get("services") if isinstance(compose, dict) else None
    if not isinstance(services, dict):
        reasons.append("docker-compose.yml has no services section.")
        return reasons

    if services and not env_file.is_file():
        reasons.append("Missing .env.")

    for service_name, service in services.items():
        if not isinstance(service, dict):
            continue
        build = service.get("build")
        if build is None:
            continue

        build_context = build.get("context") if isinstance(build, dict) else build
        if not isinstance(build_context, str) or not build_context.strip():
            continue

        context_path = Path(build_context.strip())
        if not context_path.is_absolute():
            context_path = env_dir / context_path
        if not context_path.exists():
            reasons.append(f"Missing build context for service '{service_name}': {build_context}.")
            continue
        if not context_path.is_dir():
            reasons.append(
                f"Build context for service '{service_name}' is not a directory: {build_context}."
            )
            continue

        if str(service_name) == ComposeEngine.MCP_GATEWAY_SERVICE_NAME:
            missing_gateway_files = [
                filename
                for filename in ("Dockerfile", "requirements.txt", "run_gateway.py")
                if not (context_path / filename).is_file()
            ]
            reasons.extend(
                [
                    f"Missing gateway build context file: {context_path / filename}."
                    for filename in missing_gateway_files
                ]
            )
            gateway_config = env_dir / ComposeEngine.MCP_GATEWAY_CONFIG_FILENAME
            if not gateway_config.is_file():
                reasons.append(f"Missing {ComposeEngine.MCP_GATEWAY_CONFIG_FILENAME}.")

    return reasons


def regenerate_env_artifacts(env_dir: Path) -> ComposeOutput:
    """Regenerate compose artifacts from ``env.yaml`` plus env-local custom tools."""
    config = load_env_config(env_dir)
    registry = build_registry(env_dir=env_dir)
    output = ComposeEngine(registry).compose(
        config, config_dir=env_dir, env_dir=env_dir, output_dir=env_dir
    )
    write_output(output, env_dir)
    write_generation_state(env_dir)
    return output


def ensure_env_artifacts_current(env_dir: Path, *, action_label: str) -> None:
    """Prompt to regenerate stale env outputs or fail in non-interactive flows."""
    is_stale, reasons = detect_generation_drift(env_dir)
    if not is_stale:
        return

    click.echo(
        click.style(
            f"Generated setup files are stale before {action_label}.",
            fg="yellow",
        ),
        err=True,
    )
    for reason in reasons[:5]:
        click.echo(f"  - {reason}", err=True)
    if len(reasons) > 5:
        click.echo(f"  - ... and {len(reasons) - 5} more", err=True)

    if sys.stdin.isatty() and sys.stdout.isatty():
        if click.confirm("Regenerate generated setup files now?", default=True):
            regenerate_env_artifacts(env_dir)
            click.echo(click.style("Setup files regenerated.", fg="green"))
            return
        click.echo(click.style("Cannot continue with stale generated files.", fg="red"), err=True)
        raise SystemExit(1)

    click.echo(
        click.style(
            "Run an interactive command to allow regeneration, or rerun "
            f"`simlab env init {env_dir.name} --force` to refresh generated files.",
            fg="red",
        ),
        err=True,
    )
    raise SystemExit(1)
