"""CLI commands for authentication."""

from __future__ import annotations

import os
from pathlib import Path

import click
import requests
import tomli_w

from simlab.config import SIMLAB_COLLINEAR_API_KEY_ENV_VARS
from simlab.config import _config_file_path
from simlab.config import _read_toml
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_scenario_manager_api_url


def _mask_key(key: str) -> str:
    """Mask an API key for display, e.g. 'col_89a...XBG'."""
    if len(key) <= 10:
        return key[:3] + "..."
    return key[:7] + "..." + key[-3:]


def _verify_key_with_server(api_key: str, api_url: str) -> bool | None:
    """Validate an API key against the server's /v1/auth/verify endpoint.

    Returns True (valid), False (rejected), or None (server unreachable).
    """
    try:
        resp = requests.get(
            f"{api_url}/v1/auth/verify",
            headers={"API-Key": api_key, "Accept": "application/json"},
            timeout=10,
        )
    except requests.exceptions.RequestException:
        return None
    else:
        return resp.status_code != 401


@click.group("auth")
def auth() -> None:
    """Manage authentication credentials."""


@auth.command()
@click.pass_context
def login(ctx: click.Context) -> None:
    """Save a Collinear API key to the config file."""
    click.echo()
    click.echo("  Get your API key at https://platform.collinear.ai")
    click.echo("  (Developer Resources → API Keys)")
    click.echo()

    api_key = click.prompt("  API key", hide_input=True).strip()
    if not api_key:
        click.secho("  No API key provided.", fg="red")
        raise SystemExit(1)

    # Resolve config file path (respects --config-file global option).
    root = ctx
    while root.parent is not None:
        root = root.parent
    config_file_override = (root.params or {}).get("config_file")
    config_path = _config_file_path(override=config_file_override, must_exist=False)
    if config_path is None:
        config_path = Path.home() / ".config" / "simlab" / "config.toml"

    # Validate key against the server before saving.
    api_url = resolve_scenario_manager_api_url(base_url=None)
    verified = _verify_key_with_server(api_key, api_url)

    if verified is False:
        click.echo()
        click.secho("  Invalid API key.", fg="red")
        click.echo("  Check your key at https://platform.collinear.ai")
        click.echo("  (Developer Resources → API Keys)")
        click.echo()
        raise SystemExit(1)

    # Read existing config, merge collinear_api_key, write back.
    existing = _read_toml(config_path) if config_path.is_file() else {}
    existing["collinear_api_key"] = api_key
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(tomli_w.dumps(existing), encoding="utf-8")

    click.echo()
    click.secho("  Authenticated!", fg="green", bold=True)
    click.echo(f"  Key:    {_mask_key(api_key)}")
    click.echo(f"  Saved:  {config_path}")
    if verified is None:
        click.secho(
            f"  Note:   Could not reach {api_url} to verify — key saved but not validated.",
            fg="yellow",
        )
    click.echo()


@auth.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current authentication status."""
    cfg = get_global_config_from_ctx(ctx)
    api_url = resolve_scenario_manager_api_url(base_url=cfg.scenario_manager_api_url)

    # Resolve config file path for display.
    root = ctx
    while root.parent is not None:
        root = root.parent
    config_file_override = (root.params or {}).get("config_file")
    config_path = _config_file_path(override=config_file_override, must_exist=False)

    click.echo()

    # Determine key source.
    # Check env vars directly to distinguish --collinear-api-key flag from env var.
    root_params = root.params or {}
    param_key = (root_params.get("collinear_api_key") or "").strip()

    env_key = ""
    env_source = ""
    for ev in SIMLAB_COLLINEAR_API_KEY_ENV_VARS:
        val = os.environ.get(ev, "").strip()
        if val:
            env_key = val
            env_source = ev
            break

    file_key = ""
    if config_path and config_path.is_file():
        file_data = _read_toml(config_path)
        raw = file_data.get("collinear_api_key")
        file_key = (raw or "").strip() if isinstance(raw, str) else ""

    # --collinear-api-key flag: param_key is set but differs from env var value.
    if param_key and param_key != env_key:
        source = "--collinear-api-key flag"
        active_key = param_key
    elif env_key:
        source = f"env var {env_source}"
        active_key = env_key
    elif file_key:
        source = f"config file ({config_path})"
        active_key = file_key
    else:
        click.secho("  Not authenticated.", fg="yellow")
        click.echo()
        click.echo("  Get your API key at https://platform.collinear.ai")
        click.echo("  (Developer Resources → API Keys)")
        click.echo()
        click.echo("  Then run: simlab auth login")
        click.echo()
        return

    # Verify the key against the server.
    verified = _verify_key_with_server(active_key, api_url)

    if verified is True:
        click.secho("  Authenticated", fg="green")
    elif verified is False:
        click.secho("  Invalid API key", fg="red")
    else:
        click.secho("  Key configured (server unreachable)", fg="yellow")

    click.echo(f"  Key:     {_mask_key(active_key)}")
    click.echo(f"  Source:  {source}")
    click.echo(f"  API URL: {api_url}")
    click.echo()
