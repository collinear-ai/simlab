"""Helpers for constructing Daytona SDK clients lazily."""

from __future__ import annotations

import importlib
from typing import Any

import click

from simlab.config import resolve_daytona_api_key


def get_daytona_client(daytona_api_key: str | None = None) -> Any:  # noqa: ANN401
    """Construct a Daytona SDK client only when Daytona mode is requested."""
    try:
        daytona_mod = importlib.import_module("daytona")
        daytona_cls = daytona_mod.Daytona
        daytona_config_cls = daytona_mod.DaytonaConfig
    except Exception as exc:  # pragma: no cover - import path depends on env
        click.echo(
            click.style(
                f"Daytona SDK is required for Daytona mode: {exc}",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1) from exc

    api_key = resolve_daytona_api_key(daytona_api_key)
    if not api_key:
        click.echo(
            click.style(
                "Daytona API key is required for Daytona mode via --daytona-api-key, "
                "config, SIMLAB_DAYTONA_API_KEY, or DAYTONA_API_KEY.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)
    return daytona_cls(daytona_config_cls(api_key=api_key))
