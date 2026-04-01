"""Scenario template lookup helpers.

The CLI templates commands are presentation only. This module owns the
Scenario Manager API wiring needed to fetch template metadata.
"""

from __future__ import annotations

from pathlib import Path

import click

from simlab.api.client import ScenarioManagerApiError
from simlab.api.client import ScenarioManagerClient
from simlab.api.schemas import ScenarioSummary
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_env_dir
from simlab.config import resolve_scenario_manager_api_url


def build_scenario_manager_client(
    ctx: click.Context,
    *,
    env_name: str | None = None,
) -> ScenarioManagerClient:
    """Return a configured Scenario Manager API client."""
    global_cfg = get_global_config_from_ctx(ctx)
    config_path: Path | None = None
    if env_name:
        config_path = resolve_env_dir(env_name, ctx) / "env.yaml"
    base_url = resolve_scenario_manager_api_url(
        config_path=config_path,
        config=global_cfg,
    )
    api_key = resolve_collinear_api_key(config=global_cfg)
    return ScenarioManagerClient(base_url=base_url, api_key=api_key)


def list_templates(
    ctx: click.Context,
    *,
    env_name: str | None = None,
    include_hidden: bool = False,
) -> list[ScenarioSummary]:
    """Return templates from the Scenario Manager API."""
    sm_client = build_scenario_manager_client(ctx, env_name=env_name)
    return sm_client.list_scenarios(include_hidden=include_hidden)


def get_template(
    ctx: click.Context,
    name: str,
    *,
    env_name: str | None = None,
) -> ScenarioSummary:
    """Return one template summary resolved by name."""
    sm_client = build_scenario_manager_client(ctx, env_name=env_name)
    scenarios = sm_client.list_scenarios(include_hidden=True)
    backend_id = sm_client.resolve_template_to_backend_id(name, scenarios=scenarios)
    scenario = next((s for s in scenarios if s.scenario_id == backend_id), None)
    if scenario is None:
        raise ScenarioManagerApiError(0, f"Unknown template: {name}")
    return scenario
