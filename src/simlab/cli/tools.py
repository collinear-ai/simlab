"""CLI commands for browsing the tool catalog."""

from pathlib import Path

import click

from simlab.api.client import ScenarioManagerApiError
from simlab.api.client import ScenarioManagerClient
from simlab.api.client import resolve_scenario_manager_api_url
from simlab.api.schemas import ScenarioSummary
from simlab.api.schemas import ScenarioToolServer
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_env_dir
from simlab.env_registry import build_registry
from simlab.telemetry import TelemetryCaptureConfig
from simlab.telemetry import emit_cli_event
from simlab.telemetry import normalize_config_path
from simlab.telemetry import resolve_scenario_manager_capture_config
from simlab.telemetry import with_command_telemetry


def list_tools_from_scenarios(scenarios: list[ScenarioSummary]) -> int:
    """List tools by deriving the union of tool servers from the loaded scenarios."""
    seen: set[str] = set()
    by_type: dict[str, list[tuple[str, str]]] = {}
    for s in scenarios:
        scenario: ScenarioSummary = s
        for ts in scenario.tool_servers:
            tool_server: ScenarioToolServer = ts
            name = tool_server.name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            server_type = (tool_server.server_type or "unknown").strip()
            by_type.setdefault(server_type, []).append((name, server_type))
    # Flatten to one list sorted by name for stable output; group by server_type
    for server_type in sorted(by_type.keys()):
        entries = sorted(by_type[server_type], key=lambda x: x[0])
        click.echo(click.style(f"\n  {server_type.upper()}", fg="cyan", bold=True))
        for name, _ in entries:
            name_col = click.style(f"  {name:<22}", fg="green")
            click.echo(f"  {name_col}  (type: {server_type})")
    click.echo()
    click.echo(f"  {len(seen)} tools available from Scenario Manager API")
    click.echo()
    return len(seen)


@click.group()
def tools() -> None:
    """Browse available tool servers."""


def tools_list_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for tools listing when a Collinear API key is configured."""
    _ = args
    return resolve_scenario_manager_capture_config(
        ctx,
        config_path=normalize_config_path(kwargs.get("config_path")),
    )


def tools_info_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for local catalog inspection when a Collinear API key is configured."""
    _ = args, kwargs
    return resolve_scenario_manager_capture_config(ctx)


@tools.command("list")
@click.option(
    "--env",
    "env_name",
    default=None,
    help="Environment name (under environments/). Uses that env's scenario_manager_api_url.",
)
@click.pass_context
@with_command_telemetry("tools list", resolver=tools_list_capture_config)
def list_tools(ctx: click.Context, env_name: str | None) -> None:
    """List all available tool servers grouped by category (from Scenario Manager API)."""
    global_cfg = get_global_config_from_ctx(ctx)
    config_path: Path | None = None
    if env_name:
        env_dir = resolve_env_dir(env_name, ctx)
        config_path = env_dir / "env.yaml"
    base_url = resolve_scenario_manager_api_url(
        config_path=config_path,
        config=global_cfg,
    )
    api_key = resolve_collinear_api_key(config=global_cfg)
    sm_client = ScenarioManagerClient(base_url=base_url, api_key=api_key)
    try:
        scenarios = sm_client.list_scenarios()
    except ScenarioManagerApiError as e:
        click.echo(click.style(str(e), fg="red"), err=True)
        raise SystemExit(1) from e
    tool_count = list_tools_from_scenarios(scenarios)
    emit_cli_event("tools_list_completed", {"tool_count": tool_count})


@tools.command("info")
@click.argument("name")
@click.option(
    "--env",
    "env_name",
    default=None,
    help="Environment name (include that env's custom tools when resolving the tool).",
)
@with_command_telemetry("tools info", resolver=tools_info_capture_config)
def tool_info(name: str, env_name: str | None) -> None:
    """Show detailed info about a specific tool server."""
    env_dir = resolve_env_dir(env_name, ctx=None) if env_name else None
    registry = build_registry(env_dir=env_dir)

    tool = registry.get_tool(name)
    if tool is None:
        click.echo(click.style(f"Unknown tool: {name}", fg="red"), err=True)
        raise SystemExit(1)

    click.echo()
    click.echo(click.style(f"  {tool.display_name}", fg="cyan", bold=True))
    click.echo(f"  {tool.description}")
    click.echo()
    click.echo(f"  Category:    {tool.category}")
    if tool.is_external:
        click.echo(f"  URL:         {tool.tool_server_url}")
    elif tool.tool_server_port:
        click.echo(f"  Tool port:   {tool.tool_server_port}")
    if tool.services:
        click.echo(f"  Services:    {', '.join(tool.services.keys())}")

    if tool.exposed_ports:
        click.echo()
        click.echo(click.style("  Web UIs:", bold=True))
        for ep in tool.exposed_ports:
            click.echo(f"    :{ep.port} — {ep.description}")

    if tool.seed_services:
        click.echo()
        click.echo(click.style("  Seed services:", bold=True))
        for svc_name, svc_def in tool.seed_services.items():
            click.echo(f"    {svc_name} ({svc_def.image})")

    if tool.required_env_vars:
        click.echo()
        click.echo(click.style("  Required env vars:", bold=True))
        for req in tool.required_env_vars:
            if req.description:
                click.echo(f"    {req.name} — {req.description}")
            else:
                click.echo(f"    {req.name}")

    click.echo()
