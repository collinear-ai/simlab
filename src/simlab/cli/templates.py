"""CLI commands for browsing scenario presets."""

from pathlib import Path

import click

from simlab.api.client import ScenarioManagerApiError
from simlab.api.client import ScenarioManagerClient
from simlab.api.client import resolve_scenario_manager_api_url
from simlab.api.schemas import ScenarioSummary
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_env_dir
from simlab.telemetry import TelemetryCaptureConfig
from simlab.telemetry import emit_cli_event
from simlab.telemetry import normalize_config_path
from simlab.telemetry import resolve_scenario_manager_capture_config
from simlab.telemetry import with_command_telemetry
from simlab.templates.loader import TemplateLoader


def _truncate_for_table(value: str, max_width: int) -> str:
    """Return a single-line value safe for fixed-width template list output."""
    text = (value or "").replace("\n", " ").strip()
    if len(text) <= max_width:
        return text
    if max_width < 2:
        return text[:max_width]
    return text[: max_width - 1] + "…"


def _get_loader() -> TemplateLoader:
    loader = TemplateLoader()
    loader.load_all()
    return loader


@click.group()
def templates() -> None:
    """Browse pre-defined scenario presets."""


def templates_list_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for template listing when a Collinear API key is configured."""
    _ = args
    return resolve_scenario_manager_capture_config(
        ctx,
        config_path=normalize_config_path(kwargs.get("config_path")),
    )


def templates_info_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for local template inspection when a Collinear API key is configured."""
    _ = args, kwargs
    return resolve_scenario_manager_capture_config(ctx)


@templates.command("list")
@click.option(
    "--env",
    "env_name",
    default=None,
    help="Environment name (under environments/). Uses that env's scenario_manager_api_url.",
)
@click.pass_context
@with_command_telemetry("templates list", resolver=templates_list_capture_config)
def list_templates(ctx: click.Context, env_name: str | None) -> None:
    """List all available scenario presets (from Scenario Manager scenarios)."""
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

    name_w = 28
    slug_w = 24
    desc_w = 78
    click.echo()
    click.echo(f"  {'Name':<{name_w}} {'Slug':<{slug_w}} {'Description':<{desc_w}}")
    click.echo(f"  {'─' * name_w} {'─' * slug_w} {'─' * desc_w}")
    for s in scenarios:
        scenario: ScenarioSummary = s
        scenario_id = scenario.scenario_id.strip() or "?"
        name = scenario.name.strip() or "?"
        desc = (scenario.description or "").strip() or "—"
        tool_names = [ts.name.strip() for ts in scenario.tool_servers if ts.name.strip()]
        name_text = _truncate_for_table(name, name_w)
        slug_text = _truncate_for_table(scenario_id, slug_w)
        desc_text = _truncate_for_table(desc, desc_w)
        click.echo(f"  {name_text:<{name_w}} {slug_text:<{slug_w}} {desc_text}")
        if tool_names:
            tools_text = _truncate_for_table(", ".join(tool_names), 120)
            click.echo(click.style(f"    Tools: {tools_text}", fg="yellow"))
        click.echo()

    click.echo(f"  {len(scenarios)} templates available")
    click.echo()
    emit_cli_event("templates_list_completed", {"template_count": len(scenarios)})


@templates.command("info")
@click.argument("name")
@with_command_telemetry("templates info", resolver=templates_info_capture_config)
def template_info(name: str) -> None:
    """Show detailed info about a specific template."""
    loader = _get_loader()
    tpl = loader.get_template(name)
    if tpl is None:
        click.echo(click.style(f"Unknown template: {name}", fg="red"), err=True)
        raise SystemExit(1)

    click.echo()
    click.echo(click.style(f"  {tpl.display_name}", fg="cyan", bold=True))
    click.echo(f"  {tpl.description}")
    click.echo()
    click.echo(click.style("  Included tools:", bold=True))
    for tool_name in tpl.tools:
        click.echo(f"    - {tool_name}")
    click.echo()
