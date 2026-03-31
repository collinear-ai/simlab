"""CLI commands for browsing scenario presets."""

import shutil
import textwrap
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


def _render_template_summary(
    scenario: ScenarioSummary,
    *,
    width: int,
) -> list[str]:
    """Render one template summary row as a wrapped multi-line block."""
    fields = [
        ("Name", scenario.name, "?"),
        ("Slug", scenario.scenario_id, "?"),
        ("Description", scenario.description, "—"),
    ]
    tool_names = [tool.name.strip() for tool in scenario.tool_servers if tool.name.strip()]
    if tool_names:
        fields.append(("Tools", ", ".join(tool_names), "—"))

    lines: list[str] = []
    for label, value, fallback in fields:
        text = " ".join((value or "").split()) or fallback
        prefix = f"    {label}: "
        lines.extend(
            textwrap.wrap(
                text,
                width=max(width, len(prefix) + 10),
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix),
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return lines


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

    scenario_ids = {scenario.scenario_id.strip() for scenario in scenarios}
    if "hr" in scenario_ids:
        scenarios = [
            scenario
            for scenario in scenarios
            if scenario.scenario_id.strip() not in {"hr_recruiting", "hr_people_management"}
        ]

    width = max(80, shutil.get_terminal_size((100, 20)).columns)
    click.echo()
    for s in scenarios:
        scenario: ScenarioSummary = s
        for line in _render_template_summary(scenario, width=width):
            click.echo(line)
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
