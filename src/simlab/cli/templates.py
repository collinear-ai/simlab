"""CLI commands for browsing scenario presets."""

import shutil
import textwrap

import click

from simlab.api.client import ScenarioManagerApiError
from simlab.api.schemas import ScenarioSummary
from simlab.config import resolve_env_dir
from simlab.runtime.templates import get_template
from simlab.runtime.templates import list_templates as list_templates_runtime
from simlab.telemetry import TelemetryCaptureConfig
from simlab.telemetry import emit_cli_event
from simlab.telemetry import normalize_config_path
from simlab.telemetry import resolve_scenario_manager_capture_config
from simlab.telemetry import with_command_telemetry


def _render_template_summary(
    scenario: ScenarioSummary,
    *,
    width: int,
) -> list[str]:
    """Render one template summary row as a wrapped multi-line block."""
    fields = [
        ("Name", scenario.name, "?"),
        ("Version", scenario.scenario_id, "?"),
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
    config_path = None
    env_name = kwargs.get("env_name")
    if isinstance(ctx, click.Context) and isinstance(env_name, str) and env_name:
        config_path = resolve_env_dir(env_name, ctx) / "env.yaml"
    return resolve_scenario_manager_capture_config(
        ctx,
        config_path=normalize_config_path(config_path),
    )


def templates_info_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for template inspection when a Collinear API key is configured."""
    _ = args
    config_path = None
    env_name = kwargs.get("env_name")
    if isinstance(ctx, click.Context) and isinstance(env_name, str) and env_name:
        config_path = resolve_env_dir(env_name, ctx) / "env.yaml"
    return resolve_scenario_manager_capture_config(
        ctx,
        config_path=normalize_config_path(config_path),
    )


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
    try:
        scenarios = list_templates_runtime(ctx, env_name=env_name)
    except ScenarioManagerApiError as e:
        click.echo(click.style(str(e), fg="red"), err=True)
        raise SystemExit(1) from e

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
@click.option(
    "--env",
    "env_name",
    default=None,
    help="Environment name (under environments/). Uses that env's scenario_manager_api_url.",
)
@click.pass_context
@with_command_telemetry("templates info", resolver=templates_info_capture_config)
def template_info(ctx: click.Context, name: str, env_name: str | None) -> None:
    """Show detailed info about a specific template."""
    try:
        scenario = get_template(ctx, name, env_name=env_name)
    except ScenarioManagerApiError as e:
        click.echo(click.style(str(e), fg="red"), err=True)
        raise SystemExit(1) from e

    click.echo()
    width = max(80, shutil.get_terminal_size((100, 20)).columns)
    for line in _render_template_summary(scenario, width=width):
        click.echo(line)
    click.echo()
