"""CLI command for submitting user feedback."""

import click

from simlab.config import telemetry_disabled
from simlab.telemetry import TelemetryCaptureConfig
from simlab.telemetry import emit_cli_event
from simlab.telemetry import resolve_scenario_manager_capture_config
from simlab.telemetry import with_command_telemetry


def feedback_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for the feedback command when a Collinear API key is configured."""
    _ = args, kwargs
    return resolve_scenario_manager_capture_config(ctx)


@click.command("feedback")
@click.argument("message", required=False)
@click.option("--env", "env_name", default=None, help="Environment name to include as context.")
@with_command_telemetry("feedback", resolver=feedback_capture_config)
def feedback(message: str | None, env_name: str | None) -> None:
    """Send feedback to the Collinear team."""
    if telemetry_disabled():
        click.echo()
        click.echo(
            click.style(
                "Feedback could not be sent because telemetry is disabled.",
                fg="yellow",
            )
        )
        click.echo(
            "To re-enable, unset SIMLAB_DISABLE_TELEMETRY or set "
            "[telemetry] disabled = false in config.toml."
        )
        click.echo("Or send feedback directly to simlab@collinear.ai.")
        return

    if not message:
        message = click.prompt("What's on your mind?")

    if not message or not message.strip():
        click.echo(click.style("No feedback provided.", fg="yellow"))
        return

    properties: dict[str, str] = {"message": message.strip()}
    if env_name:
        properties["env_name"] = env_name

    emit_cli_event("feedback_submitted", properties)
    click.echo()
    click.echo(click.style("Thanks! Your feedback has been sent.", fg="green"))
