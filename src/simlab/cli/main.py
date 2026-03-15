"""Main CLI entry point."""

from importlib import import_module

import click

from simlab.cli.auth import _verify_key_with_server
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_scenario_manager_api_url

COMMAND_IMPORT_PATHS = {
    "auth": "simlab.cli.auth:auth",
    "env": "simlab.cli.env:env",
    "tasks": "simlab.cli.tasks:tasks",
    "tasks-gen": "simlab.cli.tasks_gen:tasks_gen",
    "templates": "simlab.cli.templates:templates",
    "tools": "simlab.cli.tools:tools",
}

# Commands that do not require a Collinear API key (e.g. auth login/status).
_AUTH_EXEMPT_COMMANDS = {"auth"}


class SimlabGroup(click.Group):
    """Load command groups only when Click actually needs them."""

    def list_commands(self, _ctx: click.Context) -> list[str]:
        """Return the stable command list without importing command modules."""
        return list(COMMAND_IMPORT_PATHS)

    def get_command(self, _ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Import one command group only when Click resolves that command name."""
        command = self.commands.get(cmd_name)
        if command is not None:
            return command

        import_path = COMMAND_IMPORT_PATHS.get(cmd_name)
        if import_path is None:
            return None

        module_name, attr_name = import_path.split(":", maxsplit=1)
        module = import_module(module_name)
        command = getattr(module, attr_name)
        self.add_command(command, name=cmd_name)
        return command

    def resolve_command(
        self,
        ctx: click.Context,
        args: list[str],
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Resolve the requested command and gate execution when auth is missing."""
        cmd_name, command, remaining = super().resolve_command(ctx, args)
        if any(arg in ctx.help_option_names for arg in remaining):
            return cmd_name, command, remaining
        if cmd_name in _AUTH_EXEMPT_COMMANDS:
            return cmd_name, command, remaining

        global_cfg = get_global_config_from_ctx(ctx)
        if not global_cfg.collinear_api_key:
            raise click.ClickException(
                "API key required. Run: simlab auth login\n\n"
                "Get your key at https://platform.collinear.ai\n"
                "(Developer Resources → API Keys)"
            )

        api_url = resolve_scenario_manager_api_url(
            base_url=global_cfg.scenario_manager_api_url,
        )
        verified = _verify_key_with_server(global_cfg.collinear_api_key, api_url)
        if verified is False:
            raise click.ClickException(
                "Invalid API key. Run: simlab auth login\n\n"
                "Get your key at https://platform.collinear.ai\n"
                "(Developer Resources → API Keys)"
            )
        return cmd_name, command, remaining


@click.group(cls=SimlabGroup)
@click.option(
    "--config-file",
    default=None,
    type=click.Path(path_type=str, exists=False),
    help="Path to global config TOML (default: ~/.config/simlab/config.toml).",
)
@click.option(
    "--collinear-api-key",
    "collinear_api_key",
    default=None,
    help=("Collinear API key for rl-gym-api (overrides config file and SIMLAB_COLLINEAR_API_KEY)."),
)
@click.option(
    "--scenario-manager-api-url",
    "scenario_manager_api_url",
    default=None,
    help=(
        "Scenario Manager API base URL (overrides config file and SIMLAB_SCENARIO_MANAGER_API_URL)."
    ),
)
@click.option(
    "--daytona-api-key",
    default=None,
    help="Daytona API key (overrides config file and SIMLAB_DAYTONA_API_KEY / DAYTONA_API_KEY).",
)
@click.option(
    "--environments-dir",
    "environments_dir",
    default=None,
    type=click.Path(path_type=str, exists=False),
    help="Root directory for environments (default: <cwd>/environments).",
)
@click.version_option(package_name="simlab")
@click.pass_context
def cli(
    ctx: click.Context,
    config_file: str | None,
    collinear_api_key: str | None,
    scenario_manager_api_url: str | None,
    daytona_api_key: str | None,
    environments_dir: str | None,
) -> None:
    """Simlab — browse tool servers and generate environments."""
    ctx.ensure_object(dict)
    ctx.obj["global_config_overrides"] = {
        "config_file": config_file,
        "collinear_api_key": collinear_api_key,
        "scenario_manager_api_url": scenario_manager_api_url,
        "daytona_api_key": daytona_api_key,
        "environments_dir": environments_dir,
    }
