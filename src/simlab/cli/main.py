"""Main CLI entry point."""

import io
import os
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

import click
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from simlab.cli.auth import _verify_key_with_server
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_scenario_manager_api_url

try:
    __version__ = version("simulationlab")
except PackageNotFoundError:  # pragma: no cover - local source tree fallback
    __version__ = "0.1.0"

# Collinear brand palette. Hex values are eyeballed from the brand swatch sheet;
# replace with canonical values from the brand guide when we have them.
COLLINEAR_ORANGE = "#ee8342"
COLLINEAR_ECRU = "#f6f0e4"
COLLINEAR_BEIGE = "#d6cbb3"
COLLINEAR_DARK = "#4a4844"
COLLINEAR_BLACK = "#0a0908"
COLLINEAR_ACCENT = "#9fafa6"

# Primary accent used throughout `--help` output.
SIMLAB_ORANGE = COLLINEAR_ORANGE

# `ansi_regular` figlet rendering of "SimLab". 5 lines, ~46 chars wide.
# Chosen over `ansi_shadow` because shadow glyphs made the S read as a 5.
SIMLAB_ASCII_ART = r"""
 ███████ ██ ███    ███ ██       █████  ██████
 ██      ██ ████  ████ ██      ██   ██ ██   ██
 ███████ ██ ██ ████ ██ ██      ███████ ██████
      ██ ██ ██  ██  ██ ██      ██   ██ ██   ██
 ███████ ██ ██      ██ ███████ ██   ██ ██████
"""

# Journey-ordered: onboard -> build -> run -> analyze.
# Order here is the order commands appear in `simlab --help`.
COMMAND_IMPORT_PATHS = {
    "quickstart": "simlab.cli.quickstart:quickstart",
    "auth": "simlab.cli.auth:auth",
    "autoresearch": "simlab.cli.autoresearch:autoresearch",
    "templates": "simlab.cli.templates:templates",
    "env": "simlab.cli.env:env",
    "tools": "simlab.cli.tools:tools",
    "tasks-gen": "simlab.cli.tasks_gen:tasks_gen",
    "tasks": "simlab.cli.tasks:tasks",
    "eval": "simlab.cli.eval:eval_command",
    "runs": "simlab.cli.runs:runs",
    "feedback": "simlab.cli.feedback:feedback",
}

# Description + example for each top-level command. Parallel to
# COMMAND_IMPORT_PATHS; the renderer reads both to build the --help table.
COMMAND_METADATA: dict[str, dict[str, str]] = {
    "quickstart": {
        "description": "Guided tour — zero to first rollout in 5 minutes.",
        "example": "simlab quickstart",
    },
    "auth": {
        "description": "Log in and manage your Collinear API credentials.",
        "example": "simlab auth login",
    },
    "autoresearch": {
        "description": "Iterate on your agent's prompt and measure whether it improves eval.",
        "example": "simlab autoresearch init",
    },
    "templates": {
        "description": "Browse pre-built scenario presets to start from.",
        "example": "simlab templates list",
    },
    "env": {
        "description": "Create and manage Simlab environments.",
        "example": "simlab env init my-env",
    },
    "tools": {
        "description": "Browse tool servers you can plug into an env.",
        "example": "simlab tools list",
    },
    "tasks-gen": {
        "description": "Generate new tasks from seed data or specs.",
        "example": "simlab tasks-gen init",
    },
    "tasks": {
        "description": "Run your agent against tasks in an env.",
        "example": "simlab tasks run",
    },
    "eval": {
        "description": "Analyze rollout artifacts and score agent runs.",
        "example": "simlab eval ./output",
    },
    "runs": {
        "description": "Browse local run history saved in the output directory.",
        "example": "simlab runs history",
    },
    "feedback": {
        "description": "Send feedback to the Collinear team.",
        "example": "simlab feedback",
    },
}

# Commands that do not require a Collinear API key (e.g. auth login/status).
AUTH_EXEMPT_COMMANDS = {"auth"}


def _render_plain_help(ctx: click.Context, group: "SimlabGroup") -> str:
    """Render a terse, unstyled help screen intended for AI agents and scripts.

    No colors, no ASCII art, no box-drawing characters — just aligned plain
    text that any tool can parse. All global options are shown (no hiding),
    since agents usually want the full interface at a glance.
    """
    lines: list[str] = [
        "SimLab — the simulation lab for hill-climbing on long-horizon agents.",
        "",
        "Usage: simlab [OPTIONS] COMMAND [ARGS]...",
        "",
        "Commands:",
    ]

    max_name = max(len(name) for name in COMMAND_IMPORT_PATHS)
    for name in COMMAND_IMPORT_PATHS:
        meta = COMMAND_METADATA.get(name, {})
        description = meta.get("description", "")
        example = meta.get("example", "")
        lines.append(f"  {name:<{max_name}}  {description}")
        if example:
            lines.append(f"  {' ' * max_name}  Example: {example}")

    lines.append("")
    lines.append("Global options:")

    opt_rows: list[tuple[str, str]] = []
    for param in group.get_params(ctx):
        if not isinstance(param, click.Option):
            continue
        if not param.hidden:
            continue
        flag_names = ", ".join(param.opts)
        if flag_names in ("--verbose", "--plain"):
            continue
        help_text = param.help or ""
        opt_rows.append((flag_names, help_text))

    if opt_rows:
        max_flag = max(len(f) for f, _ in opt_rows)
        for flag, help_text in opt_rows:
            lines.append(f"  {flag:<{max_flag}}  {help_text}")

    lines.append("")
    lines.append("Options:")
    lines.append("  --help     Show this message and exit.")
    lines.append("  --plain    Render help in plain text (this output) for AI agents.")
    lines.append("  --version  Show the version and exit.")

    return "\n".join(lines) + "\n"


def _pick_adaptive_body_color() -> str:
    """Return the body text + border color adapted to the terminal background.

    Most modern terminals (iTerm2, xterm, gnome-terminal, kitty, Warp,
    Terminal.app) set ``COLORFGBG`` as a hint of the form ``"fg;bg"`` using
    ANSI color indices. We use it to pick between ``COLLINEAR_DARK`` (for
    light-background terminals) and ``COLLINEAR_ECRU`` (for dark-background
    terminals) so the `simlab --help` screen reads well on either. When
    ``COLORFGBG`` is unset or unparseable, fall back to Rich's ``"default"``
    style, which delegates color choice to the terminal itself — always
    readable, just slightly less branded.
    """
    colorfgbg = os.environ.get("COLORFGBG", "")
    if not colorfgbg:
        return "default"
    # Format is "fg;bg" or "fg;_;bg" — the last segment is always the background.
    bg_part = colorfgbg.split(";")[-1]
    try:
        bg_num = int(bg_part)
    except ValueError:
        return "default"
    # ANSI indices 7 (white) and 15 (bright_white) indicate a light background;
    # everything else (0 black, 1-6 dark colors, 8-14 bright variants) is dark.
    if bg_num in (7, 15):
        return COLLINEAR_DARK
    return COLLINEAR_ECRU


def _render_branded_help(
    ctx: click.Context,
    group: "SimlabGroup",
    *,
    verbose: bool,
) -> str:
    """Render the branded `simlab --help` output via Rich to a string."""
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=True,
        color_system="truecolor",
        width=100,
        legacy_windows=False,
    )

    body = _pick_adaptive_body_color()

    # Visible to humans, targeted at AI agents reading stdout sequentially so
    # they can abort and re-run with --plain before hitting the styled output.
    console.print(
        Text(
            "NOTE for AI agents: use `simlab --help --plain` for a terse, "
            "machine-readable version without design elements.",
            style=f"italic {body}",
        )
    )
    console.print()

    console.print(Text(SIMLAB_ASCII_ART, style=f"bold {COLLINEAR_ORANGE}"))
    console.print(
        Text(
            "The simulation lab for hill-climbing on long-horizon agents.",
            style=f"italic {body}",
        )
    )
    console.print()

    pointer = Text("New here? Take the guided tour:  ", style=body)
    pointer.append("simlab quickstart", style=f"bold {COLLINEAR_ORANGE}")
    console.print(pointer)
    console.print()

    table = Table(
        show_header=True,
        show_lines=True,
        box=box.HEAVY,
        header_style=f"bold {COLLINEAR_ORANGE}",
        border_style=body,
    )
    table.add_column("Command", style=f"bold {COLLINEAR_ORANGE}", no_wrap=True)
    table.add_column("What it does", style=body)
    table.add_column("Example", style=COLLINEAR_ACCENT, no_wrap=False)

    for name in COMMAND_IMPORT_PATHS:
        meta = COMMAND_METADATA.get(name)
        if meta is None:
            continue
        table.add_row(name, meta["description"], meta["example"])

    console.print(table)
    console.print()

    footer = Text(" Global options:  ", style=body)
    footer.append("simlab --help --verbose", style=f"bold {COLLINEAR_ORANGE}")
    console.print(footer)

    docs = Text(" Docs:            ", style=body)
    docs.append("https://docs.collinear.ai", style=f"underline {COLLINEAR_ACCENT}")
    console.print(docs)

    if verbose:
        _render_global_options_section(console, ctx, group)

    return buffer.getvalue()


def _render_global_options_section(
    console: Console,
    ctx: click.Context,
    group: "SimlabGroup",
) -> None:
    """Append the hidden-global-options table (only when `--verbose` is set)."""
    console.print()
    console.print(Text("Global options", style=f"bold {COLLINEAR_ORANGE}"))

    body = _pick_adaptive_body_color()
    opts_table = Table(
        show_header=True,
        show_lines=True,
        box=box.HEAVY,
        header_style=f"bold {COLLINEAR_ORANGE}",
        border_style=body,
    )
    opts_table.add_column("Option", style=f"bold {COLLINEAR_ORANGE}", no_wrap=True)
    opts_table.add_column("Description", style=body)

    for param in group.get_params(ctx):
        if not isinstance(param, click.Option):
            continue
        if not param.hidden:
            continue
        if "--verbose" in (param.opts or []):
            continue
        flag_name = ", ".join(param.opts) if param.opts else (param.name or "")
        description = param.help or ""
        opts_table.add_row(flag_name, description)

    console.print(opts_table)


class SimlabGroup(click.Group):
    """Load command groups only when Click actually needs them."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Pre-scan raw args so ``--verbose`` is visible to help rendering.

        Click's ``--help`` callback is eager — it fires during parsing, before
        non-eager options like ``--verbose`` have been stored on the context.
        That means ``simlab --help --verbose`` would render plain help because
        ``--help`` triggers before Click ever sees ``--verbose``. Pre-scanning
        the raw argv here sets the flag on ``ctx.obj`` before Click starts eager
        processing, so both orderings work.
        """
        ctx.ensure_object(dict)
        ctx.obj["_help_verbose"] = "--verbose" in args
        ctx.obj["_help_plain"] = "--plain" in args
        return super().parse_args(ctx, args)

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
        if cmd_name in AUTH_EXEMPT_COMMANDS:
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

    def get_help(self, ctx: click.Context) -> str:
        """Render branded (default) or plain (agent-friendly) help output."""
        state = ctx.ensure_object(dict)
        if state.get("_help_plain"):
            return _render_plain_help(ctx, self)
        verbose = bool(state.get("_help_verbose"))
        return _render_branded_help(ctx, self, verbose=verbose)


@click.group(cls=SimlabGroup)
@click.option(
    "--config-file",
    default=None,
    type=click.Path(path_type=str, exists=False),
    help="Path to global config TOML (default: ~/.config/simlab/config.toml).",
    hidden=True,
)
@click.option(
    "--collinear-api-key",
    "collinear_api_key",
    default=None,
    help=(
        "Collinear API key for the Scenario Manager API "
        "(overrides config file and SIMLAB_COLLINEAR_API_KEY)."
    ),
    hidden=True,
)
@click.option(
    "--scenario-manager-api-url",
    "scenario_manager_api_url",
    default=None,
    help=(
        "Scenario Manager API base URL (overrides config file and SIMLAB_SCENARIO_MANAGER_API_URL)."
    ),
    hidden=True,
)
@click.option(
    "--daytona-api-key",
    default=None,
    help="Daytona API key (overrides config file and SIMLAB_DAYTONA_API_KEY / DAYTONA_API_KEY).",
    hidden=True,
)
@click.option(
    "--environments-dir",
    "environments_dir",
    default=None,
    type=click.Path(path_type=str, exists=False),
    help="Root directory for environments (default: <cwd>/environments).",
    hidden=True,
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    hidden=True,
    help="Show extra detail in `--help` output.",
)
@click.option(
    "--plain",
    is_flag=True,
    default=False,
    hidden=True,
    help="Render `--help` in plain text for AI agents (no colors or ASCII art).",
)
@click.version_option(version=__version__)
@click.pass_context
def cli(
    ctx: click.Context,
    config_file: str | None,
    collinear_api_key: str | None,
    scenario_manager_api_url: str | None,
    daytona_api_key: str | None,
    environments_dir: str | None,
    verbose: bool,  # noqa: ARG001 — read via parse_args pre-scan for help rendering
    plain: bool,  # noqa: ARG001 — read via parse_args pre-scan for help rendering
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


if __name__ == "__main__":
    cli()
