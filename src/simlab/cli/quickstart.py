"""Interactive quickstart command — one command to run the full SimLab loop.

Extracts reusable helpers for env init and task discovery so quickstart
owns all terminal output and avoids ctx.invoke coupling.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
import yaml
from rich import box
from rich.console import Console
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from simlab.api.client import ScenarioManagerApiError
from simlab.api.client import ScenarioManagerClient
from simlab.api.schemas import ScenarioSummary
from simlab.api.schemas import ScenarioTask
from simlab.cli.env import _DEFAULT_CODING_SCENARIO_GUIDANCE
from simlab.cli.env import _extract_tools_from_scenario
from simlab.cli.progress import StepProgress
from simlab.composer.engine import EnvConfig
from simlab.config import get_env_dir
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_scenario_manager_api_url
from simlab.env_artifacts import regenerate_env_artifacts
from simlab.env_registry import build_registry

try:
    import questionary as _questionary
except ImportError:
    questionary: Any | None = None
else:
    questionary = _questionary


# ---------------------------------------------------------------------------
# Extracted helpers (no CLI output)
# ---------------------------------------------------------------------------


def init_environment(
    env_dir: Path,
    env_name: str,
    tools: list[str],
    *,
    template: str | None = None,
    image_registry: str | None = None,
    force: bool = False,
) -> None:
    """Create env.yaml and generate docker-compose artifacts.

    Pure business logic — no CLI output, no prompts, no hints.
    """
    env_dir.mkdir(parents=True, exist_ok=True)
    env_yaml = env_dir / "env.yaml"

    if env_yaml.exists() and not force:
        return

    config_kwargs: dict[str, Any] = {
        "name": env_name,
        "tools": tools,
        "registry": image_registry,
        "template": template,
    }
    if "coding" in tools:
        config_kwargs["coding"] = {
            "setup_scripts": ["./coding/setup/install-tools.sh"],
            "skills": ["./coding/skills"],
            "mounts": [
                {
                    "source": "./coding/fixtures",
                    "target": "/workspace/fixtures",
                    "read_only": True,
                }
            ],
        }
        config_kwargs["scenario_guidance_md"] = _DEFAULT_CODING_SCENARIO_GUIDANCE.strip()
    config = EnvConfig(**config_kwargs)
    config_data = {k: v for k, v in config.model_dump().items() if v is not None}
    env_yaml.write_text(yaml.dump(config_data, default_flow_style=False, sort_keys=False))
    regenerate_env_artifacts(env_dir)


def resolve_template_tools(
    sm_client: ScenarioManagerClient,
    template: str,
) -> tuple[str, list[str], ScenarioSummary]:
    """Resolve a template name to its backend ID, tool list, and scenario summary.

    Returns (backend_id, tool_names, scenario_summary).
    """
    registry = build_registry()
    scenarios = sm_client.list_scenarios(include_hidden=True)
    backend_id = sm_client.resolve_template_to_backend_id(template, scenarios=scenarios)
    scenario = next(
        (s for s in scenarios if s.scenario_id.strip() == backend_id),
        None,
    )
    if scenario is None:
        msg = (
            f"Template '{template}' resolved to '{backend_id}', but no matching scenario was found."
        )
        raise ValueError(msg)

    selected, _missing = _extract_tools_from_scenario(registry, scenario)
    if not selected:
        msg = f"Template '{backend_id}' has no compatible tool servers."
        raise ValueError(msg)
    return backend_id, selected, scenario


def discover_tasks(
    sm_client: ScenarioManagerClient,
    backend_id: str,
) -> list[ScenarioTask]:
    """Fetch available tasks for a template. Returns empty list if none."""
    resp = sm_client.list_scenario_tasks(backend_id, include_test=False)
    return resp.tasks


# ---------------------------------------------------------------------------
# Task selection UI (owned by quickstart)
# ---------------------------------------------------------------------------


def _select_default_task(tasks: list[ScenarioTask]) -> ScenarioTask:
    """Pick the first easy-difficulty task, or the first task if none are easy."""
    for t in tasks:
        if t.difficulty and t.difficulty.lower() == "easy":
            return t
    return tasks[0]


def _interactive_task_select(tasks: list[ScenarioTask]) -> str:
    """Let the user pick a task interactively."""
    default_task = _select_default_task(tasks)

    click.echo()
    click.echo(click.style("  Choose a task for your agent to solve.", bold=True))
    click.echo(
        click.style(
            "  Tasks are ranked by difficulty — start with an easy one to verify your setup.",
            dim=True,
        )
    )

    if questionary is not None:
        click.echo(click.style("  (Use arrow keys to move, Enter to select)", dim=True))
        click.echo()
        choices = [
            questionary.Choice(
                title=f"{t.name or t.task_id} ({t.difficulty or 'unknown'})",
                value=t.task_id,
            )
            for t in tasks
        ]
        selected = questionary.select(
            "  Select a task:",
            choices=choices,
            default=default_task.task_id,
        ).ask()
        if selected is None:
            raise SystemExit(1)
        return selected

    # Fallback: numbered list + click.prompt
    click.echo()
    for i, t in enumerate(tasks, 1):
        marker = " (default)" if t.task_id == default_task.task_id else ""
        click.echo(f"  {i}. {t.name or t.task_id} ({t.difficulty or 'unknown'}){marker}")
    click.echo()
    default_idx = next(i for i, t in enumerate(tasks, 1) if t.task_id == default_task.task_id)
    choice = click.prompt("  Select a task", type=int, default=default_idx)
    if choice < 1 or choice > len(tasks):
        click.echo(click.style(f"  Invalid choice: {choice}", fg="red"), err=True)
        raise SystemExit(1)
    return tasks[choice - 1].task_id


def _find_task_by_id(
    tasks: list[ScenarioTask],
    task_id: str,
) -> ScenarioTask | None:
    """Match a task ID against the task list (exact, suffix, or substring)."""
    for t in tasks:
        if t.task_id == task_id:
            return t
    for t in tasks:
        if t.task_id.endswith(task_id):
            return t
    for t in tasks:
        if task_id in t.task_id:
            return t
    return None


# ---------------------------------------------------------------------------
# Rich UI helpers
# ---------------------------------------------------------------------------


def _print_welcome_panel(console: Console, *, template: str, env_name: str) -> None:
    """Render the welcome banner with the three-stage agenda."""
    title = Text("Welcome to SimLab", style="bold cyan")
    subtitle = Text(
        "The fastest way to run your first agent rollout.",
        style="dim",
    )

    agenda = Table.grid(padding=(0, 1))
    agenda.add_column(no_wrap=True, style="cyan")
    agenda.add_column(overflow="fold")
    agenda.add_row("1.", "Set up an environment")
    agenda.add_row("2.", "Pick a task for your agent to solve")
    agenda.add_row("3.", "Run your agent and see the results")

    meta = Table.grid(padding=(0, 1))
    meta.add_column(no_wrap=True, style="bold")
    meta.add_column(overflow="fold")
    meta.add_row("Template", template)
    meta.add_row("Environment", env_name)

    body = Group(title, subtitle, Text(""), agenda, Text(""), meta)
    panel = Panel(
        body,
        box=box.SQUARE,
        border_style="cyan",
        padding=(1, 2),
    )
    console.print()
    console.print(panel, highlight=False)
    console.print()


def _print_stage_header(console: Console, index: int, total: int, label: str) -> None:
    """Print a compact step marker like `[1/3] Environment`."""
    console.print(
        Text.assemble(
            ("  ", ""),
            (f"[{index}/{total}] ", "bold cyan"),
            (label, "bold"),
        ),
        highlight=False,
    )


def _print_template_card(
    console: Console,
    scenario: ScenarioSummary,
    *,
    tools: list[str],
) -> None:
    """Render a short card describing the chosen template."""
    description = (scenario.description or "").strip() or "No description available."
    tool_names = ", ".join(tools) if tools else "none"

    fields = Table.grid(padding=(0, 1))
    fields.add_column(no_wrap=True, style="bold")
    fields.add_column(overflow="fold")
    fields.add_row("Name", scenario.name or scenario.scenario_id)
    fields.add_row("About", description)
    fields.add_row("Tools", tool_names)

    console.print(
        Panel(
            fields,
            box=box.SQUARE,
            border_style="dim",
            padding=(0, 1),
            title="Template",
            title_align="left",
        ),
        highlight=False,
    )


def _print_task_card(console: Console, task: ScenarioTask) -> None:
    """Render a short card describing the chosen task."""
    name = task.name or task.task_id
    difficulty = (task.difficulty or "unknown").lower()
    diff_color = {
        "easy": "green",
        "medium": "yellow",
        "hard": "red",
    }.get(difficulty, "white")

    description = (task.description or "").strip()
    if len(description) > 240:
        description = description[:237] + "…"
    if not description:
        description = "No description available."

    fields = Table.grid(padding=(0, 1))
    fields.add_column(no_wrap=True, style="bold")
    fields.add_column(overflow="fold")
    fields.add_row("Name", name)
    fields.add_row("Difficulty", Text(difficulty, style=diff_color))
    fields.add_row("ID", Text(task.task_id, style="dim"))
    fields.add_row("About", description)

    console.print(
        Panel(
            fields,
            box=box.SQUARE,
            border_style="dim",
            padding=(0, 1),
            title="Task",
            title_align="left",
        ),
        highlight=False,
    )


# ---------------------------------------------------------------------------
# Quickstart command
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--template",
    "-t",
    default=None,
    help="Scenario template (default: hr).",
)
@click.option(
    "--env-name",
    default=None,
    help="Environment name (default: quickstart-<template>).",
)
@click.option(
    "--task",
    "task_id",
    default=None,
    help="Task ID to run (skip interactive selection).",
)
@click.option(
    "--agent-model",
    "model",
    default=None,
    help="LLM model name (prompted if not provided).",
)
@click.option(
    "--agent-api-key",
    "api_key",
    default=None,
    help="Agent API key (falls back to SIMLAB_AGENT_API_KEY / provider env var).",
)
@click.option(
    "--daytona",
    is_flag=True,
    default=False,
    help="Run against a Daytona sandbox instead of local Docker.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing quickstart environment.",
)
@click.pass_context
def quickstart(
    ctx: click.Context,
    template: str | None,
    env_name: str | None,
    task_id: str | None,
    model: str | None,
    api_key: str | None,
    daytona: bool,
    force: bool,
) -> None:
    """Run the full SimLab loop in one command — init, pick a task, run, see results."""
    global_cfg = get_global_config_from_ctx(ctx)
    console = Console()
    progress = StepProgress(console=console)

    # 1. Resolve template, env name, API client
    template = template or "hr"
    env_name = env_name or f"quickstart-{template}"
    env_dir = get_env_dir(env_name, ctx=ctx)

    _print_welcome_panel(console, template=template, env_name=env_name)

    base_url = resolve_scenario_manager_api_url(config=global_cfg)
    collinear_key = resolve_collinear_api_key(config=global_cfg)
    sm_client = ScenarioManagerClient(base_url=base_url, api_key=collinear_key)

    # 2. Stage 1: Environment
    _print_stage_header(console, 1, 3, "Environment")
    try:
        with progress.step(
            "Resolving template",
            success_label="Template resolved",
        ):
            backend_id, tools, scenario = resolve_template_tools(sm_client, template)
    except (ScenarioManagerApiError, ValueError) as e:
        console.print(f"  [red]Error:[/red] {e}")
        raise SystemExit(1) from e

    _print_template_card(console, scenario, tools=tools)

    env_yaml = env_dir / "env.yaml"
    if not env_yaml.exists() or force:
        with progress.step(
            "Initializing environment",
            success_label="Environment initialized",
        ):
            init_environment(
                env_dir,
                env_name,
                tools,
                template=backend_id,
                force=force,
            )
    else:
        console.print("  [green]✓[/green] Using existing environment")
    console.print()

    # 3. Stage 2: Task
    _print_stage_header(console, 2, 3, "Task")
    try:
        with progress.step(
            "Fetching available tasks",
            success_label="Tasks loaded",
        ):
            available_tasks = discover_tasks(sm_client, backend_id)
    except ScenarioManagerApiError as e:
        console.print(f"  [red]Error fetching tasks:[/red] {e}")
        raise SystemExit(1) from e

    if not available_tasks:
        console.print("  [red]No tasks found for this template.[/red]")
        raise SystemExit(1)

    if task_id:
        matched = _find_task_by_id(available_tasks, task_id)
        if matched is None:
            console.print(f"  [red]Task '{task_id}' not found.[/red]")
            console.print("  Available tasks:")
            for t in available_tasks:
                console.print(
                    f"    • {t.name or t.task_id} "
                    f"[dim]({t.difficulty or 'unknown'}) — {t.task_id}[/dim]"
                )
            raise SystemExit(1)
        selected_task = matched
    else:
        task_id = _interactive_task_select(available_tasks)
        selected_task = next(t for t in available_tasks if t.task_id == task_id)

    task_id = selected_task.task_id
    _print_task_card(console, selected_task)
    console.print()

    # 4. Stage 3: Run
    _print_stage_header(console, 3, 3, "Run")
    if not model:
        model = (global_cfg.agent_model or "").strip()
    if not model:
        console.print()
        console.print("  [bold]Which LLM should run the task?[/bold]")
        console.print("  [dim]Examples: gpt-4o, claude-sonnet-4-6, gemini/gemini-2.5-flash[/dim]")
        console.print(
            "  [dim](Requires a corresponding API key, e.g. OPENAI_API_KEY or "
            "ANTHROPIC_API_KEY)[/dim]"
        )
        model = click.prompt("  Agent model", default="gpt-4o")

    console.print(f"  [green]✓[/green] Model: [bold]{model}[/bold]")
    console.print()
    console.print("  [bold cyan]▸[/bold cyan] [bold]Starting task run...[/bold]")
    console.print()

    # 5. Delegate to `simlab tasks run` via subprocess.
    #    This keeps quickstart and tasks run as separate terminal owners —
    #    tasks run controls its own Rich progress/spinner output cleanly.
    #    Forward top-level CLI overrides: non-secret flags via CLI args,
    #    secrets via environment variables (avoids exposure in `ps aux`).
    overrides = (ctx.obj or {}).get("global_config_overrides", {})
    global_args: list[str] = []
    if overrides.get("config_file"):
        global_args.extend(["--config-file", overrides["config_file"]])
    if overrides.get("scenario_manager_api_url"):
        global_args.extend(["--scenario-manager-api-url", overrides["scenario_manager_api_url"]])
    if overrides.get("environments_dir"):
        global_args.extend(["--environments-dir", overrides["environments_dir"]])

    # Pass secrets via env vars, not CLI flags (visible in process listings).
    child_env = os.environ.copy()
    if overrides.get("collinear_api_key"):
        child_env["SIMLAB_COLLINEAR_API_KEY"] = overrides["collinear_api_key"]
    if overrides.get("daytona_api_key"):
        child_env["SIMLAB_DAYTONA_API_KEY"] = overrides["daytona_api_key"]
    if api_key:
        child_env["SIMLAB_AGENT_API_KEY"] = api_key

    cmd = [
        sys.executable,
        "-m",
        "simlab.cli.main",
        *global_args,
        "tasks",
        "run",
        "--env",
        env_name,
        "--task",
        task_id,
        "--agent-model",
        model,
    ]
    if daytona:
        cmd.append("--daytona")

    proc = subprocess.run(cmd, env=child_env, check=False)  # noqa: S603
    raise SystemExit(proc.returncode)
