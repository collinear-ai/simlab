"""CLI commands for environment management — init, up, down, seed."""

from __future__ import annotations

import itertools
import json
import random
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from textwrap import dedent
from typing import Any

import click
import yaml

from simlab.api.client import ScenarioManagerApiError
from simlab.api.client import ScenarioManagerClient
from simlab.api.client import resolve_scenario_manager_api_url
from simlab.api.schemas import ScenarioSummary
from simlab.catalog.registry import ToolRegistry
from simlab.cli.progress import StepContext
from simlab.cli.progress import StepProgress
from simlab.cli.progress import StepProgressReporter
from simlab.composer.engine import DEFAULT_IMAGE_REGISTRY
from simlab.composer.engine import ComposeEngine
from simlab.composer.engine import EnvConfig
from simlab.composer.engine import write_output
from simlab.config import get_env_dir
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_env_dir
from simlab.runtime.compose_ps import parse_ps_output
from simlab.seeder import query_tool_server
from simlab.telemetry import TelemetryCaptureConfig
from simlab.telemetry import emit_cli_event
from simlab.telemetry import normalize_config_path
from simlab.telemetry import resolve_scenario_manager_capture_config
from simlab.telemetry import with_command_telemetry

try:
    import questionary as _questionary
except ImportError:
    questionary: Any | None = None
else:
    questionary = _questionary


_CODING_SETUP_SCRIPT = dedent(
    """\
    #!/usr/bin/env bash
    set -euo pipefail

    # Install any extra CLI tools needed by this environment.
    # Examples:
    #   apt-get update -qq && apt-get install -y -qq jq poppler-utils
    #   uv pip install --quiet --system xlsx2csv
    #   npm install -g @googleworkspace/cli
    """
)

_CODING_SKILL_STUB = dedent(
    """\
    # Example Coding Skill

    Use this directory for reusable workflow notes that the coding agent can discover.

    ## When to use this skill

    - When a task requires a specialized CLI workflow
    - When the environment includes tools with non-obvious flags or output formats

    ## Notes

    - Prefer installed CLI tools before writing custom parsers
    - Keep the skill short and operational
    """
)

_DEFAULT_CODING_SCENARIO_GUIDANCE = dedent(
    """\
    # Coding Scenario Guidance

    This guidance applies to tasks that run against this Simlab coding environment.

    ## Expectations

    - Use mounted fixtures under `/workspace/fixtures` when tasks reference local files
    - Prefer available CLI tools before writing one-off parsers
    - If coding skills are available, consult them before guessing uncommon commands
    """
)

_CODING_TASK_BUNDLE_README = dedent(
    """\
    # Custom Coding Tasks

    This directory is a local task bundle for the coding environment.

    ## Layout

    - `tasks/*.json`: task definitions
    - `verifiers/*.py`: verifier modules referenced by task JSON

    ## Running a task

    If this environment lives under your configured Simlab environments directory:

    ```bash
    simlab env up <env-name>

    simlab tasks run \
      --env <env-name> \
      --tasks-dir ./task-bundle \
      --task example_task \
      --agent-model <model>
    ```

    If the environment lives outside your default environments directory, add
    `--environments-dir <parent-dir>` to the command.
    """
)

_CODING_SAMPLE_TASKS = [
    {
        "filename": "example_task.json",
        "content": dedent(
            """\
            {
              "name": "Example Custom Coding Task",
              "category": "task",
              "apps": ["coding"],
              "meta": {
                "task_id": "example_task",
                "display_name": "Example Custom Coding Task",
                "difficulty": "easy"
              },
              "task": "Summarize available tools in `workspace_summary.md`.",
              "tool_servers": [
                {
                  "name": "coding-env",
                  "tool_server_url": "http://localhost:8020"
                }
              ],
              "verifiers": [
                {
                  "func": "python_module",
                  "module": "simlab.verifiers.custom_coding"
                }
              ]
            }
            """
        ),
    },
    {
        "filename": "build_cli_task.json",
        "content": dedent(
            """\
        {
          "name": "Build a Word Count CLI",
          "category": "task",
          "apps": ["coding"],
          "meta": {
            "task_id": "build_cli_task",
            "display_name": "Build a Word Count CLI",
            "difficulty": "medium"
          },
          "task": "Build `wordcount.py`: count words, lines, and chars in a file.",
          "tool_servers": [
            {
              "name": "coding-env",
              "tool_server_url": "http://localhost:8020"
            }
          ],
          "verifiers": []
        }
        """
        ),
    },
    {
        "filename": "parse_csv_task.json",
        "content": dedent(
            """\
        {
          "name": "Parse and Summarize CSV Data",
          "category": "task",
          "apps": ["coding"],
          "meta": {
            "task_id": "parse_csv_task",
            "display_name": "Parse and Summarize CSV Data",
            "difficulty": "hard"
          },
          "task": "Write `summarize_csv.py`: read CSV from fixtures, compute stats.",
          "tool_servers": [
            {
              "name": "coding-env",
              "tool_server_url": "http://localhost:8020"
            }
          ],
          "verifiers": []
        }
        """
        ),
    },
]

_CODING_VERIFIERS_INIT = dedent(
    """\
    \"\"\"Verifier modules for local custom coding tasks.\"\"\"
    """
)


def _get_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.load_all()
    return registry


def _extract_tools_from_scenario(
    registry: ToolRegistry, scenario: ScenarioSummary
) -> tuple[list[str], list[str]]:
    """Return (recognized_tools, missing_tools) from scenario.tool_servers[].name."""
    service_to_tool = {
        "coding-env": "coding",
        "email-env": "email",
        "chronos-server": "calendar",
        "frappe-hrms-env": "frappe-hrms",
        "google-workspace-tool-server": "google-workspace",
        "playwright-mcp": "playwright",
        "rocketchat-env": "rocketchat",
        "sec-edgar-env": "sec-edgar",
        "twelve-data-env": "twelve-data",
    }

    def _to_registry_tool_name(raw_name: str) -> str:
        name = raw_name.strip()
        if not name:
            return ""
        mapped = service_to_tool.get(name, name)
        if registry.get_tool(mapped) is not None:
            return mapped
        if mapped.endswith("-env"):
            env_trimmed = mapped[: -len("-env")]
            if registry.get_tool(env_trimmed) is not None:
                return env_trimmed
        return mapped

    requested_tools = []
    for entry in scenario.tool_servers:
        name = entry.name.strip()
        if name:
            requested_tools.append(_to_registry_tool_name(name))

    selected_tools = list(
        dict.fromkeys(name for name in requested_tools if registry.get_tool(name) is not None)
    )
    missing_tools = list(
        dict.fromkeys(name for name in requested_tools if registry.get_tool(name) is None)
    )
    return (selected_tools, missing_tools)


@click.group()
def env() -> None:
    """Manage Simlab environments."""


def env_init_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for env init when a Collinear API key is configured."""
    _ = args, kwargs
    return resolve_scenario_manager_capture_config(ctx)


def env_command_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for environment lifecycle commands with an API key."""
    _ = args
    return resolve_scenario_manager_capture_config(
        ctx,
        config_path=normalize_config_path(kwargs.get("config_path")),
    )


def _load_scenario_guidance_file(path_str: str) -> str:
    """Read scenario guidance markdown from a file path."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if path.is_dir():
        raise click.ClickException(
            f"Scenario guidance file must be a file, not a directory: {path}"
        )
    if not path.exists():
        raise click.ClickException(f"Scenario guidance file does not exist: {path}")
    return path.read_text(encoding="utf-8").strip()


@env.command()
@click.argument("env_name")
@click.option(
    "--template", "-t", "template_name", default=None, help="Start from a scenario preset."
)
@click.option("--non-interactive", is_flag=True, help="Skip interactive prompts.")
@click.option(
    "--registry",
    "-r",
    "image_registry",
    default=DEFAULT_IMAGE_REGISTRY,
    show_default=True,
    help="Container registry prefix for private images (e.g. ghcr.io/collinear-ai).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing environment without prompting.",
)
@click.option(
    "--scenario-guidance-file",
    "scenario_guidance_file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Read scenario guidance markdown from a file and store it in env.yaml.",
)
@click.pass_context
@with_command_telemetry("env init", resolver=env_init_capture_config)
def init(
    ctx: click.Context,
    env_name: str,
    template_name: str | None,
    non_interactive: bool,
    image_registry: str | None,
    force: bool,
    scenario_guidance_file: Path | None,
) -> None:
    """Initialize an environment and generate docker-compose.yml in one step."""
    global_cfg = get_global_config_from_ctx(ctx)
    env_dir = get_env_dir(env_name, ctx=ctx)
    env_yaml = env_dir / "env.yaml"
    registry = _get_registry()
    scenario_guidance_md = (
        _load_scenario_guidance_file(str(scenario_guidance_file))
        if scenario_guidance_file is not None
        else None
    )

    selected_tools: list[str] = []
    missing_tools: list[str] = []
    scenario: ScenarioSummary | None = None

    # Start from template if specified
    if template_name:
        base_url = resolve_scenario_manager_api_url(config=global_cfg)
        api_key = resolve_collinear_api_key(config=global_cfg)
        sm_client = ScenarioManagerClient(base_url=base_url, api_key=api_key)
        try:
            scenarios = sm_client.list_scenarios(include_hidden=True)
            backend_id = sm_client.resolve_template_to_backend_id(
                template_name,
                scenarios=scenarios,
            )
        except ScenarioManagerApiError as e:
            click.echo(click.style(str(e), fg="red"), err=True)
            raise SystemExit(1) from e

        scenario = next(
            (s for s in scenarios if s.scenario_id.strip() == backend_id),
            None,
        )
        if scenario is None:
            click.echo(
                click.style(
                    f"Template '{template_name}' resolved to '{backend_id}', "
                    "but no scenario was found.",
                    fg="red",
                ),
                err=True,
            )
            raise SystemExit(1)

        selected_tools, missing_tools = _extract_tools_from_scenario(registry, scenario)
        if not selected_tools:
            click.echo(
                click.style(
                    f"Template '{backend_id}' has no tool servers compatible "
                    "with this CLI registry.",
                    fg="red",
                ),
                err=True,
            )
            raise SystemExit(1)

        if missing_tools:
            click.echo(
                click.style(
                    "Ignoring unsupported tool servers from template "
                    f"'{backend_id}': {', '.join(missing_tools)}",
                    fg="yellow",
                )
            )

        template_name = backend_id
        click.echo(f"Starting from template '{backend_id}': {', '.join(selected_tools)}")

    if not non_interactive and not selected_tools:
        selected_tools = _interactive_select(registry)
    elif not non_interactive and selected_tools:
        remaining = [t for t in registry.tool_names if t not in selected_tools]
        if remaining:
            extra = _interactive_add_more(registry, remaining)
            selected_tools.extend(extra)

    # Require tool selection only when creating/overwriting env.yaml (not when regenerating from
    # existing)
    if not selected_tools and not (env_yaml.exists() and force):
        click.echo(click.style("No tools selected. Aborting.", fg="red"), err=True)
        raise SystemExit(1)

    # Don't silently overwrite unless --force
    if env_yaml.exists() and not force:
        if non_interactive:
            click.echo(
                click.style(f"Environment '{env_name}' already exists: {env_yaml}", fg="red"),
                err=True,
            )
            click.echo(
                "Delete it first, use --force to overwrite, or use a different env name.", err=True
            )
            raise SystemExit(1)
        if not click.confirm(
            click.style(f"Environment '{env_name}' already exists. Overwrite?", fg="yellow")
        ):
            raise SystemExit(0)

    env_dir.mkdir(parents=True, exist_ok=True)
    if scenario_guidance_file is None and template_name and scenario:
        scenario_guidance_md = scenario.scenario_guidance_md
    if env_yaml.exists() and force:
        # Regenerate docker-compose and .env from existing env.yaml; do not overwrite env.yaml
        data = yaml.safe_load(env_yaml.read_text()) or {}
        config = EnvConfig(**data)
        updates: dict[str, object] = {"name": env_name}
        if (
            "coding" in config.tools
            and not config.scenario_guidance_md
            and config.scenario_guidance_path
        ):
            guidance_path = Path(config.scenario_guidance_path).expanduser()
            if not guidance_path.is_absolute():
                guidance_path = env_yaml.parent / guidance_path
            updates["scenario_guidance_md"] = guidance_path.read_text(encoding="utf-8").strip()
            updates["scenario_guidance_path"] = None
        if scenario_guidance_file is not None:
            updates["scenario_guidance_md"] = scenario_guidance_md
        config = config.model_copy(update=updates)
        if scenario_guidance_file is not None:
            config_data = {k: v for k, v in config.model_dump().items() if v is not None}
            env_yaml.write_text(yaml.dump(config_data, default_flow_style=False, sort_keys=False))
        click.echo(click.style(f"\nRegenerating from existing {env_yaml}", fg="green"))
        if "coding" in config.tools:
            _scaffold_coding_environment(env_dir, env_name)
    else:
        config_kwargs: dict[str, Any] = {
            "name": env_name,
            "tools": selected_tools,
            "registry": image_registry,
            "template": template_name,
        }
        if "coding" in selected_tools:
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
            config_kwargs["scenario_guidance_md"] = (
                scenario_guidance_md or _DEFAULT_CODING_SCENARIO_GUIDANCE
            ).strip()
        elif scenario_guidance_md:
            config_kwargs["scenario_guidance_md"] = scenario_guidance_md

        config = EnvConfig(**config_kwargs)
        config_data = {k: v for k, v in config.model_dump().items() if v is not None}
        env_yaml.write_text(yaml.dump(config_data, default_flow_style=False, sort_keys=False))
        click.echo(click.style(f"\nConfig written to {env_yaml}", fg="green"))
        if "coding" in selected_tools:
            _scaffold_coding_environment(env_dir, env_name)

    # Generate docker-compose.yml and .env.
    engine = ComposeEngine(registry)
    try:
        compose_output = engine.compose(config, config_dir=env_yaml.parent)
    except (KeyError, FileNotFoundError, ValueError) as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1) from e

    write_output(compose_output, env_dir)

    click.echo(click.style(f"Environment generated in {env_dir}/", fg="green"))
    click.echo("  env.yaml")
    click.echo("  docker-compose.yml")
    click.echo("  .env")
    if "coding" in config.tools:
        click.echo("  README.md")
        click.echo("  coding/setup/install-tools.sh")
        click.echo("  coding/fixtures/")
        click.echo("  coding/skills/")
        click.echo("  task-bundle/")
    click.echo()

    click.echo(click.style("Tool endpoints:", bold=True))
    for tool_name, endpoint in compose_output.tool_endpoints.items():
        click.echo(f"  {tool_name}: {endpoint}")

    if compose_output.env_file.strip() != "# No environment variables required":
        click.echo()
        click.echo(click.style("Fill in required env vars in .env before starting.", fg="yellow"))
    if "coding" in config.tools:
        install_tools_path = env_dir / "coding" / "setup" / "install-tools.sh"
        fixtures_path = env_dir / "coding" / "fixtures"
        skills_path = env_dir / "coding" / "skills"
        task_bundle_path = env_dir / "task-bundle"
        click.echo()
        click.echo(click.style("Next edits for this coding environment:", bold=True))
        click.echo(f"  1. Edit {install_tools_path} to install custom tools")
        click.echo(f"  2. Drop sample files into {fixtures_path}")
        click.echo(f"  3. Add reusable skills under {skills_path}")
        click.echo(f"  4. Edit tasks and verifiers under {task_bundle_path}")
    click.echo()
    click.echo(f"Next: simlab env up {env_name}")
    emit_cli_event(
        "env_init_completed",
        {
            "template_used": bool(template_name),
            "selected_tool_count": len(selected_tools),
            "unsupported_tool_count": len(missing_tools),
            "non_interactive": non_interactive,
        },
    )


def _interactive_select(registry: ToolRegistry) -> list[str]:
    """Interactive tool selection with questionary."""
    if questionary is None:
        click.echo("Install questionary for interactive mode: pip install questionary", err=True)
        raise SystemExit(1) from None

    categories = registry.list_by_category()
    choices: list[Any] = []
    for category, tools in categories.items():
        choices.append(questionary.Separator(f"── {category.upper()} ──"))
        choices.extend(
            questionary.Choice(
                title=f"{tool.name:<20} {tool.description}",
                value=tool.name,
            )
            for tool in tools
        )

    selected = questionary.checkbox(
        "Select tools for your environment:",
        choices=choices,
    ).ask()

    return selected or []


def _write_text_if_missing(path: Path, content: str, *, executable: bool = False) -> None:
    """Create a file only when missing so user edits survive regeneration."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def _ensure_placeholder_dir(path: Path) -> None:
    """Create a directory and keep it visible in git-friendly trees."""
    path.mkdir(parents=True, exist_ok=True)
    _write_text_if_missing(path / ".gitkeep", "")


def _generate_coding_env_readme(env_name: str) -> str:
    """Return the generated README for coding-enabled environments."""
    return dedent(
        f"""\
        # {env_name}

        This environment includes the coding toolset and a local custom task bundle.

        ## Files You Edit

        - `coding/setup/install-tools.sh`: install additional CLI tools for the coding runtime
        - `coding/fixtures/`: files mounted into `/workspace/fixtures`
        - `coding/skills/`: reusable coding skills exposed as project skills
        - `task-bundle/tasks/`: local custom task definitions
        - `task-bundle/verifiers/`: verifier modules for local tasks

        ## Common Commands

        If this environment lives under your configured Simlab environments directory:

        ```bash
        simlab env up {env_name}

        simlab tasks run \\
          --env {env_name} \\
          --tasks-dir ./task-bundle \\
          --task example_task \\
          --agent-model <model>

        simlab env down {env_name}
        ```

        If the environment lives outside your default environments directory, add
        `--environments-dir <parent-dir>` to the command.
        """
    )


def _scaffold_coding_environment(env_dir: Path, env_name: str) -> None:
    """Create user-owned coding files for a newly initialized coding environment."""
    _write_text_if_missing(
        env_dir / "coding" / "setup" / "install-tools.sh",
        _CODING_SETUP_SCRIPT,
        executable=True,
    )
    _ensure_placeholder_dir(env_dir / "coding" / "fixtures")
    _write_text_if_missing(
        env_dir / "coding" / "skills" / "example-skill" / "SKILL.md",
        _CODING_SKILL_STUB,
    )
    _write_text_if_missing(
        env_dir / "task-bundle" / "README.md",
        _CODING_TASK_BUNDLE_README.format(env_name=env_name),
    )
    for sample in _CODING_SAMPLE_TASKS:
        _write_text_if_missing(
            env_dir / "task-bundle" / "tasks" / sample["filename"],
            sample["content"],
        )
    _write_text_if_missing(
        env_dir / "task-bundle" / "verifiers" / "__init__.py",
        _CODING_VERIFIERS_INIT,
    )
    _write_text_if_missing(env_dir / "README.md", _generate_coding_env_readme(env_name))


def _validate_daytona_coding_assets(config: EnvConfig, config_dir: Path) -> None:
    """Fail early when Daytona-backed coding envs reference files outside the env dir."""
    external_asset_paths = ComposeEngine.get_external_coding_asset_paths(config, config_dir)
    if not external_asset_paths:
        return

    click.echo(
        click.style(
            "Daytona mode only supports coding assets located inside the environment "
            f"directory ({config_dir}).",
            fg="red",
        ),
        err=True,
    )
    for path in external_asset_paths:
        click.echo(f"  - {path}", err=True)
    raise SystemExit(1)


def _interactive_add_more(registry: ToolRegistry, remaining: list[str]) -> list[str]:
    """Ask user if they want to add more tools beyond the template."""
    if questionary is None:
        return []

    add_more = questionary.confirm("Add more tools beyond the template?", default=False).ask()
    if not add_more:
        return []

    choices = []
    for name in remaining:
        tool = registry.get_tool(name)
        if tool:
            choices.append(
                questionary.Choice(
                    title=f"{tool.name:<20} {tool.description}",
                    value=tool.name,
                )
            )

    selected = questionary.checkbox("Select additional tools:", choices=choices).ask()
    return selected or []


def _compose_has_build_contexts(compose_file: Path) -> bool:
    """Return True if the compose file defines any service with a build context."""
    existing = yaml.safe_load(compose_file.read_text())
    if not existing or "services" not in existing:
        return False
    return any("build" in svc for svc in existing["services"].values())


@env.command()
@click.argument("env_name")
@click.option(
    "--rebuild",
    is_flag=True,
    help=(
        "Tear down containers and start them again (down then up). "
        "Does not regenerate docker-compose.yml."
    ),
)
@click.option(
    "--daytona", is_flag=True, help="Run on Daytona (remote sandbox) instead of local Docker."
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output for each step.")
@click.pass_context
@with_command_telemetry("env up", resolver=env_command_capture_config)
def up(ctx: click.Context, env_name: str, rebuild: bool, daytona: bool, verbose: bool) -> None:
    """Start the environment."""
    env_dir = resolve_env_dir(env_name, ctx=ctx)
    config_file = env_dir / "env.yaml"
    data = yaml.safe_load(config_file.read_text()) or {}
    config = EnvConfig(**data)
    compose_file = env_dir / "docker-compose.yml"

    if not compose_file.exists():
        click.echo(
            click.style(f"No docker-compose.yml found in {env_dir}", fg="red"),
            err=True,
        )
        click.echo(
            f"Run `simlab env init {env_name}` "
            f"(or `simlab env init {env_name} --force` to regenerate) first.",
            err=True,
        )
        raise SystemExit(1)

    has_builds = _compose_has_build_contexts(compose_file)

    if daytona:
        global_cfg = get_global_config_from_ctx(ctx)
        if rebuild:
            runner = _get_daytona_runner(daytona_api_key=global_cfg.daytona_api_key)
            state_file = env_dir / "daytona-state.json"
            if state_file.exists():
                runner.down(env_dir)
            else:
                click.echo("No Daytona state found; starting fresh.")
        _up_daytona(
            config,
            config_file,
            env_dir,
            daytona_api_key=global_cfg.daytona_api_key,
            verbose=verbose,
        )
    else:
        if rebuild:
            click.echo("Tearing down existing containers...")
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "down"],
                capture_output=True,
                text=True,
                cwd=env_dir,
            )
            if result.returncode != 0:
                click.echo(click.style("Failed to stop services:", fg="red"), err=True)
                click.echo(result.stderr, err=True)
                raise SystemExit(1)
        _up_local(compose_file, env_dir, config, config_file, has_builds, verbose=verbose)
    emit_cli_event(
        "env_up_completed",
        {
            "mode": "daytona" if daytona else "local",
            "tool_count": len(config.tools),
            "rebuild": rebuild,
            "has_builds": has_builds,
            "preseed_service_count": len(_get_preseed_service_names(config, config_file)),
            "seed_service_count": len(_get_seed_service_names(config, config_file)),
        },
    )


def _up_local(
    compose_file: Path,
    out_dir: Path,
    config: EnvConfig,
    config_path: Path | None = None,
    has_builds: bool = False,
    verbose: bool = False,
) -> None:
    """Start services locally via docker compose."""
    progress = StepProgress(verbose=verbose)
    t0 = time.time()

    preseed_svc_names = _get_preseed_service_names(config, config_path)
    if preseed_svc_names:
        with progress.step("Environment preseeded") as ctx:
            _run_profiled_services_local(
                out_dir, preseed_svc_names, profile="preseed", quiet=True, step_ctx=ctx
            )

    with progress.step("Services started") as ctx:
        up_cmd = ["docker", "compose", "-f", str(compose_file), "up", "-d"]
        if has_builds:
            up_cmd.append("--build")
        result = subprocess.run(up_cmd, capture_output=True, text=True)
        if result.stdout:
            ctx.detail(result.stdout.strip())
        if result.returncode != 0:
            ctx.detail(result.stderr.strip() if result.stderr else "")
            raise SystemExit(1)

    # Health polling has its own animated display — run it outside step()
    click.echo()
    _poll_health(
        _local_health_fetcher(compose_file),
        timeout=180,
    )

    # Auto-seed if tools have seed services
    seed_svc_names = _get_seed_service_names(config, config_path)
    if seed_svc_names:
        with progress.step("Environment seeded") as ctx:
            _run_profiled_services_local(
                out_dir, seed_svc_names, profile="seed", quiet=True, step_ctx=ctx
            )
        _verify_seed_local(config, config_path)

    tool_ports = _get_tool_ports(config, config_path)
    endpoints = {name: f"http://localhost:{port}" for name, port in tool_ports.items()}
    progress.finish(time.time() - t0, endpoints=endpoints or None)


def _get_daytona_runner(daytona_api_key: str | None = None):  # noqa: ANN202
    """Import Daytona runner lazily so base CLI works without optional deps."""
    try:
        daytona_runner_module = import_module("simlab.runtime.daytona_runner")
    except ModuleNotFoundError as exc:
        click.echo(
            click.style(
                "Daytona support is unavailable in this installation. "
                "Install simulationlab[daytona] or run without --daytona.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1) from exc

    return daytona_runner_module.DaytonaRunner(daytona_api_key=daytona_api_key)


def _get_tool_ports(config: EnvConfig, config_path: Path | None = None) -> dict[str, int]:
    """Get tool server ports keyed by tool name from loaded catalog entries."""
    _ = config_path
    registry = _get_registry()
    ports: dict[str, int] = {}
    for tool_name in config.tools:
        tool = registry.get_tool(tool_name)
        if tool and tool.tool_server_port is not None:
            ports[tool_name] = tool.tool_server_port
    return ports


def _verify_seed_daytona(config: EnvConfig, endpoints: dict[str, str], config_path: Path) -> None:
    """Verify seed data by querying Daytona-exposed tool endpoints."""
    _ = config_path
    registry = _get_registry()
    for tool_name in config.tools:
        tool = registry.get_tool(tool_name)
        if not tool or not tool.seed_services:
            continue
        if tool.name == "frappe-hrms":
            url = endpoints.get(tool_name)
            if not url:
                click.echo(
                    click.style(
                        "Could not resolve Daytona endpoint for frappe-hrms.",
                        fg="yellow",
                    ),
                    err=True,
                )
                continue
            _print_frappe_verification(url)


def _up_daytona(
    config: EnvConfig,
    config_path: Path,
    out_dir: Path,
    daytona_api_key: str | None = None,
    verbose: bool = False,
) -> None:
    """Start services on a remote Daytona sandbox."""
    _validate_daytona_coding_assets(config, config_path.parent)
    progress = StepProgress(verbose=verbose)
    reporter = StepProgressReporter(progress)
    t0 = time.time()

    runner = _get_daytona_runner(daytona_api_key=daytona_api_key)
    tool_ports = _get_tool_ports(config, config_path)
    preseed_svc_names = _get_preseed_service_names(config, config_path)
    endpoints = runner.up(
        out_dir, tool_ports, preseed_svc_names=preseed_svc_names, reporter=reporter
    )

    # Health polling has its own animated display
    click.echo()
    _poll_health(
        lambda: runner.get_health(out_dir),
        timeout=180,
    )

    seed_svc_names = _get_seed_service_names(config, config_path)
    if seed_svc_names:
        with progress.step("Environment seeded") as ctx:
            runner.seed(out_dir, seed_svc_names)
            _ = ctx  # available for future detail messages
        _verify_seed_daytona(config, endpoints, config_path)

    progress.finish(time.time() - t0, endpoints=endpoints or None)


@env.command()
@click.argument("env_name")
@click.option("--daytona", is_flag=True, help="Tear down Daytona sandbox instead of local Docker.")
@click.pass_context
@with_command_telemetry("env down", resolver=env_command_capture_config)
def down(ctx: click.Context, env_name: str, daytona: bool) -> None:
    """Stop and remove the environment."""
    env_dir = resolve_env_dir(env_name, ctx=ctx)

    if daytona:
        global_cfg = get_global_config_from_ctx(ctx)
        runner = _get_daytona_runner(daytona_api_key=global_cfg.daytona_api_key)
        runner.down(env_dir)
        emit_cli_event(
            "env_down_completed",
            {
                "mode": "daytona",
                "config_provided": False,
            },
        )
        return

    compose_file = env_dir / "docker-compose.yml"
    if not compose_file.exists():
        click.echo(click.style(f"No compose file found at {compose_file}", fg="red"), err=True)
        raise SystemExit(1)

    click.echo("Stopping services...")
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "down"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(click.style("Failed to stop services:", fg="red"), err=True)
        click.echo(result.stderr, err=True)
        raise SystemExit(1)

    click.echo(click.style("Environment stopped.", fg="green"))
    emit_cli_event(
        "env_down_completed",
        {
            "mode": "local",
            "config_provided": False,
        },
    )


@env.command()
@click.argument("env_name")
@click.option("--daytona", is_flag=True, help="Run seed on Daytona sandbox.")
@click.option("--verify-only", is_flag=True, help="Only show seed data counts, don't re-seed.")
@click.pass_context
@with_command_telemetry("env seed", resolver=env_command_capture_config)
def seed(ctx: click.Context, env_name: str, daytona: bool, verify_only: bool) -> None:
    """Seed the environment with initial data."""
    env_dir = resolve_env_dir(env_name, ctx=ctx)
    config_file = env_dir / "env.yaml"
    data = yaml.safe_load(config_file.read_text()) or {}
    config = EnvConfig(**data)

    seed_svc_names = _get_seed_service_names(config, config_file)
    if not seed_svc_names:
        click.echo("No seed services configured for this environment.")
        emit_cli_event(
            "env_seed_completed",
            {
                "mode": "daytona" if daytona else "local",
                "verify_only": verify_only,
                "seed_service_count": 0,
                "tool_count": len(config.tools),
            },
        )
        return

    if daytona:
        global_cfg = get_global_config_from_ctx(ctx)
        runner = _get_daytona_runner(daytona_api_key=global_cfg.daytona_api_key)
        if not verify_only:
            runner.seed(env_dir, seed_svc_names)
        endpoints = runner.get_urls(env_dir, _get_tool_ports(config, config_file))
        _verify_seed_daytona(config, endpoints, config_file)
    elif verify_only:
        _verify_seed_local(config, config_file)
    else:
        _seed_local(env_dir, seed_svc_names)
        _verify_seed_local(config, config_file)
    emit_cli_event(
        "env_seed_completed",
        {
            "mode": "daytona" if daytona else "local",
            "verify_only": verify_only,
            "seed_service_count": len(seed_svc_names),
            "tool_count": len(config.tools),
        },
    )


def _get_seed_service_names(config: EnvConfig, config_path: Path | None = None) -> list[str]:
    """Get seed service names from the tool definitions in config."""
    return _get_profiled_service_names(config, profile="seed", config_path=config_path)


def _get_preseed_service_names(config: EnvConfig, config_path: Path | None = None) -> list[str]:
    """Get preseed service names from the tool definitions in config."""
    return _get_profiled_service_names(config, profile="preseed", config_path=config_path)


def _get_profiled_service_names(
    config: EnvConfig,
    profile: str,
    config_path: Path | None = None,
    tool_names: list[str] | None = None,
) -> list[str]:
    """Get profiled service names from the tool definitions in config."""
    _ = config_path
    registry = _get_registry()
    names: list[str] = []
    for tool_name in tool_names or config.tools:
        tool = registry.get_tool(tool_name)
        if not tool:
            continue
        service_defs = tool.preseed_services if profile == "preseed" else tool.seed_services
        if service_defs:
            names.extend(service_defs.keys())
    return names


def _seed_local(out_dir: Path, seed_svc_names: list[str]) -> None:
    """Run seed containers locally via docker compose."""
    _run_profiled_services_local(out_dir, seed_svc_names, profile="seed")


def _run_profiled_services_local(
    out_dir: Path,
    svc_names: list[str],
    profile: str,
    env_overrides: dict[str, str] | None = None,
    quiet: bool = False,
    step_ctx: object | None = None,
) -> None:
    """Run profiled containers locally via docker compose.

    When *quiet* is True, output is routed to *step_ctx* (if provided) instead
    of being printed directly via ``click.echo``.
    """
    compose_file = out_dir / "docker-compose.yml"
    if not compose_file.exists():
        click.echo(
            click.style(f"No compose file at {compose_file}", fg="red"),
            err=True,
        )
        raise SystemExit(1)

    phase_label = "Preseeding" if profile == "preseed" else "Seeding"
    ctx: StepContext | None = step_ctx if isinstance(step_ctx, StepContext) else None

    def _echo(msg: str) -> None:
        if quiet and ctx is not None:
            ctx.detail(msg)
        elif not quiet:
            click.echo(msg)

    for svc_name in svc_names:
        _echo(f"{phase_label}: {svc_name}...")
        override_args: list[str] = []
        for key, value in (env_overrides or {}).items():
            override_args.extend(["-e", f"{key}={value}"])
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "--profile",
                profile,
                "run",
                "--rm",
                *override_args,
                svc_name,
            ],
            capture_output=quiet,
            text=True,
        )
        if result.returncode != 0:
            msg = f"{phase_label[:-3]} service '{svc_name}' failed (exit {result.returncode})."
            if quiet and result.stderr:
                _echo(result.stderr.strip())
            click.echo(click.style(msg, fg="red"), err=True)
            raise SystemExit(1)
        if quiet and result.stdout:
            _echo(result.stdout.strip())
        if env_overrides:
            rendered = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in env_overrides.items()
            )
            _echo(f"  Applied overrides: {rendered}")

    if not quiet:
        click.echo(click.style(f"\n{phase_label} complete.", fg="green"))


def _query_tool_server(
    url: str,
    tool_name: str,
    parameters: dict[str, Any],
) -> Any:
    """Execute a tool action against a tool server."""
    return query_tool_server(url, tool_name, parameters)


def _verify_seed_local(config: EnvConfig, config_path: Path | None = None) -> None:
    """Verify seed data by querying the frappe tool server."""
    _ = config_path
    registry = _get_registry()
    for tool_name in config.tools:
        tool = registry.get_tool(tool_name)
        if not tool or not tool.seed_services:
            continue
        if tool.name == "frappe-hrms":
            url = f"http://localhost:{tool.tool_server_port}"
            _print_frappe_verification(url)


_DOC_PREFIXES = {
    "Employee Records": "Employee Record: ",
    "Health Enrollments": "Health Enrollment: ",
    "Job Requisitions": "Job Requisition: ",
    "NPC Personas": "NPC Persona: ",
    "Candidate Applications": "Candidate Application: ",
}


def _categorize_docs(titles: list[str]) -> dict[str, list[str]]:
    """Categorize document titles by known prefix into buckets."""
    categories: dict[str, list[str]] = {k: [] for k in _DOC_PREFIXES}
    categories["Policies"] = []
    for title in titles:
        matched = False
        for cat, prefix in _DOC_PREFIXES.items():
            if title.startswith(prefix):
                categories[cat].append(title)
                matched = True
                break
        if not matched:
            categories["Policies"].append(title)
    return categories


def _extract_titles(resp: Any) -> list[str]:
    """Extract a list of title strings from a tool server response."""
    records = _extract_samples_all(resp)
    return [
        r.get("title") or r.get("name", "")
        for r in records
        if isinstance(r, dict) and (r.get("title") or r.get("name"))
    ]


def _extract_samples_all(resp: Any) -> list[dict[str, Any]]:
    """Extract all records (not capped) from a tool server response."""
    if resp is None:
        return []
    obs = resp.get("observation") if isinstance(resp, dict) else None
    if obs is None:
        return []
    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except json.JSONDecodeError:
            return []
    if isinstance(obs, list):
        return obs
    if isinstance(obs, dict):
        text = obs.get("text")
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, list):
                    return parsed
        if "data" in obs:
            data = obs["data"]
            if isinstance(data, list):
                return data
    return []


def _print_frappe_verification(url: str) -> None:
    """Query frappe tool server and print verification output."""
    click.echo(click.style("\nSeed verification:\n", bold=True))

    # 1. Employee count
    emp_resp = _query_tool_server(
        url,
        "frappe_list_resource",
        {"doctype": "Employee", "limit_page_length": 0},
    )
    emp_count = _extract_count(emp_resp)

    # 2. Employee samples (random 3)
    emp_sample_resp = _query_tool_server(
        url,
        "frappe_list_resource",
        {
            "doctype": "Employee",
            "fields": ["employee_name", "designation"],
            "limit_page_length": 0,
        },
    )
    all_employees = _extract_samples_all(emp_sample_resp)
    emp_samples = random.sample(all_employees, min(3, len(all_employees)))

    # 3. All doc titles (Wiki Page or Note)
    doc_titles: list[str] = []
    for dt in ("Wiki Page", "Note"):
        title_resp = _query_tool_server(
            url,
            "frappe_list_resource",
            {"doctype": dt, "fields": ["title"], "limit_page_length": 0},
        )
        doc_titles = _extract_titles(title_resp)
        if doc_titles:
            break

    categories = _categorize_docs(doc_titles)

    # -- Employees --
    if emp_count is not None:
        click.echo(f"  Employees:           {emp_count} loaded")
    else:
        click.echo(click.style("  Employees:           could not verify", fg="yellow"))

    if emp_samples:
        click.echo("\n  Sample employees:")
        for s in emp_samples:
            name = s.get("employee_name", "?")
            desig = s.get("designation", "")
            click.echo(f"    {name:<25} {desig}")

    # -- Documents breakdown --
    total_docs = sum(len(v) for v in categories.values())
    if total_docs > 0:
        click.echo(f"\n  Documents:           {total_docs} total")
        for cat_name in [*_DOC_PREFIXES.keys(), "Policies"]:
            items = categories.get(cat_name, [])
            if items:
                click.echo(f"    {cat_name + ':':<23}{len(items)}")

        # Sample documents (1 random per category)
        click.echo("\n  Sample documents:")
        for cat_name in [*_DOC_PREFIXES.keys(), "Policies"]:
            items = categories.get(cat_name, [])
            if items:
                click.echo(f"    {random.choice(items)}")
    else:
        click.echo(click.style("\n  Documents:           could not verify", fg="yellow"))

    click.echo()


def _extract_count(resp: Any) -> int | None:
    """Extract a record count from a tool server response."""
    if resp is None:
        return None
    obs = resp.get("observation") if isinstance(resp, dict) else None
    if obs is None:
        return None
    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except json.JSONDecodeError:
            return None
    if isinstance(obs, list):
        return len(obs)
    if isinstance(obs, dict):
        # Tool server wraps data in {"text": "<json-string>", ...}
        text = obs.get("text")
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, list):
                    return len(parsed)
        if "data" in obs:
            data = obs["data"]
            if isinstance(data, list):
                return len(data)
    return None


def _extract_samples(resp: Any) -> list[dict[str, Any]]:
    """Extract sample records from a tool server response."""
    if resp is None:
        return []
    obs = resp.get("observation") if isinstance(resp, dict) else None
    if obs is None:
        return []
    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except json.JSONDecodeError:
            return []
    if isinstance(obs, list):
        return obs[:3]
    if isinstance(obs, dict):
        # Tool server wraps data in {"text": "<json-string>", ...}
        text = obs.get("text")
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, list):
                    return parsed[:3]
        if "data" in obs:
            data = obs["data"]
            if isinstance(data, list):
                return data[:3]
    return []


# ---------------------------------------------------------------------------
# Health polling with animated status display
# ---------------------------------------------------------------------------

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_WAITING_MESSAGES = [
    "Warming up containers...",
    "Waking up the databases...",
    "Convincing MongoDB to cooperate...",
    "Brewing some coffee while we wait...",
    "Teaching containers to talk to each other...",
    "Reticulating splines...",
    "Aligning the bits...",
    "Herding microservices...",
    "Almost there, probably...",
    "Good things come to those who wait...",
    "Negotiating with Docker...",
    "Spinning up your workspace...",
    "Connecting the dots...",
    "Loading the good stuff...",
    "Building something beautiful...",
]


def _local_health_fetcher(compose_file: Path) -> Callable[[], dict[str, str]]:
    """Return a callable that fetches health from local docker compose."""

    def _fetch() -> dict[str, str]:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "ps",
                "--format",
                "{{.Name}}\t{{.Status}}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {}
        return parse_ps_output(result.stdout)

    return _fetch


def _render_status_line(
    services: dict[str, str],
    spinner_frame: str,
    message: str,
    elapsed: float,
) -> str:
    """Render a single status line with spinner, message, and service counts."""
    ready = sum(1 for status in services.values() if status in ("healthy", "running"))
    total = len(services)
    failed = sum(1 for status in services.values() if status in ("exited", "unhealthy"))

    bar_width = 20
    filled = int(bar_width * ready / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)

    elapsed_str = f"{int(elapsed)}s"

    parts = [
        click.style(f" {spinner_frame} ", fg="cyan"),
        click.style(message, fg="white"),
        "  ",
        click.style("[", fg="white"),
        click.style(bar[:filled], fg="green"),
        click.style(bar[filled:], fg="bright_black"),
        click.style("]", fg="white"),
        click.style(f" {ready}/{total}", fg="green" if ready == total else "yellow"),
        click.style(f"  {elapsed_str}", fg="bright_black"),
    ]
    if failed:
        parts.append(click.style(f"  {failed} failed", fg="red"))

    return "".join(parts)


def _render_service_detail(
    services: dict[str, str],
) -> list[str]:
    """Render per-service status lines."""
    lines = []
    icons = {
        "healthy": ("✓", "green"),
        "running": ("●", "green"),
        "starting": ("◌", "yellow"),
        "unhealthy": ("✗", "red"),
        "exited": ("✗", "red"),
        "unknown": ("?", "bright_black"),
    }
    for name, status in sorted(services.items()):
        icon, color = icons.get(status, ("?", "white"))
        lines.append(f"   {click.style(icon, fg=color)} {name:<30} {click.style(status, fg=color)}")
    return lines


def _clear_lines(n: int) -> None:
    """Move cursor up n lines and clear them."""
    for _ in range(n):
        sys.stdout.write("\033[A\033[2K")
    sys.stdout.flush()


def _poll_health(
    health_fetcher: Callable[[], dict[str, str]],
    timeout: int = 180,
) -> None:
    """Poll services with an animated status display.

    Args:
        health_fetcher: Callable returning {service: status} dict.
        timeout: Max seconds to wait.

    """
    start = time.time()
    spinner = itertools.cycle(_SPINNER_FRAMES)
    msg_index = 0
    current_message = _WAITING_MESSAGES[0]
    last_msg_change = start
    msg_interval = 3.0
    lines_printed = 0
    poll_interval = 2.0

    shuffled_messages = list(_WAITING_MESSAGES)
    random.shuffle(shuffled_messages)

    while time.time() - start < timeout:
        elapsed = time.time() - start
        services = health_fetcher()

        if time.time() - last_msg_change > msg_interval:
            msg_index = (msg_index + 1) % len(shuffled_messages)
            current_message = shuffled_messages[msg_index]
            last_msg_change = time.time()

        if lines_printed > 0:
            _clear_lines(lines_printed)

        frame = next(spinner)
        status_line = _render_status_line(
            services,
            frame,
            current_message,
            elapsed,
        )
        detail_lines = _render_service_detail(services)

        output_lines = [status_line, "", *detail_lines, ""]
        for line in output_lines:
            click.echo(line)
        lines_printed = len(output_lines)

        if services:
            healthy_or_running = sum(
                1 for status in services.values() if status in ("healthy", "running")
            )
            failed = sum(1 for status in services.values() if status in ("exited", "unhealthy"))
            total = len(services)

            if failed:
                _clear_lines(lines_printed)
                click.echo(
                    click.style(
                        " ✗ One or more services failed during startup.",
                        fg="red",
                        bold=True,
                    )
                )
                click.echo()
                for line in _render_service_detail(services):
                    click.echo(line)
                click.echo()
                raise SystemExit(1)

            if healthy_or_running == total and total > 0:
                _clear_lines(lines_printed)
                click.echo(click.style(" ✓ All services are healthy!", fg="green", bold=True))
                click.echo()
                for line in _render_service_detail(services):
                    click.echo(line)
                click.echo()
                return

        time.sleep(poll_interval)

    click.echo()
    click.echo(
        click.style(" ✗ Timed out waiting for services to become healthy.", fg="red", bold=True)
    )
    click.echo()
    services = health_fetcher()
    for line in _render_service_detail(services):
        click.echo(line)
    click.echo()
    raise SystemExit(1)
