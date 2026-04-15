"""CLI commands for environment management — init, up, down, seed, list, delete."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
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
from simlab.cli.progress import StepProgress
from simlab.composer.engine import DEFAULT_IMAGE_REGISTRY
from simlab.composer.engine import ComposeEngine
from simlab.composer.engine import EnvConfig
from simlab.composer.engine import get_mcp_gateway_host_port
from simlab.config import get_env_dir
from simlab.config import get_environments_dir
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_env_dir
from simlab.env_artifacts import ensure_env_artifacts_current
from simlab.env_artifacts import load_env_config
from simlab.env_artifacts import regenerate_env_artifacts
from simlab.env_custom_tools import add_custom_tool
from simlab.env_registry import build_registry
from simlab.mcp_config import MCP_SERVERS_FILENAME
from simlab.mcp_config import get_mcp_command_servers
from simlab.mcp_config import load_mcp_servers_config
from simlab.mcp_config import load_mcp_servers_from_env_dir
from simlab.mcp_config import validate_mcp_server_name_conflicts
from simlab.runtime.env_lifecycle import _compose_has_build_contexts
from simlab.runtime.env_lifecycle import _compose_has_services
from simlab.runtime.env_lifecycle import _get_daytona_runner
from simlab.runtime.env_lifecycle import _get_preseed_service_names
from simlab.runtime.env_lifecycle import _get_seed_service_names
from simlab.runtime.env_lifecycle import _get_tool_ports
from simlab.runtime.env_lifecycle import _run_profiled_services_local
from simlab.runtime.env_lifecycle import _seed_local
from simlab.runtime.env_lifecycle import _verify_seed_daytona
from simlab.runtime.env_lifecycle import _verify_seed_local
from simlab.runtime.env_lifecycle import detect_env_status
from simlab.runtime.env_lifecycle import ensure_env_started_daytona
from simlab.runtime.env_lifecycle import ensure_env_started_local
from simlab.runtime.env_lifecycle import env_down_daytona
from simlab.runtime.env_lifecycle import env_down_local
from simlab.runtime.env_lifecycle import env_purge_docker_local
from simlab.runtime.env_lifecycle import get_env_created_date
from simlab.runtime.env_lifecycle import has_any_running_containers
from simlab.runtime.env_lifecycle import run_env_seed_daytona
from simlab.runtime.env_lifecycle import run_env_seed_local
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
    simlab tasks run \
      --env <env-name> \
      --tasks-dir ./task-bundle \
      --task example_task \
      --agent-model <model>
    ```

    `tasks run` automatically starts the environment if it is not already running.

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

_CODING_SAMPLE_VERIFIERS = [
    {
        "filename": "custom_coding.py",
        "content": dedent(
            """\
            \"\"\"Verifier for the example custom coding task.\"\"\"

            from __future__ import annotations

            import json
            import re
            import urllib.request
            from typing import Any
            from typing import Protocol


            class _RunArtifactsLike(Protocol):
                tool_server_url: str | None

                def server_url(self, name: str) -> str | None: ...


            def _call_tool(
                tool_server_url: str,
                tool_name: str,
                parameters: dict[str, Any],
            ) -> dict[str, Any]:
                payload = json.dumps(
                    {"action": {"tool_name": tool_name, "parameters": parameters}}
                ).encode()
                request = urllib.request.Request(  # noqa: S310
                    f"{tool_server_url}/step",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                    return json.loads(response.read().decode("utf-8"))


            def verify(run_artifacts: _RunArtifactsLike) -> tuple[bool, str]:
                \"\"\"Verify the example task wrote a plausible workspace summary.\"\"\"
                tool_server_url = (
                    run_artifacts.server_url("coding-env") or run_artifacts.tool_server_url
                )
                if not tool_server_url:
                    return False, "coding-env tool server URL was not available to the verifier."

                report_result = _call_tool(
                    tool_server_url,
                    "read_file",
                    {"path": "workspace_summary.md"},
                )
                observation = report_result.get("observation", {})
                if observation.get("is_error"):
                    return False, f"workspace_summary.md missing: {observation.get('text', '')}"

                summary_text = str(observation.get("text", "")).lower()
                if len(summary_text.strip()) < 40:
                    return (
                        False,
                        "workspace_summary.md exists but is too short to be a useful summary.",
                    )

                expected_tools = ("read_file", "write_file", "list_dir", "run_command")
                mentioned_tools = [
                    tool_name
                    for tool_name in expected_tools
                    if re.search(rf"\\b{tool_name}\\b", summary_text)
                ]
                if len(mentioned_tools) < 2:
                    return (
                        False,
                        "workspace_summary.md should mention at least two available tools such as "
                        "read_file, write_file, list_dir, or run_command.",
                    )

                return True, "workspace_summary.md describes the available tools."
            """
        ),
    }
]


def _extract_tools_from_scenario(
    registry: ToolRegistry, scenario: ScenarioSummary
) -> tuple[list[str], list[str]]:
    """Return (recognized_tools, missing_tools) from scenario.tool_servers[].name."""
    service_to_tool = {
        "coding-env": "coding",
        "crm-env": "crm",
        "email-env": "email",
        "erp-env": "erp",
        "chronos-server": "calendar",
        "frappe-hrms-env": "frappe-hrms",
        "google-workspace-tool-server": "google-workspace",
        "playwright-mcp": "playwright",
        "project-management-env": "project-management",
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
@click.option(
    "--mcp-servers",
    "mcp_servers_path",
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
    default=None,
    help="Path to JSON file describing MCP servers (mcpServers with url or command/args/env).",
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
    mcp_servers_path: Path | None,
) -> None:
    """Initialize an environment and generate docker-compose.yml in one step."""
    global_cfg = get_global_config_from_ctx(ctx)
    env_dir = get_env_dir(env_name, ctx=ctx)
    env_yaml = env_dir / "env.yaml"
    registry = build_registry()
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
        if mcp_servers_path is not None and not click.confirm(
            "Add catalog tools in addition to the MCP servers?",
            default=False,
        ):
            selected_tools = []
        else:
            selected_tools = _interactive_select(registry)
    elif not non_interactive and selected_tools:
        remaining = [t for t in registry.tool_names if t not in selected_tools]
        if remaining:
            extra = _interactive_add_more(registry, remaining)
            selected_tools.extend(extra)

    # Require tool selection or MCP servers when creating/overwriting env.yaml
    # (not when regenerating)
    if not selected_tools and not mcp_servers_path and not (env_yaml.exists() and force):
        click.echo(
            click.style("No tools selected and no --mcp-servers. Aborting.", fg="red"), err=True
        )
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

    mcp_file = env_dir / MCP_SERVERS_FILENAME
    should_clear_persisted_mcp = mcp_servers_path is None

    # Load and persist MCP servers config if provided
    if mcp_servers_path is not None:
        try:
            mcp_config = load_mcp_servers_config(mcp_servers_path)
            validate_mcp_server_name_conflicts(
                mcp_config,
                existing_tool_names=frozenset(build_registry(env_dir=env_dir).tool_names),
            )
        except (TypeError, ValueError) as e:
            click.echo(click.style(str(e), fg="red"), err=True)
            raise SystemExit(1) from e
        mcp_file.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
        click.echo(click.style(f"MCP servers config written to {mcp_file}", fg="green"))
    elif should_clear_persisted_mcp and mcp_file.exists():
        mcp_file.unlink()
        click.echo(click.style(f"Removed persisted MCP servers config at {mcp_file}", fg="yellow"))

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

    try:
        compose_output = regenerate_env_artifacts(env_dir)
    except (KeyError, FileNotFoundError, ValueError) as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1) from e

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
    if (env_dir / MCP_SERVERS_FILENAME).is_file():
        click.echo(f"  {MCP_SERVERS_FILENAME}")
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
    click.echo(click.style(f"Next: simlab tasks list --env {env_name}", bold=True), err=True)
    click.echo(
        click.style(
            f"  Then: simlab tasks run --env {env_name} --task <task_id> --agent-model <model>",
            dim=True,
        ),
        err=True,
    )
    emit_cli_event(
        "env_init_completed",
        {
            "template_used": bool(config.template),
            "template_name": config.template,
            "environment_name": config.name,
            "selected_tool_count": len(config.tools),
            "selected_tools": list(config.tools),
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

    if selected is None:
        raise click.Abort
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
        simlab tasks run \\
          --env {env_name} \\
          --tasks-dir ./task-bundle \\
          --task example_task \\
          --agent-model <model>
        ```

        `tasks run` automatically starts and tears down the environment.

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
    for sample in _CODING_SAMPLE_VERIFIERS:
        _write_text_if_missing(
            env_dir / "task-bundle" / "verifiers" / sample["filename"],
            sample["content"],
        )
    _write_text_if_missing(env_dir / "README.md", _generate_coding_env_readme(env_name))


def _interactive_add_more(registry: ToolRegistry, remaining: list[str]) -> list[str]:
    """Ask user if they want to add more tools beyond the template."""
    if questionary is None:
        return []

    add_more = questionary.confirm("Add more tools beyond the template?", default=False).ask()
    if add_more is None:
        raise click.Abort
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
    if selected is None:
        raise click.Abort
    return selected or []


@env.group("custom-tools")
def custom_tools() -> None:
    """Manage env-local custom tool definitions."""


@custom_tools.command("add")
@click.argument("env_name")
@click.argument("name")
@click.option("--force", is_flag=True, help="Overwrite the scaffold if it already exists.")
@click.pass_context
def custom_tools_add(ctx: click.Context, env_name: str, name: str, force: bool) -> None:
    """Scaffold one env-local custom tool and enable it in the environment config."""
    env_dir = resolve_env_dir(env_name, ctx=ctx)
    try:
        result = add_custom_tool(env_dir, name, force=force)
    except FileExistsError as e:
        click.echo(click.style(str(e), fg="red"), err=True)
        click.echo("Use --force to overwrite it.", err=True)
        raise SystemExit(1) from e
    except (KeyError, FileNotFoundError, ValueError) as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1) from e

    click.echo(click.style(f"Custom tool scaffold written to {result.tool_file}", fg="green"))
    click.echo(f"Enabled '{name}' in {result.env_yaml}.")
    click.echo("Edit the scaffold before using this tool in a real run.")


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
    ensure_env_artifacts_current(env_dir, action_label="env up")
    config_file = env_dir / "env.yaml"
    config = load_env_config(env_dir)
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
    has_services = _compose_has_services(compose_file)

    if daytona:
        if not has_services:
            click.echo("No Daytona services defined; environment uses external endpoints only.")
            click.echo(click.style("Environment ready.", fg="green", bold=True))
        else:
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
        if rebuild and has_services:
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
        _up_local(
            env_dir,
            config,
            config_file,
            verbose=verbose,
            has_services=has_services,
        )
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
    out_dir: Path,
    config: EnvConfig,
    config_path: Path | None = None,
    verbose: bool = False,
    has_services: bool = True,
) -> None:
    """Start services locally via docker compose."""
    progress = StepProgress(verbose=verbose)
    t0 = time.time()

    if not has_services:
        # Preseed may still exist for external-only envs (e.g. schema setup).
        preseed_svc_names = _get_preseed_service_names(config, config_path)
        if preseed_svc_names:
            with progress.step("Environment preseeded") as ctx:
                _run_profiled_services_local(
                    out_dir, preseed_svc_names, profile="preseed", quiet=True, step_ctx=ctx
                )
        click.echo("No local services defined; environment uses external endpoints only.")
        click.echo(click.style("Environment ready.", fg="green", bold=True))
        return

    with progress.step("Services started") as ctx:
        ensure_env_started_local(out_dir, config, config_path)
        _ = ctx  # available for future detail messages

    with progress.step("Environment seeded") as ctx:
        run_env_seed_local(out_dir, config, config_path)
        _ = ctx

    tool_ports = _get_tool_ports(config, config_path)
    endpoints = {name: f"http://localhost:{port}" for name, port in tool_ports.items()}
    endpoints = _add_mcp_gateway_endpoint(endpoints, env_dir=out_dir)
    progress.finish(time.time() - t0, endpoints=endpoints or None)


def _add_mcp_gateway_endpoint(
    endpoints: dict[str, str],
    *,
    env_dir: Path,
) -> dict[str, str]:
    """Add the MCP gateway endpoint when the env has command-based MCP servers."""
    mcp_config = load_mcp_servers_from_env_dir(env_dir)
    if not mcp_config or not get_mcp_command_servers(mcp_config):
        return endpoints

    gateway_port = get_mcp_gateway_host_port(env_dir)
    return {
        **endpoints,
        ComposeEngine.MCP_GATEWAY_SERVICE_NAME: f"http://localhost:{gateway_port}/mcp",
    }


def _up_daytona(
    config: EnvConfig,
    config_path: Path,
    out_dir: Path,
    daytona_api_key: str | None = None,
    verbose: bool = False,
) -> None:
    """Start services on a remote Daytona sandbox."""
    progress = StepProgress(verbose=verbose)
    t0 = time.time()

    endpoints = ensure_env_started_daytona(
        out_dir,
        config,
        config_path,
        daytona_api_key=daytona_api_key,
        verbose=verbose,
        progress=progress,
    )

    with progress.step("Environment seeded") as ctx:
        run_env_seed_daytona(
            out_dir,
            config,
            config_path,
            daytona_api_key=daytona_api_key,
            endpoints=endpoints,
        )
        _ = ctx

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
        env_down_daytona(env_dir, daytona_api_key=global_cfg.daytona_api_key)
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

    if not _compose_has_services(compose_file):
        click.echo("No local services defined in current compose file; attempting teardown anyway.")

    env_down_local(env_dir)
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
    config = load_env_config(env_dir)

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


# ---------------------------------------------------------------------------
# env list / env delete
# ---------------------------------------------------------------------------


@env.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--quiet", "-q", is_flag=True, help="Print environment names only.")
@click.pass_context
@with_command_telemetry("env list", resolver=env_command_capture_config)
def list_envs(ctx: click.Context, as_json: bool, quiet: bool) -> None:
    """List all environments."""
    envs_root = get_environments_dir(ctx=ctx)

    if not envs_root.is_dir():
        click.echo(f"No environments directory found at {envs_root}.")
        return

    entries: list[dict[str, Any]] = []
    for child in sorted(envs_root.iterdir()):
        if not child.is_dir():
            continue
        env_yaml = child / "env.yaml"
        if not env_yaml.is_file():
            continue

        status = detect_env_status(child)
        created = get_env_created_date(child)
        tools: list[str] = []
        if status != "error":
            try:
                config = load_env_config(child)
                tools = list(config.tools)
            except Exception:  # noqa: S110
                pass

        entries.append(
            {
                "name": child.name,
                "status": status,
                "tools": tools,
                "created": created,
                "path": str(child),
            }
        )

    if not entries:
        click.echo("No environments found.")
        emit_cli_event("env_list_completed", {"env_count": 0})
        return

    if quiet:
        for entry in entries:
            click.echo(entry["name"])
        emit_cli_event("env_list_completed", {"env_count": len(entries)})
        return

    if as_json:
        click.echo(json.dumps(entries, indent=2))
        emit_cli_event("env_list_completed", {"env_count": len(entries)})
        return

    # Table output
    name_width = max(*(len(e["name"]) for e in entries), 14)
    status_width = 10
    tools_width = 20
    header = (
        f"{'NAME':<{name_width}}  {'STATUS':<{status_width}}  {'TOOLS':<{tools_width}}  CREATED"
    )
    click.echo(header)
    for entry in entries:
        tools_str = ", ".join(entry["tools"]) if entry["tools"] else ""
        click.echo(
            f"{entry['name']:<{name_width}}  {entry['status']:<{status_width}}  "
            f"{tools_str:<{tools_width}}  {entry['created']}"
        )

    emit_cli_event("env_list_completed", {"env_count": len(entries)})


@env.command()
@click.argument("env_name")
@click.option("--force", "-f", is_flag=True, help="Skip deletion confirmation.")
@click.option("--daytona", is_flag=True, help="Also tear down Daytona sandbox.")
@click.pass_context
@with_command_telemetry("env delete", resolver=env_command_capture_config)
def delete(ctx: click.Context, env_name: str, force: bool, daytona: bool) -> None:
    """Delete an environment and clean up Docker resources."""
    env_dir = resolve_env_dir(env_name, ctx=ctx, require_env_yaml=False)

    # Refuse if path escapes the environments root.
    envs_root = get_environments_dir(ctx=ctx)
    if not env_dir.resolve().is_relative_to(envs_root.resolve()):
        click.echo(
            click.style(
                f"Refusing to delete '{env_name}': resolves outside the environments directory.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    # Refuse if not a SimLab environment.
    if not (env_dir / "env.yaml").is_file():
        click.echo(
            click.style(
                f"'{env_name}' is not a SimLab environment (no env.yaml).",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    # Refuse if any local container is running.
    if has_any_running_containers(env_dir):
        click.echo(
            click.style(
                f"Environment '{env_name}' is currently running.",
                fg="red",
            ),
            err=True,
        )
        click.echo(
            f"Stop it first with 'simlab env down {env_name}'.",
            err=True,
        )
        raise SystemExit(1)

    # Daytona state handling.
    daytona_state_file = env_dir / "daytona-state.json"
    has_daytona_state = daytona_state_file.is_file()
    if has_daytona_state and not daytona:
        click.echo(
            click.style(
                "This environment has Daytona state. "
                "Pass --daytona to also clean up the remote sandbox.",
                fg="yellow",
            ),
        )

    # Confirmation prompt.
    if not force:
        confirmed = click.confirm(
            f"This will remove all data for '{env_name}' (config, tasks, results). Are you sure?",
            default=False,
        )
        if not confirmed:
            click.echo("Aborted.")
            return

    # Tear down Daytona sandbox if requested.
    if daytona and has_daytona_state:
        global_cfg = get_global_config_from_ctx(ctx)
        env_down_daytona(env_dir, daytona_api_key=global_cfg.daytona_api_key)

    # Clean up Docker resources (volumes, networks).
    click.echo("Cleaning up Docker resources...")
    docker_ok = env_purge_docker_local(env_dir)
    if not docker_ok:
        click.echo(
            click.style(
                "Aborting: Docker resources could not be cleaned up. "
                "Resolve the Docker errors above, then retry.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    # Remove the environment directory.
    try:
        shutil.rmtree(env_dir)
    except OSError as exc:
        click.echo(
            click.style(f"Failed to delete environment directory: {exc}", fg="red"),
            err=True,
        )
        raise SystemExit(1) from exc

    click.echo(click.style(f"Environment '{env_name}' deleted.", fg="green"))
    emit_cli_event(
        "env_delete_completed",
        {"docker_purged": True},
    )
