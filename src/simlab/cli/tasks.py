"""CLI commands for browsing and seeding scenario tasks."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource

from simlab.api.client import ScenarioManagerApiError
from simlab.api.client import ScenarioManagerClient
from simlab.api.client import resolve_scenario_manager_api_url
from simlab.api.schemas import ScenarioTask
from simlab.cli.progress import ParallelRolloutProgress
from simlab.cli.progress import StepProgress
from simlab.cli.rollout_summary_card import print_parallel_rollout_summary_card
from simlab.cli.rollout_summary_card import print_single_rollout_summary_card
from simlab.composer.engine import EnvConfig
from simlab.config import get_global_config_from_ctx
from simlab.config import provider_api_key_env_var
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_env_dir
from simlab.env_artifacts import ensure_env_artifacts_current
from simlab.env_artifacts import load_env_config
from simlab.env_registry import build_registry
from simlab.runtime.adapters.harbor import prepare as harbor_prepare
from simlab.runtime.adapters.harbor import workspace as harbor_workspace_runtime
from simlab.runtime.env_lifecycle import ensure_daytona_sandbox_ready
from simlab.runtime.env_lifecycle import ensure_env_started_daytona
from simlab.runtime.env_lifecycle import ensure_env_started_local
from simlab.runtime.env_lifecycle import env_down_daytona
from simlab.runtime.env_lifecycle import env_down_local
from simlab.runtime.env_lifecycle import env_has_local_services
from simlab.runtime.env_lifecycle import is_env_running_local
from simlab.runtime.env_lifecycle import run_env_seed_daytona
from simlab.runtime.env_lifecycle import run_env_seed_local
from simlab.runtime.rollout_runner import ROLLOUT_FORMAT_ATIF
from simlab.runtime.rollout_runner import ROLLOUT_FORMAT_DEFAULT
from simlab.runtime.rollout_runner import SingleRolloutOutcome
from simlab.runtime.rollout_runner import (
    ensure_task_calendar_accounts as _ensure_task_calendar_accounts,
)
from simlab.runtime.rollout_runner import get_env_runtime_helpers
from simlab.runtime.rollout_runner import load_local_task as _load_local_task
from simlab.runtime.rollout_runner import load_profiles as _load_profiles
from simlab.runtime.rollout_runner import load_tasks as _load_tasks
from simlab.runtime.rollout_runner import needed_task_endpoints
from simlab.runtime.rollout_runner import (
    provision_task_calendar_users as _provision_task_calendar_users,
)
from simlab.runtime.rollout_runner import (
    provision_task_group_channels as _provision_task_group_channels,
)
from simlab.runtime.rollout_runner import (
    require_reachable_endpoints as _require_reachable_endpoints,
)
from simlab.runtime.rollout_runner import (
    resolve_agent_runtime_settings as _resolve_shared_agent_runtime_settings,
)
from simlab.runtime.rollout_runner import resolve_endpoints as _resolve_endpoints
from simlab.runtime.rollout_runner import resolve_rollout_format as _resolve_rollout_format
from simlab.runtime.rollout_runner import rewrite_tool_server_urls as _rewrite_tool_server_urls
from simlab.runtime.rollout_runner import run_single_rollout as _run_single_rollout
from simlab.runtime.rollout_runner import seed_task_data as _seed_task_data
from simlab.seeder import get_tool_endpoints
from simlab.telemetry import TelemetryCaptureConfig
from simlab.telemetry import emit_cli_event
from simlab.telemetry import normalize_config_path
from simlab.telemetry import resolve_scenario_manager_capture_config
from simlab.telemetry import with_command_telemetry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tasks_source_label(tasks_dir: str | None) -> str:
    """Return the telemetry-safe source label for task commands."""
    return "local_bundle" if tasks_dir else "scenario_manager_api"


def scenario_manager_task_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for task commands when a Collinear API key is configured."""
    _ = args
    return resolve_scenario_manager_capture_config(
        ctx,
        config_path=normalize_config_path(kwargs.get("config_path")),
    )


def task_bundle_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for local task bundle commands with an API key."""
    _ = args, kwargs
    return resolve_scenario_manager_capture_config(ctx)


def _is_test_task(task: dict[str, Any]) -> bool:
    """Return True when a local task should be treated as a test task."""
    meta = task.get("meta", {})
    category = meta.get("category", task.get("category", ""))
    return str(category or "").strip().lower() == "test"


def _task_has_verifier(task: dict[str, Any]) -> bool:
    """Return True when a task has a runnable verifier configured."""
    return bool(task.get("verifiers") or task.get("evaluators") or task.get("harbor_verifier"))


def _failed_verifier_files(bundle_dir: Path) -> list[Path]:
    """Return failed verifier artifacts from a local task-gen bundle."""
    verifiers_dir = bundle_dir / "verifiers"
    if not verifiers_dir.is_dir():
        return []
    return sorted(verifiers_dir.glob("*.FAILED.py"))


def _warn_about_local_bundle_verifiers(
    bundle_dir: Path,
    tasks_list: list[dict[str, Any]],
) -> None:
    """Warn when a local generated bundle has missing or failed verifiers."""
    failed_files = _failed_verifier_files(bundle_dir)
    missing_count = sum(1 for task in tasks_list if not _task_has_verifier(task))
    if not failed_files and missing_count == 0:
        return

    total = len(tasks_list)
    parts = []
    if missing_count:
        parts.append(f"{missing_count}/{total} task(s) have no verifier wiring")
    if failed_files:
        parts.append(f"{len(failed_files)} failed verifier file(s) are present")
    click.echo(
        click.style(
            "Warning: local task bundle verifier coverage is incomplete: "
            + "; ".join(parts)
            + ". Unverified runs will not be scored.",
            fg="yellow",
        ),
        err=True,
    )


def _warn_if_local_task_has_no_verifier(bundle_dir: Path | None, task: dict[str, Any]) -> None:
    """Warn when the selected local task will run without verifier scoring."""
    if bundle_dir is None or _task_has_verifier(task):
        return
    task_id = str(task.get("meta", {}).get("task_id", "") or "selected task")
    failed_count = len(_failed_verifier_files(bundle_dir))
    suffix = (
        f" The bundle also contains {failed_count} failed verifier file(s)." if failed_count else ""
    )
    click.echo(
        click.style(
            f"Warning: task '{task_id}' has no verifier wiring, so this run will not be scored."
            + suffix,
            fg="yellow",
        ),
        err=True,
    )


def _local_task_rows(
    tasks_list: list[dict[str, Any]], *, include_test: bool
) -> list[dict[str, str]]:
    """Convert local bundle tasks into table rows for CLI display."""
    rows: list[dict[str, str]] = []
    for task in tasks_list:
        if not include_test and _is_test_task(task):
            continue
        meta = task.get("meta", {})
        apps = task.get("apps", [])
        rows.append(
            {
                "task_id": str(meta.get("task_id", "")),
                "name": str(meta.get("display_name", task.get("name", "?")) or "?"),
                "difficulty": str(meta.get("difficulty", task.get("difficulty", "?")) or "?"),
                "apps": ", ".join(str(app) for app in apps if app),
            }
        )
    return rows


def _print_task_table(title: str, rows: list[dict[str, str]]) -> None:
    """Render a fixed-width task table for API or local bundle sources."""
    id_w = 100
    name_w = 50
    difficulty_w = 10
    apps_w = 48
    click.echo(click.style(f"\n  {title}\n", bold=True))
    click.echo(
        f"  {'ID':<{id_w}} {'Name':<{name_w}} {'Difficulty':<{difficulty_w}} {'Apps':<{apps_w}}"
    )
    click.echo(f"  {'─' * id_w} {'─' * name_w} {'─' * difficulty_w} {'─' * apps_w}")
    for row in rows:
        task_id = _truncate_for_table(row["task_id"], id_w)
        name = _truncate_for_table(row["name"], name_w)
        difficulty = _truncate_for_table(row["difficulty"], difficulty_w)
        apps = _truncate_for_table(row["apps"], apps_w)
        click.echo(
            f"  {task_id:<{id_w}} {name:<{name_w}} {difficulty:<{difficulty_w}} {apps:<{apps_w}}"
        )
    click.echo(f"\n  {len(rows)} task(s) total.\n")


def _api_task_to_local(
    api_task: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Convert API ScenarioTaskResponse dict to local task dict + profiles for seed/run."""
    meta = {
        "task_id": api_task.get("task_id", ""),
        "display_name": api_task.get("name", ""),
        "difficulty": api_task.get("difficulty"),
        "category": api_task.get("category", ""),
    }
    tool_servers = api_task.get("tool_servers") or []
    task_dict: dict[str, Any] = {
        "meta": meta,
        "task": api_task.get("description", ""),
        "apps": [ts.get("name", "") for ts in tool_servers if ts.get("name")],
        "tool_servers": list(tool_servers),
        "seed_emails": api_task.get("seed_emails") or [],
        "seed_calendar_events": api_task.get("seed_calendar_events") or [],
        "seed_group_channels": api_task.get("seed_group_channels") or [],
        "npcs": [{"id": p.get("profile_id", "")} for p in (api_task.get("npc_profiles") or [])],
        "verifiers": [
            {"func": "python_module", "module": m} for m in (api_task.get("verifier_modules") or [])
        ],
    }
    npc_profiles = api_task.get("npc_profiles") or []
    profiles = {}
    for p in npc_profiles:
        if not isinstance(p, dict):
            continue
        pid = p.get("profile_id")
        if pid:
            profiles[pid] = p
    return task_dict, profiles


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


def _require_config_with_template(config_path: str | None, config: EnvConfig | None) -> str:
    """Ensure config has a template; return template name. Exits on error."""
    if not config and config_path:
        path = Path(config_path)
        if not path.exists():
            click.echo(click.style(f"Config not found: {path}", fg="red"), err=True)
            raise SystemExit(1)
        config = load_env_config(path.parent)
    if not config:
        click.echo(
            click.style(
                "Provide --env with an environment that has a template (from env init --template <name>).",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)
    template = (config.template or "").strip()
    if not template:
        click.echo(
            click.style(
                "Config has no 'template'. Add template: <name> or recreate with simlab env init --template <name>.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)
    return template


@click.group()
def tasks() -> None:
    """Browse tasks and run agents (uses template from env config)."""


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@tasks.command("list")
@click.option(
    "--env",
    "env_name",
    default=None,
    help="Environment name (list tasks for this env's template).",
)
@click.option("--include-test", is_flag=True, help="Include test-category tasks.")
@click.option(
    "--tasks-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Path to a local task bundle directory from tasks-gen.",
)
@click.pass_context
@with_command_telemetry("tasks list", resolver=scenario_manager_task_capture_config)
def list_tasks(
    ctx: click.Context,
    env_name: str | None,
    include_test: bool,
    tasks_dir: str | None,
) -> None:
    """List all tasks. Require either --env or --tasks-dir."""
    if tasks_dir and env_name:
        click.echo(click.style("Provide --env or --tasks-dir, not both.", fg="red"), err=True)
        raise SystemExit(1)
    if not tasks_dir and not env_name:
        click.echo(
            click.style("Provide --env <name> or --tasks-dir <path>.", fg="red"),
            err=True,
        )
        raise SystemExit(1)

    if tasks_dir:
        bundle_dir = Path(tasks_dir)
        local_tasks = _load_tasks(bundle_dir)
        visible_tasks = [task for task in local_tasks if include_test or not _is_test_task(task)]
        _warn_about_local_bundle_verifiers(bundle_dir, visible_tasks)
        rows = _local_task_rows(local_tasks, include_test=include_test)
        _print_task_table(f"Tasks (local bundle: {bundle_dir})", rows)
        emit_cli_event(
            "tasks_list_completed",
            {
                "task_source": tasks_source_label(tasks_dir),
                "task_count": len(rows),
                "include_test": include_test,
            },
        )
        return

    if env_name is None:  # guaranteed by early exit above
        raise click.ClickException("--env is required")
    env_dir = resolve_env_dir(env_name, ctx=ctx)
    config_path = str(env_dir / "env.yaml")
    global_cfg = get_global_config_from_ctx(ctx)
    config = load_env_config(env_dir)
    template = _require_config_with_template(config_path, config)
    base_url = resolve_scenario_manager_api_url(
        config_path=Path(config_path),
        config=config,
        base_url=resolve_scenario_manager_api_url(config=global_cfg),
    )
    api_key = resolve_collinear_api_key(config=global_cfg)
    sm_client = ScenarioManagerClient(base_url=base_url, api_key=api_key)
    try:
        backend_id = sm_client.resolve_template_to_backend_id(template)
        data_resp = sm_client.list_scenario_tasks(
            backend_id,
            include_hidden=True,
            include_test=include_test,
        )
    except ScenarioManagerApiError as e:
        click.echo(click.style(str(e), fg="red"), err=True)
        raise SystemExit(1) from e
    rows = [
        {
            "task_id": task.task_id,
            "name": task.name or "?",
            "difficulty": task.difficulty or "?",
            "apps": ", ".join(ts.name for ts in task.tool_servers if ts.name),
        }
        for task in data_resp.tasks
    ]
    _print_task_table(f"Tasks ({template} / slug: {backend_id})", rows)
    emit_cli_event(
        "tasks_list_completed",
        {
            "task_source": tasks_source_label(tasks_dir),
            "task_count": len(rows),
            "include_test": include_test,
        },
    )


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def _print_task_info(task: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> None:
    """Print task details (shared for local and API task dicts)."""
    meta = task.get("meta", {})
    click.echo(click.style(f"\n  {meta.get('display_name', task.get('name', '?'))}", bold=True))
    click.echo(f"  ID:         {meta.get('task_id', task.get('task_id', '?'))}")
    click.echo(f"  Difficulty: {meta.get('difficulty', task.get('difficulty', '?'))}")
    click.echo(f"  Category:   {meta.get('category', task.get('category', '?'))}")
    click.echo(f"  Apps:       {', '.join(task.get('apps', []))}")

    desc = task.get("task", task.get("description", ""))
    click.echo(click.style("\n  Description:", bold=True))
    for line in (desc or "").split("\n"):
        click.echo(f"    {line}")

    servers = task.get("tool_servers", [])
    if servers:
        click.echo(click.style("\n  Tool servers:", bold=True))
        for s in servers:
            name = s.get("name", "?")
            url = s.get("tool_server_url", "")
            click.echo(f"    {name:<25} {url}")

    npcs = task.get("npcs", [])
    if npcs:
        click.echo(click.style("\n  NPCs:", bold=True))
        for npc in npcs:
            npc_id = npc.get("id", "?") if isinstance(npc, dict) else "?"
            profile = profiles.get(npc_id, {})
            email = profile.get("email", "")
            name_str = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
            label = f"{name_str} <{email}>" if name_str and email else npc_id
            click.echo(f"    {npc_id:<25} {label}")

    emails = task.get("seed_emails", [])
    cal_events = task.get("seed_calendar_events", [])
    if emails or cal_events:
        click.echo(click.style("\n  Seed data:", bold=True))
        if emails:
            click.echo(f"    Emails:           {len(emails)}")
        if cal_events:
            click.echo(f"    Calendar events:  {len(cal_events)}")

    verifiers = task.get("verifiers", [])
    if verifiers:
        click.echo(click.style("\n  Verifiers:", bold=True))
        for v in verifiers:
            mod = v.get("module", v.get("name", "?")) if isinstance(v, dict) else "?"
            click.echo(f"    {mod}")
    verifier_modules = task.get("verifier_modules", [])
    if verifier_modules and not verifiers:
        click.echo(click.style("\n  Verifiers:", bold=True))
        for m in verifier_modules:
            click.echo(f"    {m}")

    click.echo()


def _match_task(tasks_list: list[ScenarioTask], task_id: str) -> ScenarioTask | None:
    """Match by exact, suffix, or contains (supports short IDs)."""
    api_task = next((t for t in tasks_list if t.task_id.strip() == task_id), None)
    if api_task:
        return api_task
    return next(
        (t for t in tasks_list if t.task_id.endswith(task_id) or task_id in t.task_id),
        None,
    )


def _resolve_api_task_by_id(
    sm_client: ScenarioManagerClient,
    backend_id: str,
    task_id: str,
    include_test: bool,
) -> ScenarioTask | None:
    data_resp = sm_client.list_scenario_tasks(
        backend_id,
        include_hidden=True,
        include_test=include_test,
    )
    api_task = _match_task(data_resp.tasks, task_id)
    if api_task is not None or include_test:
        return api_task

    # Explicit IDs should resolve even when default CLI listings hide test tasks.
    data_resp_with_test = sm_client.list_scenario_tasks(
        backend_id,
        include_hidden=True,
        include_test=True,
    )
    return _match_task(data_resp_with_test.tasks, task_id)


def _resolve_agent_runtime_settings(
    global_cfg: Any,
    model: str | None,
    provider: str | None,
    api_key: str | None,
    base_url: str | None,
) -> tuple[str, str, str | None, str | None]:
    return _resolve_shared_agent_runtime_settings(
        global_cfg=global_cfg,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )


def run_preflight_checks(
    *,
    agent_import_path: str | None,
    api_key: str | None,
    model: str,
    provider: str,
    env_dir: Path,
    daytona: bool,
    daytona_api_key: str | None,
    skip_env_setup: bool,
) -> None:
    """Validate common-mistake inputs before any heavy work (env startup, seed, etc.)."""
    if not daytona and not skip_env_setup and env_has_local_services(env_dir):
        docker_path = shutil.which("docker")
        if docker_path is None:
            click.echo(
                click.style("Error: Docker is not installed.", fg="red"),
                err=True,
            )
            click.echo(
                click.style(
                    "  Install Docker Desktop from https://docs.docker.com/get-docker/",
                    dim=True,
                ),
                err=True,
            )
            raise SystemExit(1)
        try:
            subprocess.run(
                [docker_path, "info"],
                capture_output=True,
                timeout=5,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            click.echo(
                click.style("Error: Docker is not running.", fg="red"),
                err=True,
            )
            click.echo(
                click.style("  Start Docker Desktop, then retry.", dim=True),
                err=True,
            )
            raise SystemExit(1)

    if daytona and not daytona_api_key:
        click.echo(
            click.style("Error: Daytona API key required when using --daytona.", fg="red"),
            err=True,
        )
        click.echo(
            click.style("  Set DAYTONA_API_KEY or SIMLAB_DAYTONA_API_KEY.", dim=True),
            err=True,
        )
        raise SystemExit(1)

    if not agent_import_path and not api_key:
        click.echo(
            click.style("Error: Agent API key required.", fg="red"),
            err=True,
        )
        hint = "  Run: simlab tasks run --agent-api-key <key>\n  Or set SIMLAB_AGENT_API_KEY."
        provider_env = provider_api_key_env_var(provider)
        if provider_env:
            display = {"openai": "OpenAI", "together_ai": "Together AI"}.get(
                provider, provider.title()
            )
            hint += f"\n  For {display} models, {provider_env} also works."
        click.echo(click.style(hint, dim=True), err=True)
        raise SystemExit(1)

    if agent_import_path:
        try:
            from simlab.agents.loader import load_agent_class

            load_agent_class(agent_import_path)
        except Exception as exc:
            click.echo(
                click.style(
                    f"Error: Cannot load agent from '{agent_import_path}'.",
                    fg="red",
                ),
                err=True,
            )
            click.echo(click.style(f"  {exc}", dim=True), err=True)
            click.echo(
                click.style(
                    "  Check the module path and ensure dependencies are installed.",
                    dim=True,
                ),
                err=True,
            )
            raise SystemExit(1) from exc

    if not agent_import_path and model:
        stripped = model.strip()
        is_placeholder = stripped in {"<model>", "model", "<model_name>"}
        has_whitespace = " " in stripped or "	" in stripped
        if is_placeholder or has_whitespace:
            click.echo(
                click.style(
                    f"Error: '{model}' doesn't look like a valid model name.",
                    fg="red",
                ),
                err=True,
            )
            click.echo(
                click.style("  Example: --agent-model gpt-4o", dim=True),
                err=True,
            )
            raise SystemExit(1)


@tasks.command()
@click.option(
    "--env",
    "env_name",
    default=None,
    help="Environment name (task template from this env).",
)
@click.option("--task", "task_id", required=True, help="Task ID to show info for.")
@click.option("--include-test", is_flag=True, help="Include test-category tasks.")
@click.option(
    "--tasks-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Path to a local task bundle directory from tasks-gen.",
)
@click.pass_context
@with_command_telemetry("tasks info", resolver=scenario_manager_task_capture_config)
def info(
    ctx: click.Context,
    env_name: str | None,
    task_id: str,
    include_test: bool,
    tasks_dir: str | None,
) -> None:
    """Show detailed information about a task. Require either --env or --tasks-dir."""
    if tasks_dir and env_name:
        click.echo(click.style("Provide --env or --tasks-dir, not both.", fg="red"), err=True)
        raise SystemExit(1)
    if not tasks_dir and not env_name:
        click.echo(
            click.style("Provide --env <name> or --tasks-dir <path>.", fg="red"),
            err=True,
        )
        raise SystemExit(1)

    if tasks_dir:
        bundle_dir = Path(tasks_dir)
        task_dict, profiles, _ = _load_local_task(bundle_dir, task_id)
        _print_task_info(task_dict, profiles)
        emit_cli_event(
            "tasks_info_completed",
            {
                "task_source": tasks_source_label(tasks_dir),
                "include_test": include_test,
            },
        )
        return

    if env_name is None:  # guaranteed by early exit above
        raise click.ClickException("--env is required")
    env_dir = resolve_env_dir(env_name, ctx=ctx)
    config_path = str(env_dir / "env.yaml")
    global_cfg = get_global_config_from_ctx(ctx)
    config = load_env_config(env_dir)
    template = _require_config_with_template(config_path, config)
    base_url = resolve_scenario_manager_api_url(
        config_path=Path(config_path),
        config=config,
        base_url=resolve_scenario_manager_api_url(config=global_cfg),
    )
    api_key = resolve_collinear_api_key(config=global_cfg)
    sm_client = ScenarioManagerClient(base_url=base_url, api_key=api_key)
    try:
        backend_id = sm_client.resolve_template_to_backend_id(template)
        api_task = _resolve_api_task_by_id(sm_client, backend_id, task_id, include_test)
    except ScenarioManagerApiError as e:
        click.echo(click.style(str(e), fg="red"), err=True)
        raise SystemExit(1) from e
    if api_task is None:
        click.echo(click.style(f"Task not found: {task_id}", fg="red"), err=True)
        raise SystemExit(1)
    task_dict, profiles = _api_task_to_local(api_task.model_dump())
    # Normalize API-provided tool_server_url values for local CLI display.
    endpoints = get_tool_endpoints(config, config_path=Path(config_path))
    rewritten_for_display = _rewrite_tool_server_urls(task_dict, endpoints)
    _print_task_info(rewritten_for_display, profiles)
    emit_cli_event(
        "tasks_info_completed",
        {
            "task_source": tasks_source_label(tasks_dir),
            "include_test": include_test,
        },
    )


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


@tasks.command()
@click.option(
    "--env",
    "env_name",
    required=True,
    help="Environment name (for config and endpoints).",
)
@click.option("--task", "task_id", required=True, help="Task ID to seed data for.")
@click.option("--include-test", is_flag=True, help="Include test-category tasks.")
@click.option(
    "--tasks-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Path to a local task bundle (default: fetch task from API using env template).",
)
@click.option(
    "--daytona", is_flag=True, help="Seed into a Daytona sandbox instead of local Docker."
)
@click.pass_context
@with_command_telemetry("tasks seed", resolver=scenario_manager_task_capture_config)
def seed(
    ctx: click.Context,
    env_name: str,
    task_id: str,
    include_test: bool,
    tasks_dir: str | None,
    daytona: bool,
) -> None:
    """Inject task-specific seed data (emails, calendar events). Requires --env."""
    global_cfg = get_global_config_from_ctx(ctx)
    env_dir = resolve_env_dir(env_name, ctx=ctx)
    ensure_env_artifacts_current(env_dir, action_label="tasks seed")
    config_path = str(env_dir / "env.yaml")
    config_file = Path(config_path)
    config = load_env_config(env_dir)
    if tasks_dir:
        bundle_dir = Path(tasks_dir)
        task_dict, profiles, _ = _load_local_task(bundle_dir, task_id)
    else:
        template = _require_config_with_template(config_path, config)
        base_url = resolve_scenario_manager_api_url(
            config_path=config_file,
            config=config,
            base_url=resolve_scenario_manager_api_url(config=global_cfg),
        )
        api_key = resolve_collinear_api_key(config=global_cfg)
        sm_client = ScenarioManagerClient(base_url=base_url, api_key=api_key)
        try:
            backend_id = sm_client.resolve_template_to_backend_id(template)
            api_task = _resolve_api_task_by_id(sm_client, backend_id, task_id, include_test)
        except ScenarioManagerApiError as e:
            click.echo(click.style(str(e), fg="red"), err=True)
            raise SystemExit(1) from e
        if api_task is None:
            click.echo(click.style(f"Task not found: {task_id}", fg="red"), err=True)
            raise SystemExit(1)
        task_dict, profiles = _api_task_to_local(api_task.model_dump())
    emails = task_dict.get("seed_emails", [])
    cal_events = task_dict.get("seed_calendar_events", [])
    group_channels = task_dict.get("seed_group_channels", [])
    if not emails and not cal_events and not group_channels:
        click.echo("No seed data for this task.")
        emit_cli_event(
            "tasks_seed_completed",
            {
                "task_source": "local_bundle" if tasks_dir else "scenario_manager_api",
                "mode": "daytona" if daytona else "local",
                "seed_email_count": 0,
                "seed_calendar_event_count": 0,
                "ok_count": 0,
                "fail_count": 0,
            },
        )
        return

    meta = task_dict.get("meta", {})
    click.echo(
        click.style(
            f"\nSeeding task: {meta.get('display_name', meta.get('task_id', '?'))}\n", bold=True
        )
    )

    endpoints, using_daytona = _resolve_endpoints(
        config_path=config_path,
        config=config,
        daytona_requested=daytona,
        daytona_api_key=global_cfg.daytona_api_key,
    )
    needed_seed_endpoints = needed_task_endpoints(
        task_dict,
        endpoints,
        include_tool_servers=False,
    )
    if needed_seed_endpoints:
        _require_reachable_endpoints(
            endpoints=needed_seed_endpoints,
            action="task seeding",
            using_daytona=using_daytona,
            config_path=config_path,
            wait=using_daytona,
        )
    if using_daytona:
        click.echo(click.style("Using Daytona tool endpoints.", fg="cyan"))
    _provision_task_group_channels(
        task_dict,
        profiles,
        config,
        config_path,
        using_daytona=using_daytona,
        daytona_api_key=global_cfg.daytona_api_key,
    )
    _provision_task_calendar_users(
        task_dict,
        config,
        config_path,
        using_daytona=using_daytona,
        daytona_api_key=global_cfg.daytona_api_key,
    )
    _ensure_task_calendar_accounts(task_dict, profiles, endpoints, config)
    ok, fail = _seed_task_data(task_dict, profiles, endpoints)

    click.echo()
    if fail == 0:
        click.echo(click.style(f"  Done — {ok} item(s) seeded successfully.", fg="green"))
    else:
        click.echo(click.style(f"  Done — {ok} succeeded, {fail} failed.", fg="yellow"))
    click.echo()
    emit_cli_event(
        "tasks_seed_completed",
        {
            "task_source": "local_bundle" if tasks_dir else "scenario_manager_api",
            "mode": "daytona" if using_daytona else "local",
            "seed_email_count": len(emails),
            "seed_calendar_event_count": len(cal_events),
            "ok_count": ok,
            "fail_count": fail,
        },
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@tasks.command()
@click.option(
    "--env",
    "env_name",
    default=None,
    help="Environment name (for config and endpoints).",
)
@click.option("--task", "task_id", default=None, help="Task ID to run.")
@click.option("--include-test", is_flag=True, help="Include test-category tasks.")
@click.option(
    "--tasks-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Path to a local task bundle (default: fetch task from API using env template).",
)
@click.option(
    "--harbor",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Path to a single Harbor task directory to run through a generated temporary env.",
)
@click.option(
    "--tasks-rollout-format",
    default=None,
    type=click.Choice((ROLLOUT_FORMAT_DEFAULT, ROLLOUT_FORMAT_ATIF), case_sensitive=False),
    help="Rollout artifact format to write (defaults to Harbor=atif, otherwise default).",
)
@click.option(
    "--agent-model",
    "model",
    default=None,
    help="Model name for the reference agent (defaults to SIMLAB_AGENT_MODEL / global config). Only used when not using --agent-import-path.",
)
@click.option(
    "--agent-provider",
    "provider",
    default=None,
    help="Provider for the reference agent (defaults to SIMLAB_AGENT_PROVIDER / global config, else openai). Only used when not using --agent-import-path.",
)
@click.option(
    "--agent-api-key",
    "api_key",
    default=None,
    help="API key for the reference agent (defaults to SIMLAB_AGENT_API_KEY or OPENAI_API_KEY). Only used when not using --agent-import-path. See LiteLLM docs for provider-specific keys.",
)
@click.option(
    "--agent-base-url",
    "base_url",
    default=None,
    help="Base URL for the reference agent LLM API (defaults to SIMLAB_AGENT_BASE_URL / global config). Only used when not using --agent-import-path.",
)
@click.option("--max-steps", type=int, default=30, help="Maximum agent steps (default: 30).")
@click.option(
    "--agent-import-path",
    default=None,
    help=(
        "Custom external agent import path in module:Class format. "
        "If omitted, uses the baked-in reference agent."
    ),
)
@click.option(
    "--agent-timeout-seconds",
    type=float,
    default=600.0,
    show_default=True,
    help="Hard timeout for agent setup+run lifecycle.",
)
@click.option(
    "--no-seed",
    is_flag=True,
    help="Skip all seeding and provisioning (env seed, task data, channels, calendar).",
)
@click.option(
    "--keep-alive", is_flag=True, help="Do not tear down the environment after the run completes."
)
@click.option(
    "--skip-env-setup",
    is_flag=True,
    help="Skip automatic environment startup (assume env is already running).",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed agent logs.")
@click.option("--quiet", is_flag=True, help="Suppress the post-run summary card.")
@click.option(
    "--daytona", is_flag=True, help="Run against a Daytona sandbox instead of local Docker."
)
@click.option(
    "--rollout-count",
    type=int,
    default=1,
    show_default=True,
    help="Number of parallel rollouts to execute (requires --daytona).",
)
@click.option(
    "--max-parallel",
    type=int,
    default=3,
    show_default=True,
    help="Maximum concurrent Daytona sandboxes for parallel rollouts.",
)
@click.pass_context
@with_command_telemetry("tasks run", resolver=scenario_manager_task_capture_config)
def run(
    ctx: click.Context,
    env_name: str | None,
    task_id: str | None,
    include_test: bool,
    tasks_dir: str | None,
    harbor: str | None,
    tasks_rollout_format: str | None,
    model: str,
    provider: str,
    api_key: str | None,
    base_url: str | None,
    max_steps: int,
    agent_import_path: str | None,
    agent_timeout_seconds: float,
    no_seed: bool,
    keep_alive: bool,
    skip_env_setup: bool,
    verbose: bool,
    quiet: bool,
    daytona: bool,
    rollout_count: int,
    max_parallel: int,
) -> None:
    """Seed task data and run external agent contract.

    Automatically starts the environment if it is not already running,
    and tears it down after the run completes (unless --keep-alive).
    Use --skip-env-setup to assume the env is already running.
    Requires --env. Optionally use --tasks-dir for a local task bundle.
    """
    global_cfg = get_global_config_from_ctx(ctx)
    harbor_workspace: tempfile.TemporaryDirectory[str] | None = None
    harbor_workspace_root: Path | None = None
    harbor_task_id_for_debug = task_id or "harbor-task"
    run_succeeded = False
    if harbor:
        if env_name:
            click.echo(click.style("Provide --env or --harbor, not both.", fg="red"), err=True)
            raise SystemExit(1)
        if tasks_dir:
            click.echo(click.style("Do not combine --tasks-dir with --harbor.", fg="red"), err=True)
            raise SystemExit(1)
        if skip_env_setup:
            click.echo(
                click.style("--skip-env-setup is not supported with --harbor.", fg="red"),
                err=True,
            )
            raise SystemExit(1)
        if rollout_count > 1:
            click.echo(
                click.style("--rollout-count > 1 is not supported with --harbor yet.", fg="red"),
                err=True,
            )
            raise SystemExit(1)
        try:
            if keep_alive:
                harbor_workspace_root = harbor_workspace_runtime.create_harbor_workspace_root(
                    task_label=task_id or Path(harbor).name
                )
            else:
                harbor_workspace = tempfile.TemporaryDirectory(prefix="simlab-harbor-")
                harbor_workspace_root = Path(harbor_workspace.name)
            prepared = harbor_prepare.prepare_harbor_run(
                Path(harbor),
                workspace_root=harbor_workspace_root,
                task_id=task_id,
            )
        except ValueError as e:
            click.echo(click.style(f"Harbor task error: {e}", fg="red"), err=True)
            raise SystemExit(1) from e
        env_dir = prepared.env_dir
        ensure_env_artifacts_current(env_dir, action_label="tasks run")
        config_path = str(env_dir / "env.yaml")
        config_file = Path(config_path)
        config = load_env_config(env_dir)
        bundle_dir: Path | None = prepared.bundle_dir
        tasks_dir = str(bundle_dir)
        task_id = prepared.task_id
        harbor_task_id_for_debug = prepared.task_id
        if (
            prepared.agent_timeout_seconds is not None
            and ctx.get_parameter_source("agent_timeout_seconds") is ParameterSource.DEFAULT
        ):
            agent_timeout_seconds = prepared.agent_timeout_seconds
        base_url_api = resolve_scenario_manager_api_url(
            config_path=config_file,
            config=config,
            base_url=resolve_scenario_manager_api_url(config=global_cfg),
        )
        scenario_manager_api_key = resolve_collinear_api_key(config=global_cfg)
    else:
        if not env_name:
            click.echo(
                click.style("Provide --env <name> or --harbor <task-dir>.", fg="red"), err=True
            )
            raise SystemExit(1)
        if not task_id:
            click.echo(click.style("Provide --task <id> when using --env.", fg="red"), err=True)
            raise SystemExit(1)
        env_dir = resolve_env_dir(env_name, ctx=ctx)
        ensure_env_artifacts_current(env_dir, action_label="tasks run")
        config_path = str(env_dir / "env.yaml")
        config_file = Path(config_path)
        config = load_env_config(env_dir)
        bundle_dir = Path(tasks_dir) if tasks_dir else None
        base_url_api = resolve_scenario_manager_api_url(
            config_path=config_file,
            config=config,
            base_url=resolve_scenario_manager_api_url(config=global_cfg),
        )
        scenario_manager_api_key = resolve_collinear_api_key(config=global_cfg)

    rollout_format = _resolve_rollout_format(
        requested=tasks_rollout_format,
        config=config,
        global_cfg=global_cfg,
        harbor=bool(harbor),
    )
    backend_id: str | None = None
    if bundle_dir is not None:
        task_data, profiles, _local_task_file = _load_local_task(bundle_dir, task_id)
        _warn_if_local_task_has_no_verifier(bundle_dir, task_data)
    else:
        template = _require_config_with_template(config_path, config)
        sm_client = ScenarioManagerClient(base_url=base_url_api, api_key=scenario_manager_api_key)
        try:
            backend_id = sm_client.resolve_template_to_backend_id(template)
            api_task = _resolve_api_task_by_id(sm_client, backend_id, task_id, include_test)
        except ScenarioManagerApiError as e:
            click.echo(click.style(str(e), fg="red"), err=True)
            raise SystemExit(1) from e
        if api_task is None:
            click.echo(click.style(f"Task not found: {task_id}", fg="red"), err=True)
            raise SystemExit(1)
        task_data, profiles = _api_task_to_local(api_task.model_dump())
    requested_model = model
    requested_provider = provider
    model, provider, api_key, base_url = _resolve_agent_runtime_settings(
        global_cfg,
        model,
        provider,
        api_key,
        base_url,
    )
    if not agent_import_path and not model:
        click.echo(
            click.style(
                "Agent model required via --agent-model or SIMLAB_AGENT_MODEL / global config.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)
    if agent_import_path:
        model = requested_model or "custom-agent"
        provider = requested_provider or "custom-agent"
    run_preflight_checks(
        agent_import_path=agent_import_path,
        api_key=api_key,
        model=model,
        provider=provider,
        env_dir=env_dir,
        daytona=daytona,
        daytona_api_key=global_cfg.daytona_api_key,
        skip_env_setup=skip_env_setup,
    )

    # --- Parallel rollouts branch ---
    if rollout_count < 1:
        click.echo(
            click.style("--rollout-count must be at least 1.", fg="red"),
            err=True,
        )
        raise SystemExit(1)
    if rollout_count > 1:
        if not daytona:
            click.echo(
                click.style(
                    "--rollout-count > 1 requires --daytona for isolated sandboxes.",
                    fg="red",
                ),
                err=True,
            )
            raise SystemExit(1)
        if max_parallel < 1:
            click.echo(
                click.style(
                    "--max-parallel must be at least 1.",
                    fg="red",
                ),
                err=True,
            )
            raise SystemExit(1)
        if not agent_import_path and not api_key:
            click.echo(
                click.style(
                    "Reference agent requires --agent-api-key or SIMLAB_AGENT_API_KEY "
                    "(or OPENAI_API_KEY for OpenAI). "
                    "Custom agents (--agent-import-path) use their own credentials.",
                    fg="red",
                ),
                err=True,
            )
            raise SystemExit(1)

        from simlab.runtime.env_lifecycle import validate_daytona_coding_assets
        from simlab.runtime.parallel_daytona import ParallelDaytonaOrchestrator

        validate_daytona_coding_assets(config, env_dir)

        # Clean up any sandboxes leaked by a previous crashed parallel run.
        parallel_state = env_dir / "parallel-sandboxes.json"
        if parallel_state.exists():
            cleaned = ParallelDaytonaOrchestrator.cleanup_orphaned_sandboxes(
                parallel_state, daytona_api_key=global_cfg.daytona_api_key
            )
            if cleaned:
                click.echo(
                    click.style(
                        f"Cleaned up {cleaned} orphaned sandbox(es) from a previous run.",
                        fg="yellow",
                    ),
                    err=True,
                )

        registry = build_registry(env_dir=env_dir)
        tool_ports: dict[str, int] = {}
        extra_tool_urls: dict[str, str] = {}
        for tool_name in config.tools:
            tool = registry.get_tool(tool_name)
            if tool is None:
                continue
            if tool.tool_server_port is not None:
                tool_ports[tool_name] = tool.tool_server_port
            elif tool.tool_server_url:
                extra_tool_urls[tool_name] = tool.tool_server_url

        get_profiled_service_names, _ = get_env_runtime_helpers()
        preseed_svc_names = get_profiled_service_names(
            config, profile="preseed", config_path=config_file
        )
        seed_svc_names = get_profiled_service_names(config, profile="seed", config_path=config_file)

        orchestrator = ParallelDaytonaOrchestrator(
            rollout_count=rollout_count,
            max_parallel=max_parallel,
            daytona_api_key=global_cfg.daytona_api_key,
        )
        parallel_task_id = task_data.get("meta", {}).get("task_id", task_id)
        parallel_progress = (
            ParallelRolloutProgress(
                rollout_count=rollout_count,
                task_name=parallel_task_id,
                max_steps=max_steps,
            )
            if not verbose and sys.stdout.isatty()
            else None
        )
        execute_kwargs = {
            "task_id": parallel_task_id,
            "task_data": task_data,
            "profiles": profiles,
            "compose_dir": env_dir,
            "tool_ports": tool_ports,
            "extra_tool_urls": extra_tool_urls,
            "preseed_svc_names": preseed_svc_names,
            "seed_svc_names": seed_svc_names,
            "config": config,
            "config_path": config_path,
            "model": model,
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "max_steps": max_steps,
            "agent_import_path": agent_import_path,
            "agent_timeout_seconds": agent_timeout_seconds,
            "no_seed": no_seed,
            "bundle_dir": bundle_dir,
            "global_cfg": global_cfg,
            "backend_id": backend_id,
            "base_url_api": base_url_api,
            "scenario_manager_api_key": scenario_manager_api_key,
            "rollout_format": rollout_format,
            "progress": parallel_progress,
        }
        if parallel_progress is None:
            summary = orchestrator.execute(**execute_kwargs)
        else:
            with parallel_progress:
                summary = orchestrator.execute(**execute_kwargs)
        print_parallel_rollout_summary_card(
            task_id=summary.task_id,
            rollout_count=summary.rollout_count,
            results=summary.results,
            total_duration_seconds=summary.total_duration_seconds,
            output_dir=summary.output_dir,
            quiet=quiet,
        )
        has_failures = any(
            r.error is not None or r.verification_passed is False for r in summary.results
        )
        if has_failures:
            raise SystemExit(1)
        return

    meta = task_data.get("meta", {})
    display = meta.get("display_name", meta.get("task_id", task_id))
    progress = StepProgress(verbose=False) if not verbose else None

    # --- Phase 0: Auto-start environment if needed ---
    managed_env = False  # True only when *this* run started the env
    has_local_services = env_has_local_services(env_dir)

    run_started = time.monotonic()
    outcome: SingleRolloutOutcome | None = None

    try:
        # --- Phase 0: Auto-start environment if needed ---
        if not skip_env_setup and has_local_services:
            if daytona:
                env_was_running = ensure_daytona_sandbox_ready(
                    env_dir, daytona_api_key=global_cfg.daytona_api_key
                )
            else:
                env_was_running = is_env_running_local(env_dir)

            if not env_was_running:
                if daytona and (env_dir / "daytona-state.json").exists():
                    click.echo(
                        click.style(
                            "An existing daytona-state.json could not be safely "
                            "reconciled with Daytona.\n"
                            "Verify/clean up the referenced sandbox, then retry:\n"
                            f"  rm {env_dir / 'daytona-state.json'}",
                            fg="red",
                        ),
                        err=True,
                    )
                    raise SystemExit(1)

                managed_env = True
                if progress is None:
                    click.echo(click.style("\nStarting services...", bold=True))
                if daytona:
                    ensure_env_started_daytona(
                        env_dir,
                        config,
                        env_dir / "env.yaml",
                        daytona_api_key=global_cfg.daytona_api_key,
                        verbose=verbose,
                        progress=progress,
                    )
                    if not no_seed:
                        run_env_seed_daytona(
                            env_dir,
                            config,
                            env_dir / "env.yaml",
                            daytona_api_key=global_cfg.daytona_api_key,
                        )
                else:
                    env_step = (
                        progress.step("Services started") if progress else contextlib.nullcontext()
                    )
                    with env_step:
                        ensure_env_started_local(
                            env_dir,
                            config,
                            env_dir / "env.yaml",
                            quiet=progress is not None,
                        )
                        if not no_seed:
                            run_env_seed_local(
                                env_dir,
                                config,
                                env_dir / "env.yaml",
                                quiet=progress is not None,
                            )
                if progress is None:
                    click.echo(click.style("Services started.", fg="green"))

        # --- Phases 1-3: Seed, run agent, verify ---
        outcome = _run_single_rollout(
            env_dir=env_dir,
            config=config,
            config_path=config_path,
            global_cfg=global_cfg,
            task_data=task_data,
            task_id=task_id,
            profiles=profiles,
            meta=meta,
            display=display,
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            max_steps=max_steps,
            agent_import_path=agent_import_path,
            agent_timeout_seconds=agent_timeout_seconds,
            no_seed=no_seed,
            verbose=verbose,
            daytona=daytona,
            tasks_dir=tasks_dir,
            bundle_dir=bundle_dir,
            backend_id=backend_id,
            base_url_api=base_url_api,
            scenario_manager_api_key=scenario_manager_api_key,
            managed_env=managed_env,
            keep_alive=keep_alive,
            skip_env_setup=skip_env_setup,
            rollout_format=rollout_format,
            progress=progress,
        )
        run_succeeded = outcome.exit_code == 0
    finally:
        # --- Phase 4: Teardown ---
        if harbor_workspace is not None and not run_succeeded:
            try:
                preserved = harbor_workspace_runtime.preserve_harbor_workspace(
                    Path(harbor_workspace.name),
                    task_id=harbor_task_id_for_debug,
                )
            except Exception as e:
                click.echo(
                    click.style(
                        f"Warning: failed to preserve Harbor workspace: {e}",
                        fg="yellow",
                    ),
                    err=True,
                )
            else:
                click.echo(
                    click.style(f"Preserved Harbor workspace: {preserved}", fg="yellow"),
                    err=True,
                )
        elif harbor_workspace_root is not None and keep_alive:
            click.echo(
                click.style(f"Harbor workspace retained: {harbor_workspace_root}", fg="yellow"),
                err=True,
            )
        if managed_env and not keep_alive:
            if progress is None:
                click.echo(click.style("\nTearing down services...", bold=True))
            try:
                if daytona:
                    env_down_daytona(
                        env_dir,
                        daytona_api_key=global_cfg.daytona_api_key,
                        progress=progress,
                    )
                else:
                    teardown_step = (
                        progress.step("Services stopped") if progress else contextlib.nullcontext()
                    )
                    with teardown_step:
                        env_down_local(env_dir)
            except (Exception, SystemExit):
                click.echo(
                    click.style("Warning: services teardown failed.", fg="yellow"),
                    err=True,
                )
        if harbor_workspace is not None:
            harbor_workspace.cleanup()

    duration_seconds = time.monotonic() - run_started
    if outcome is not None:
        if progress is not None or quiet:
            if outcome.verification_passed is True:
                click.echo(click.style("  ✓ PASS", fg="green"))
            elif outcome.verification_passed is False:
                click.echo(click.style("  ✗ FAIL", fg="red"))
            elif outcome.run_error:
                click.echo(click.style("  ✗ ERROR", fg="red"))
            else:
                click.echo(click.style("  ✓ DONE", fg="yellow"))
        print_single_rollout_summary_card(
            task_id=outcome.task_id,
            model=outcome.model,
            provider=outcome.provider,
            steps_taken=outcome.steps_taken,
            max_steps=outcome.max_steps,
            duration_seconds=duration_seconds,
            reward=outcome.reward,
            verification_passed=outcome.verification_passed,
            run_error=outcome.run_error,
            verifier_results=outcome.verifier_results,
            output_dir=outcome.output_dir,
            quiet=quiet,
        )
        if outcome.exit_code:
            raise SystemExit(outcome.exit_code)


def _print_run_summary(
    artifacts: Any,
    npc_session: Any | None,
    run_dir: Path,
) -> None:
    """Print a concise post-run summary to the terminal."""
    from simlab.run_summary import extract_run_summary

    summary = extract_run_summary(artifacts)

    if summary.error:
        status = click.style(f"failed: {summary.error}", fg="red")
    else:
        status = click.style("completed", fg="green")

    click.echo(click.style("Run summary", bold=True))
    click.echo(f"  Status:       {status}")
    click.echo(f"  Steps:        {summary.steps}")
    if summary.tool_counts:
        parts = [f"{srv} ({cnt})" for srv, cnt in sorted(summary.tool_counts.items())]
        click.echo(f"  Tool calls:   {', '.join(parts)}")
    if summary.npc_msg_count:
        click.echo(f"  NPC msgs:     {summary.npc_msg_count}")
    elif npc_session is not None:
        click.echo(
            click.style(
                "  NPC msgs:     0 (NPC chat available but agent did not use it)",
                fg="yellow",
            )
        )

    if summary.final_observation:
        preview = summary.final_observation[:200].replace("\n", " ")
        if len(summary.final_observation) > 200:
            preview += "..."
        click.echo(f"  Final answer: {preview}")

    click.echo(f"  Output:       {run_dir}")
    click.echo()


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

_REQUIRED_META_FIELDS = {"task_id", "display_name", "difficulty"}


def _truncate_for_table(value: Any, max_width: int) -> str:
    """Return single-line text trimmed for fixed-width CLI tables."""
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_width:
        return text
    if max_width < 2:
        return text[:max_width]
    return text[: max_width - 1] + "…"


@tasks.command()
@click.option(
    "--tasks-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to directory containing task JSON files (and optionally npcs/profiles.json).",
)
@with_command_telemetry("tasks validate", resolver=task_bundle_capture_config)
def validate(tasks_dir: str) -> None:
    """Validate task JSON files in a directory (e.g. a local task bundle)."""
    tasks_path = Path(tasks_dir)
    task_jsons_dir = tasks_path / "tasks"
    if not task_jsons_dir.is_dir():
        click.echo(click.style(f"Tasks directory not found: {task_jsons_dir}", fg="red"), err=True)
        raise SystemExit(1)

    profiles = _load_profiles(tasks_path)
    profile_ids = set(profiles.keys())

    errors: list[str] = []
    warnings: list[str] = []
    task_files = sorted(task_jsons_dir.glob("*.json"))

    for path in task_files:
        fname = path.name
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{fname}: invalid JSON — {exc}")
            continue

        # Required top-level fields
        if "meta" not in data:
            errors.append(f"{fname}: missing 'meta' object")
        else:
            meta = data["meta"]
            for field in _REQUIRED_META_FIELDS:
                if field not in meta:
                    errors.append(f"{fname}: missing meta.{field}")

        if "task" not in data:
            errors.append(f"{fname}: missing 'task' description")

        if "tool_servers" not in data:
            errors.append(f"{fname}: missing 'tool_servers'")

        # Validate NPC profile_ids exist
        for npc in data.get("npcs", []):
            npc_id = npc.get("id", "")
            if npc_id and profile_ids and npc_id not in profile_ids:
                warnings.append(f"{fname}: NPC '{npc_id}' not found in profiles.json")

        # Validate seed_emails from_profile_id and to_addr
        profile_emails = {p.get("email", "").lower() for p in profiles.values() if p.get("email")}
        for em in data.get("seed_emails", []):
            from_id = em.get("from_profile_id", "")
            if from_id and profile_ids and from_id not in profile_ids and "@" not in from_id:
                warnings.append(
                    f"{fname}: seed_email from_profile_id '{from_id}' not found in profiles.json"
                )
            to_addr = (em.get("to_addr") or "").strip().lower()
            if to_addr and profile_emails and to_addr not in profile_emails:
                warnings.append(
                    f"{fname}: seed_email to_addr '{to_addr}' not a known profile email"
                )

        # Validate seed_calendar_events account
        for ev in data.get("seed_calendar_events", []):
            account = ev.get("account", "")
            if account and profile_ids and account not in profile_ids:
                warnings.append(
                    f"{fname}: seed_calendar_events account '{account}' not found in profiles.json"
                )

        # Validate seed_group_channels member_profile_ids and message from_profile_ids
        for ch in data.get("seed_group_channels", []):
            ch_name = ch.get("channel_name", "?")
            for mid in ch.get("member_profile_ids", []):
                if mid and profile_ids and mid not in profile_ids:
                    warnings.append(
                        f"{fname}: seed_group_channels '{ch_name}' member '{mid}' "
                        f"not found in profiles.json"
                    )
            for msg in ch.get("messages", []):
                msg_from = msg.get("from_profile_id", "")
                if msg_from and profile_ids and msg_from not in profile_ids:
                    warnings.append(
                        f"{fname}: seed_group_channels '{ch_name}' message from_profile_id "
                        f"'{msg_from}' not found in profiles.json"
                    )

    # Report
    click.echo(click.style(f"\n  Validation: {tasks_dir}\n", bold=True))
    click.echo(f"  Files scanned:  {len(task_files)}")

    if errors:
        click.echo(click.style(f"\n  Errors ({len(errors)}):", fg="red", bold=True))
        for e in errors:
            click.echo(f"    {click.style('✗', fg='red')} {e}")

    if warnings:
        click.echo(click.style(f"\n  Warnings ({len(warnings)}):", fg="yellow", bold=True))
        for w in warnings:
            click.echo(f"    {click.style('!', fg='yellow')} {w}")

    if not errors and not warnings:
        click.echo(click.style("  All tasks valid.", fg="green"))

    if errors:
        click.echo()
        raise SystemExit(1)

    emit_cli_event(
        "tasks_validate_completed",
        {
            "task_count": len(task_files),
            "warning_count": len(warnings),
        },
    )
    click.echo()
