"""CLI commands for browsing and seeding scenario tasks."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import cast

import click
import yaml

from simlab.agents.mcp_client import MCPClientHandle
from simlab.api.client import ScenarioManagerApiError
from simlab.api.client import ScenarioManagerClient
from simlab.api.client import resolve_scenario_manager_api_url
from simlab.api.schemas import ScenarioTask
from simlab.catalog.registry import ToolRegistry
from simlab.composer.engine import ComposeEngine
from simlab.composer.engine import EnvConfig
from simlab.composer.engine import get_mcp_gateway_host_port
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_agent_api_key
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_daytona_api_key
from simlab.config import resolve_env_dir
from simlab.mcp_config import get_mcp_command_servers
from simlab.mcp_config import get_mcp_server_urls
from simlab.mcp_config import load_mcp_servers_from_env_dir
from simlab.runtime.env_lifecycle import ensure_daytona_sandbox_ready
from simlab.runtime.env_lifecycle import ensure_env_started_daytona
from simlab.runtime.env_lifecycle import ensure_env_started_local
from simlab.runtime.env_lifecycle import env_down_daytona
from simlab.runtime.env_lifecycle import env_down_local
from simlab.runtime.env_lifecycle import env_has_local_services
from simlab.runtime.env_lifecycle import is_env_running_local
from simlab.runtime.env_lifecycle import run_env_seed_daytona
from simlab.runtime.env_lifecycle import run_env_seed_local
from simlab.seeder import get_tool_endpoints
from simlab.seeder import query_tool_server
from simlab.telemetry import TelemetryCaptureConfig
from simlab.telemetry import emit_cli_event
from simlab.telemetry import normalize_config_path
from simlab.telemetry import resolve_scenario_manager_capture_config
from simlab.telemetry import with_command_telemetry

# Maps Docker service names (used in task JSONs) to tool-catalog names.
_SERVICE_TO_TOOL = {
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
    "web-search-env": "web-search",
}
_TOOL_TO_SERVICE = {tool: service for service, tool in _SERVICE_TO_TOOL.items()}

_TOOL_WEB_SERVICE_META = {
    "frappe-hrms": {
        "compose_service": "frappe-hrms",
        "label": "Frappe HRMS",
        "default_port": 8000,
        "credentials": " (login: Administrator / admin)",
    },
    "rocketchat": {
        "compose_service": "rocketchat",
        "label": "Rocket.Chat",
        "default_port": 3000,
        "credentials": " (login: agent / agent123)",
    },
    "email": {
        "compose_service": "mailhog",
        "label": "MailHog",
        "default_port": 8025,
        "credentials": "",
    },
    "calendar": {
        "compose_service": "baikal",
        "label": "Baikal Calendar",
        "default_port": 80,
        "credentials": "",
    },
}

_CALENDAR_INTERNAL_BASE_URL = "http://baikal:80/dav.php"
_CALENDAR_DEFAULT_USERNAME = "chronos"
_CALENDAR_DEFAULT_PASSWORD = "admin"  # noqa: S105 - seeded test credential for Baikal


def get_env_runtime_helpers() -> tuple[Any, Any]:
    """Return the environment helpers used for profiled service orchestration."""
    from simlab.runtime.env_lifecycle import _get_profiled_service_names
    from simlab.runtime.env_lifecycle import _run_profiled_services_local

    return _get_profiled_service_names, _run_profiled_services_local


def get_daytona_runner_class() -> Any:
    """Return the Daytona runner class without importing the SDK on module import."""
    from simlab.runtime.daytona_runner import DaytonaRunner

    return DaytonaRunner


def get_rubric_judge_runner() -> Any:
    """Return the rubric judge runner only when verifier flows need it."""
    from simlab.verifiers import run_rubric_judge

    return run_rubric_judge


def get_agent_runtime_helpers() -> tuple[Any, Any]:
    """Return the external-agent runtime helpers only when tasks run."""
    from simlab.agents import UnifiedToolEnvironment
    from simlab.agents import run_with_agent_contract

    return UnifiedToolEnvironment, run_with_agent_contract


def get_verifier_runtime_helpers() -> tuple[Any, Any, Any]:
    """Return verifier helpers only when task runs need verifier execution."""
    from simlab.verifiers import build_verifier_artifacts
    from simlab.verifiers import infer_scenario_from_evaluator
    from simlab.verifiers import run_verifier

    return build_verifier_artifacts, infer_scenario_from_evaluator, run_verifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_tasks(scenario_dir: Path) -> list[dict[str, Any]]:
    """Load all task JSON files from ``<scenario_dir>/tasks/``."""
    tasks_dir = scenario_dir / "tasks"
    if not tasks_dir.is_dir():
        click.echo(click.style(f"Tasks directory not found: {tasks_dir}", fg="red"), err=True)
        raise SystemExit(1)
    tasks: list[dict[str, Any]] = []
    for p in sorted(tasks_dir.glob("*.json")):
        try:
            tasks.append(json.loads(p.read_text()))
        except json.JSONDecodeError as exc:
            click.echo(
                click.style(f"Invalid JSON in {p}: {exc}", fg="red"),
                err=True,
            )
            raise SystemExit(1) from exc
    return tasks


def _find_task_entry(
    scenario_dir: Path,
    task_id: str,
) -> tuple[dict[str, Any], Path] | None:
    """Find a local task payload and its source file using the same match order."""
    tasks_dir = scenario_dir / "tasks"
    if not tasks_dir.is_dir():
        click.echo(click.style(f"Tasks directory not found: {tasks_dir}", fg="red"), err=True)
        raise SystemExit(1)

    task_entries: list[tuple[dict[str, Any], Path]] = []
    for path in sorted(tasks_dir.glob("*.json")):
        try:
            task_entries.append((json.loads(path.read_text()), path))
        except json.JSONDecodeError as exc:
            click.echo(
                click.style(f"Invalid JSON in {path}: {exc}", fg="red"),
                err=True,
            )
            raise SystemExit(1) from exc

    for task, path in task_entries:
        current_id = task.get("meta", {}).get("task_id", "")
        if current_id == task_id:
            return task, path

    for task, path in task_entries:
        current_id = task.get("meta", {}).get("task_id", "")
        if current_id.endswith(task_id) or task_id in current_id:
            return task, path

    return None


def _find_task(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
    """Find a task by its ``meta.task_id`` (exact or suffix match)."""
    for t in tasks:
        tid = t.get("meta", {}).get("task_id", "")
        if tid == task_id:
            return t
    # Fallback: suffix / partial match
    for t in tasks:
        tid = t.get("meta", {}).get("task_id", "")
        if tid.endswith(task_id) or task_id in tid:
            return t
    return None


def _load_local_task(
    bundle_dir: Path,
    task_id: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path | None]:
    """Load a task, profiles, and source file path from a local bundle."""
    task_entry = _find_task_entry(bundle_dir, task_id)
    if task_entry is None:
        click.echo(click.style(f"Task not found: {task_id}", fg="red"), err=True)
        raise SystemExit(1)
    task_dict, task_file = task_entry
    return task_dict, _load_profiles(bundle_dir), task_file


def _build_mcp_clients(
    mcp_config: dict[str, Any] | None,
    endpoints: dict[str, str],
) -> dict[str, MCPClientHandle]:
    """Build MCP client handles for URL-based and command-based MCP servers."""
    if mcp_config is None:
        return {}

    clients: dict[str, MCPClientHandle] = {}
    for server_name, url in get_mcp_server_urls(mcp_config).items():
        clients[server_name] = MCPClientHandle(url, server_name)

    command_servers = get_mcp_command_servers(mcp_config)
    if command_servers:
        gateway_url = endpoints.get(ComposeEngine.MCP_GATEWAY_SERVICE_NAME) or (
            f"http://localhost:{ComposeEngine.MCP_GATEWAY_PORT}/mcp"
        )
        for server_name in command_servers:
            clients[server_name] = MCPClientHandle(
                gateway_url,
                server_name,
                tool_prefix=f"{server_name}_",
            )
    return clients


def _collect_mcp_tool_failures(
    mcp_clients: dict[str, MCPClientHandle],
) -> list[str]:
    """Return discovery failures for configured MCP clients."""
    failures: list[str] = []
    for server_name, client in mcp_clients.items():
        try:
            tools = asyncio.run(client.alist_tools())
        except Exception as exc:
            failures.append(f"{server_name}: tool discovery failed ({exc})")
            continue
        if not tools:
            failures.append(f"{server_name}: exposed no tools")
    return failures


def _require_mcp_tools_available(
    mcp_clients: dict[str, MCPClientHandle],
) -> None:
    """Fail fast when configured MCP servers cannot enumerate any tools."""
    if not mcp_clients:
        return

    failures = _collect_mcp_tool_failures(mcp_clients)

    if failures:
        click.echo(
            click.style(
                "Configured MCP servers are not usable for task run:",
                fg="red",
            ),
            err=True,
        )
        for failure in failures:
            click.echo(f"  - {failure}", err=True)
        click.echo(
            click.style(
                "Check the MCP server command/args or upstream authentication, then retry.",
                fg="yellow",
            ),
            err=True,
        )
        raise SystemExit(1)


def _wait_for_mcp_tools_available(
    mcp_clients: dict[str, MCPClientHandle],
    *,
    timeout: int = 120,
    poll_interval: int = 5,
    log_prefix: str = "",
) -> None:
    """Poll MCP tool discovery until every configured client exposes tools."""
    if not mcp_clients:
        return

    prefix = f"{log_prefix} " if log_prefix else ""
    deadline = time.monotonic() + timeout
    failures = _collect_mcp_tool_failures(mcp_clients)
    while failures:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{prefix}Timed out waiting for MCP tools: {'; '.join(failures)}")
        click.echo(f"{prefix}Waiting for MCP tools...")
        time.sleep(poll_interval)
        failures = _collect_mcp_tool_failures(mcp_clients)


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


def _get_local_verifier_file_path(tasks_dir: Path, module_path: str) -> Path | None:
    """Return the local verifier file matching a task's module path."""
    verifiers_dir = tasks_dir / "verifiers"
    if not verifiers_dir.is_dir():
        return None
    module_stem = module_path.rsplit(".", 1)[-1].strip()
    if not module_stem:
        return None
    verifier_file = verifiers_dir / f"{module_stem}.py"
    if verifier_file.is_file():
        return verifier_file
    return None


def _is_test_task(task: dict[str, Any]) -> bool:
    """Return True when a local task should be treated as a test task."""
    meta = task.get("meta", {})
    category = meta.get("category", task.get("category", ""))
    return str(category or "").strip().lower() == "test"


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


def _load_profiles(scenario_dir: Path) -> dict[str, dict[str, Any]]:
    """Load NPC profiles, keyed by ``profile_id``."""
    profiles_path = scenario_dir / "npcs" / "profiles.json"
    if not profiles_path.exists():
        return {}
    try:
        data = json.loads(profiles_path.read_text())
    except json.JSONDecodeError:
        return {}
    return {p["profile_id"]: p for p in data if "profile_id" in p}


def _load_skills_markdown(
    *,
    config: EnvConfig,
    bundle_dir: Path | None,
) -> str | None:
    """Load scenario guidance from the environment config."""
    _ = bundle_dir
    if config.scenario_guidance_md:
        content = config.scenario_guidance_md.strip()
        if content:
            return content
    return None


def _build_skills_guidance_section(skills_md: str | None) -> str:
    """Format scenario-level skills guidance for the task instruction."""
    if not skills_md:
        return ""
    return f"Scenario guidance:\n{skills_md}"


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


def _query_tool_server_with_retries(
    endpoint_url: str,
    tool_name: str,
    params: dict[str, Any],
    *,
    retries: int = 3,
) -> Any:
    """Query a tool server with simple retry/backoff."""
    for attempt in range(1, retries + 1):
        resp = query_tool_server(endpoint_url, tool_name, params)
        if resp is not None:
            return resp
        if attempt < retries:
            time.sleep(1.0 * attempt)
    return None


def _parse_tool_response_payload(resp: Any) -> Any:
    """Best-effort structured payload extraction from a tool response."""
    if not isinstance(resp, dict):
        return None

    obs = resp.get("observation")
    if isinstance(obs, dict):
        structured = obs.get("structured_content")
        if structured is not None:
            return structured
        text = obs.get("text")
        if isinstance(text, str):
            stripped = text.strip()
            if not stripped:
                return None
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
        return obs

    if isinstance(obs, str):
        stripped = obs.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped

    return None


def _tool_response_is_error(
    resp: Any,
    *,
    allowed_error_codes: set[str] | None = None,
) -> bool:
    """Return True when a tool response represents a failed action."""
    if resp is None or not isinstance(resp, dict):
        return True

    allowed = allowed_error_codes or set()
    obs = resp.get("observation")
    payload = _parse_tool_response_payload(resp)

    error_code = ""
    payload_error = False
    if isinstance(payload, dict):
        raw_error_code = payload.get("error_code")
        if isinstance(raw_error_code, str):
            error_code = raw_error_code.strip()
        payload_error = payload.get("success") is False or bool(payload.get("error"))

    if isinstance(obs, dict) and bool(obs.get("is_error")):
        return error_code not in allowed
    if error_code:
        return error_code not in allowed
    return payload_error


def _task_uses_calendar(task: dict[str, Any]) -> bool:
    """Return True when the task references the calendar tool."""
    apps = task.get("apps", [])
    if isinstance(apps, list) and "calendar" in apps:
        return True
    for tool_server in task.get("tool_servers", []):
        if not isinstance(tool_server, dict):
            continue
        if _SERVICE_TO_TOOL.get(tool_server.get("name", "")) == "calendar":
            return True
    return False


def needed_task_endpoints(
    task: dict[str, Any],
    endpoints: dict[str, str],
    *,
    include_tool_servers: bool,
) -> dict[str, str]:
    """Return only the endpoints a task seed or run path will actually call."""
    needed: dict[str, str] = {}
    if task.get("seed_emails") and "email" in endpoints:
        needed["email"] = endpoints["email"]

    if (
        task.get("seed_calendar_events")
        or (_task_uses_calendar(task) and _collect_task_calendar_accounts(task))
    ) and "calendar" in endpoints:
        needed["calendar"] = endpoints["calendar"]

    if not include_tool_servers:
        return needed

    for tool_server in task.get("tool_servers", []):
        if not isinstance(tool_server, dict):
            continue
        tool_name = _SERVICE_TO_TOOL.get(str(tool_server.get("name") or "").strip())
        if tool_name and tool_name in endpoints:
            needed[tool_name] = endpoints[tool_name]
    return needed


def _profile_display_name(profile_id: str, profile: dict[str, Any]) -> str:
    """Return a human-readable display name for a profile/account."""
    first_name = str(profile.get("first_name") or "").strip()
    last_name = str(profile.get("last_name") or "").strip()
    display_name = f"{first_name} {last_name}".strip()
    return display_name or profile_id.replace("_", " ").title()


def _collect_task_calendar_accounts(
    task: dict[str, Any],
) -> list[str]:
    """Collect task-specific calendar account aliases to register."""
    accounts: list[str] = []
    for npc in task.get("npcs", []):
        if not isinstance(npc, dict):
            continue
        npc_id = str(npc.get("id") or "").strip()
        if npc_id:
            accounts.append(npc_id)
    for event in task.get("seed_calendar_events", []):
        if not isinstance(event, dict):
            continue
        account = str(event.get("account") or "").strip()
        if account:
            accounts.append(account)
    return list(dict.fromkeys(accounts))


def _compose_dir_from_config(config_path: str | Path) -> Path:
    """Return the env directory (parent of env.yaml)."""
    return Path(config_path).parent.resolve()


def _resolve_calendar_account_settings(config: EnvConfig) -> tuple[str, str, str]:
    """Resolve the internal CalDAV URL and shared credentials for account registration."""
    overrides = config.overrides.get("calendar", {})
    base_url = str(overrides.get("CALDAV_BASE_URL", _CALENDAR_INTERNAL_BASE_URL)).strip()
    username = str(overrides.get("CALDAV_USERNAME", _CALENDAR_DEFAULT_USERNAME)).strip()
    password = str(overrides.get("CALDAV_PASSWORD", _CALENDAR_DEFAULT_PASSWORD)).strip()
    return (
        base_url or _CALENDAR_INTERNAL_BASE_URL,
        username or _CALENDAR_DEFAULT_USERNAME,
        password or _CALENDAR_DEFAULT_PASSWORD,
    )


def _extract_calendar_account_aliases(resp: Any) -> set[str]:
    """Extract registered calendar account aliases from ``list_accounts`` output."""
    payload = _parse_tool_response_payload(resp)
    if not isinstance(payload, dict):
        return set()
    accounts = payload.get("accounts")
    if not isinstance(accounts, list):
        return set()

    aliases: set[str] = set()
    for account in accounts:
        if not isinstance(account, dict):
            continue
        alias = str(account.get("alias") or account.get("name") or "").strip()
        if alias:
            aliases.add(alias)
    return aliases


def _calendar_account_is_connected(resp: Any) -> bool:
    """Return True when a ``test_account`` response says the account is connected."""
    if _tool_response_is_error(resp):
        return False
    payload = _parse_tool_response_payload(resp)
    if isinstance(payload, dict):
        connected = payload.get("connected")
        if isinstance(connected, bool):
            return connected
        status = payload.get("status")
        if isinstance(status, str):
            return status.strip().lower() == "connected"
    return False


def _resolve_calendar_uid(
    cal_url: str,
    *,
    account: str,
    calendar_id: str,
    cache: dict[tuple[str, str], str],
) -> str | None:
    """Resolve a task ``calendar_id`` to the Chronos ``calendar_uid`` for an account."""
    cache_key = (account, calendar_id)
    if cache_key in cache:
        return cache[cache_key]

    resp = _query_tool_server_with_retries(
        cal_url,
        "list_calendars",
        {"account": account},
    )
    if _tool_response_is_error(resp):
        return None

    payload = _parse_tool_response_payload(resp)
    if not isinstance(payload, dict):
        return None

    calendars = payload.get("calendars")
    if not isinstance(calendars, list):
        return None

    requested = calendar_id.strip()
    fallback_uid: str | None = None
    for calendar in calendars:
        if not isinstance(calendar, dict):
            continue
        uid = str(calendar.get("uid") or "").strip()
        if not uid:
            continue
        if fallback_uid is None:
            fallback_uid = uid
        candidates = {
            uid,
            str(calendar.get("name") or "").strip(),
            str(calendar.get("display_name") or "").strip(),
            str(calendar.get("id") or "").strip(),
        }
        if requested in candidates:
            cache[cache_key] = uid
            return uid

    if requested == "default" and fallback_uid is not None and len(calendars) == 1:
        cache[cache_key] = fallback_uid
        return fallback_uid
    return None


def _provision_task_calendar_users(
    task: dict[str, Any],
    config: EnvConfig,
    config_path: str,
    *,
    using_daytona: bool,
    daytona_api_key: str | None = None,
) -> None:
    """Ensure Baikal has backing CalDAV users/calendars for task-specific accounts."""
    if not _task_uses_calendar(task):
        return

    accounts = [
        account
        for account in _collect_task_calendar_accounts(task)
        if account and account != _CALENDAR_DEFAULT_USERNAME
    ]
    if not accounts:
        return

    config_file = Path(config_path)
    compose_dir = _compose_dir_from_config(config_file)
    env_overrides = {"CALDAV_USERS": ",".join(accounts)}
    get_profiled_service_names, run_profiled_services_local = get_env_runtime_helpers()

    preseed_svc_names = get_profiled_service_names(
        config,
        profile="preseed",
        config_path=config_file,
        tool_names=["calendar"],
    )
    seed_svc_names = get_profiled_service_names(
        config,
        profile="seed",
        config_path=config_file,
        tool_names=["calendar"],
    )

    click.echo(f"  Provisioning calendar users: {', '.join(accounts)}")
    if using_daytona:
        DaytonaRunner = get_daytona_runner_class()
        runner = DaytonaRunner(daytona_api_key=daytona_api_key)
        if preseed_svc_names:
            runner.run_profiled_services(
                compose_dir,
                preseed_svc_names,
                profile="preseed",
                env_overrides=env_overrides,
            )
        if seed_svc_names:
            runner.run_profiled_services(
                compose_dir,
                seed_svc_names,
                profile="seed",
                env_overrides=env_overrides,
            )
        return

    if preseed_svc_names:
        run_profiled_services_local(
            compose_dir,
            preseed_svc_names,
            profile="preseed",
            env_overrides=env_overrides,
        )
    if seed_svc_names:
        run_profiled_services_local(
            compose_dir,
            seed_svc_names,
            profile="seed",
            env_overrides=env_overrides,
        )


def _provision_task_group_channels(
    task: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    config: EnvConfig,
    config_path: str,
    *,
    using_daytona: bool,
    daytona_api_key: str | None = None,
) -> None:
    """Re-run rocketchat-seed with task-specific NPC users and group channels.

    Generated tasks may reference NPC users and group channels that don't exist
    in Rocket.Chat yet.  This function re-runs the rocketchat-seed container
    with ``ROCKETCHAT_NPC_CONFIGS`` (so the users get created) and
    ``ROCKETCHAT_SEED_GROUP_CHANNELS`` (so the channels get created, members
    get invited, and seed messages get posted).
    """
    group_channels = task.get("seed_group_channels") or []
    if not group_channels:
        return

    if "rocketchat" not in config.tools:
        return

    # Collect all NPC profile IDs referenced by the task (from npcs + group channel members).
    npc_ids: list[str] = []
    for npc in task.get("npcs", []):
        if isinstance(npc, dict):
            npc_id = str(npc.get("id") or "").strip()
            if npc_id:
                npc_ids.append(npc_id)
    for ch in group_channels:
        if isinstance(ch, dict):
            for mid in ch.get("member_profile_ids", []):
                mid_str = str(mid).strip()
                if mid_str:
                    npc_ids.append(mid_str)
    npc_ids = list(dict.fromkeys(npc_ids))  # unique, order-preserving

    if not npc_ids:
        return

    # Build NPC credentials in the format rocketchat-seed expects.
    # Use rocketchat_username from profile (dots) when available, falling
    # back to profile_id (underscores).  This matches the server-side
    # _build_npc_configs in workspace_controller.py.
    #
    # No client-side dedup of usernames/emails here: npc_ids are already
    # deduplicated by profile_id (line 780), and the source profile data
    # guarantees unique rocketchat_usernames and emails.  Silent renaming
    # would mask bad data — if a collision somehow occurs, the seed script
    # (seed_rocketchat.py create_user) will surface the HTTP 400 so the
    # profile data can be fixed at the source.
    npc_configs: dict[str, dict[str, Any]] = {}
    for pid in npc_ids:
        profile = profiles.get(pid, {})
        rc_username = str(profile.get("rocketchat_username") or "").strip() or pid
        first = str(profile.get("first_name") or "").strip()
        last = str(profile.get("last_name") or "").strip()
        display_name = (
            f"{first} {last}".strip() if (first or last) else pid.replace("_", " ").title()
        )
        email = str(profile.get("email") or f"{rc_username}@example.com").strip()
        npc_configs[pid] = {
            "username": rc_username,
            "password": "npc123",  # deterministic — matches CLI agent password pattern
            "name": display_name,
            "email": email,
        }

    env_overrides: dict[str, str] = {
        "ROCKETCHAT_NPC_CONFIGS": json.dumps(npc_configs),
        "ROCKETCHAT_SEED_GROUP_CHANNELS": json.dumps(group_channels),
    }

    config_file = Path(config_path)
    compose_dir = _compose_dir_from_config(config_file)
    get_profiled_service_names, run_profiled_services_local = get_env_runtime_helpers()

    seed_svc_names = get_profiled_service_names(
        config,
        profile="seed",
        config_path=config_file,
        tool_names=["rocketchat"],
    )
    if not seed_svc_names:
        return

    channel_names = [ch.get("channel_name", "?") for ch in group_channels if isinstance(ch, dict)]
    click.echo(
        f"  Provisioning group channels: {', '.join(channel_names)} ({len(npc_ids)} NPC user(s))"
    )

    if using_daytona:
        DaytonaRunner = get_daytona_runner_class()
        runner = DaytonaRunner(daytona_api_key=daytona_api_key)
        runner.run_profiled_services(
            compose_dir,
            seed_svc_names,
            profile="seed",
            env_overrides=env_overrides,
        )
        return

    run_profiled_services_local(
        compose_dir,
        seed_svc_names,
        profile="seed",
        env_overrides=env_overrides,
    )


def run_env_seed_services(
    config: EnvConfig,
    config_path: str,
    *,
    using_daytona: bool,
    daytona_api_key: str | None = None,
) -> None:
    """Run environment seed services so mutable tools start each run from seed state."""
    config_file = Path(config_path)
    compose_dir = _compose_dir_from_config(config_file)
    get_profiled_service_names, run_profiled_services_local = get_env_runtime_helpers()
    seed_svc_names = get_profiled_service_names(config, profile="seed", config_path=config_file)
    if not seed_svc_names:
        return

    click.echo("  Resetting environment state from seed services")
    if using_daytona:
        DaytonaRunner = get_daytona_runner_class()
        runner = DaytonaRunner(daytona_api_key=daytona_api_key)
        runner.run_profiled_services(compose_dir, seed_svc_names, profile="seed")
        return

    run_profiled_services_local(compose_dir, seed_svc_names, profile="seed")


def _ensure_task_calendar_accounts(
    task: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    endpoints: dict[str, str],
    config: EnvConfig,
) -> None:
    """Register task-specific calendar accounts before seed/run."""
    if not _task_uses_calendar(task):
        return

    cal_url = endpoints.get("calendar")
    if not cal_url:
        return

    required_accounts = _collect_task_calendar_accounts(task)
    if not required_accounts:
        return

    caldav_base_url, default_username, caldav_password = _resolve_calendar_account_settings(config)

    existing_aliases: set[str] = set()
    list_resp = _query_tool_server_with_retries(cal_url, "list_accounts", {})
    if not _tool_response_is_error(list_resp):
        existing_aliases = _extract_calendar_account_aliases(list_resp)

    for account in required_accounts:
        if account == default_username:
            continue

        profile = profiles.get(account, {})
        test_resp = None
        if account in existing_aliases:
            test_resp = _query_tool_server_with_retries(
                cal_url,
                "test_account",
                {"alias": account},
            )
            if _calendar_account_is_connected(test_resp):
                click.echo(f"  {click.style('✓', fg='green')} Calendar account: [{account}] ready")
                continue

        resp = _query_tool_server_with_retries(
            cal_url,
            "add_account",
            {
                "alias": account,
                "url": caldav_base_url,
                "username": account,
                "password": caldav_password,
                "display_name": _profile_display_name(account, profile),
            },
        )
        if _tool_response_is_error(resp, allowed_error_codes={"ACCOUNT_EXISTS"}):
            payload = _parse_tool_response_payload(resp)
            detail = (
                payload if isinstance(payload, str) else json.dumps(payload or resp, default=str)
            )
            click.echo(
                click.style(
                    f"  ✗ Calendar account failed: {account} ({detail})",
                    fg="red",
                )
            )
            raise SystemExit(1)

        test_resp = _query_tool_server_with_retries(
            cal_url,
            "test_account",
            {"alias": account},
        )
        if not _calendar_account_is_connected(test_resp):
            payload = _parse_tool_response_payload(test_resp)
            detail = (
                payload
                if isinstance(payload, str)
                else json.dumps(payload or test_resp, default=str)
            )
            click.echo(
                click.style(
                    f"  ✗ Calendar account failed: {account} ({detail})",
                    fg="red",
                )
            )
            raise SystemExit(1)

        existing_aliases.add(account)
        click.echo(f"  {click.style('✓', fg='green')} Calendar account: [{account}] ready")


def _seed_task_data(
    task: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    endpoints: dict[str, str],
) -> tuple[int, int]:
    """Seed task emails and calendar events. Returns ``(ok, fail)`` counts."""
    emails = task.get("seed_emails", [])
    cal_events = task.get("seed_calendar_events", [])
    ok, fail = 0, 0
    calendar_uid_cache: dict[tuple[str, str], str] = {}

    if emails:
        email_url = endpoints.get("email")
        if not email_url:
            click.echo(click.style("  No email tool server in environment config.", fg="yellow"))
        else:
            for em in emails:
                from_id = em.get("from_profile_id", "")
                profile = profiles.get(from_id, {})
                if profile:
                    from_addr = profile.get("email", f"{from_id}@unknown.com")
                elif "@" in from_id:
                    from_addr = from_id
                else:
                    from_addr = f"{from_id}@unknown.com"
                    click.echo(
                        click.style(
                            f"  ! from_profile_id '{from_id}' not in profiles, "
                            f"falling back to {from_addr}",
                            fg="yellow",
                        )
                    )

                params: dict[str, Any] = {
                    "from_email": from_addr,
                    "to": em["to_addr"],
                    "subject": em["subject"],
                    "body": em.get("body_text", ""),
                }
                if em.get("body_html"):
                    params["body_html"] = em["body_html"]

                resp = _query_tool_server_with_retries(email_url, "send_email", params)
                if not _tool_response_is_error(resp):
                    ok += 1
                    click.echo(
                        f"  {click.style('✓', fg='green')} Email: "
                        f'{from_addr} → {em["to_addr"]}  "{em["subject"]}"'
                    )
                else:
                    fail += 1
                    click.echo(
                        click.style(
                            f'  ✗ Email failed: "{em["subject"]}" (endpoint: {email_url})',
                            fg="red",
                        )
                    )

    if cal_events:
        cal_url = endpoints.get("calendar")
        if not cal_url:
            click.echo(click.style("  No calendar tool server in environment config.", fg="yellow"))
        else:
            for ev in cal_events:
                account = str(ev.get("account") or "").strip()
                calendar_id = str(ev.get("calendar_id", "default")).strip() or "default"
                calendar_uid = _resolve_calendar_uid(
                    cal_url,
                    account=account,
                    calendar_id=calendar_id,
                    cache=calendar_uid_cache,
                )
                if not calendar_uid:
                    fail += 1
                    click.echo(
                        click.style(
                            f"  ✗ Calendar UID lookup failed: [{account}] {calendar_id} (endpoint: {cal_url})",
                            fg="red",
                        )
                    )
                    continue
                params = {
                    "account": account,
                    "calendar_uid": calendar_uid,
                    "summary": ev["summary"],
                    "description": ev.get("description", ""),
                    "start": ev["start"],
                    "end": ev["end"],
                }
                resp = _query_tool_server_with_retries(cal_url, "create_event", params)
                if not _tool_response_is_error(resp):
                    ok += 1
                    click.echo(
                        f"  {click.style('✓', fg='green')} Calendar: "
                        f"[{ev['account']}] {ev['summary']}  {ev['start']}"
                    )
                else:
                    fail += 1
                    click.echo(
                        click.style(
                            f"  ✗ Calendar event failed: {ev['summary']} (endpoint: {cal_url})",
                            fg="red",
                        )
                    )

    return ok, fail


def _rewrite_tool_server_urls(
    task_data: dict[str, Any],
    endpoints: dict[str, str],
) -> dict[str, Any]:
    """Replace Docker-internal tool_server_urls with actual endpoints.

    Uses ``_SERVICE_TO_TOOL`` to match Docker service names (e.g. ``email-env``)
    to tool-catalog names (e.g. ``email``), then looks up the endpoint URL from
    *endpoints*.  Falls back to replacing the hostname with ``localhost``.
    """
    task_copy = copy.deepcopy(task_data)
    for ts in task_copy.get("tool_servers", []):
        service_name = ts.get("name", "")
        tool_name = _SERVICE_TO_TOOL.get(service_name)
        if tool_name and tool_name in endpoints:
            ts["tool_server_url"] = endpoints[tool_name]
        else:
            # Fallback: replace hostname with localhost, keep port
            url = ts.get("tool_server_url", "")
            ts["tool_server_url"] = re.sub(r"http://[^:]+:(\d+)", r"http://localhost:\1", url)
    return task_copy


def _effective_tool_servers(
    rewritten_task: dict[str, Any],
    endpoints: dict[str, str],
) -> dict[str, str]:
    """Return merged tool-server URLs from task JSON plus environment endpoints."""
    merged: dict[str, str] = {}
    for server in rewritten_task.get("tool_servers", []):
        name = server.get("name")
        url = server.get("tool_server_url")
        if name and url:
            merged[name] = url

    for tool_name, url in endpoints.items():
        service_name = _TOOL_TO_SERVICE.get(tool_name, tool_name)
        merged.setdefault(service_name, url)
    return merged


def _verifier_tool_servers(
    rewritten_task: dict[str, Any],
    endpoints: dict[str, str],
    mcp_clients: dict[str, MCPClientHandle] | None = None,
) -> dict[str, str]:
    """Return verifier-facing tool URLs across HTTP and MCP namespaces.

    Verifiers historically expect an HTTP-style base URL. For MCP endpoints, use the
    HTTP base when the client URL ends with ``/mcp`` so existing verifiers can keep
    constructing ``/step`` URLs when the underlying server supports both transports.
    TODO: replace this compatibility shim once verifiers can consume MCP-native tool
    calls directly instead of assuming the legacy HTTP ``/step`` contract.
    """
    merged = _effective_tool_servers(rewritten_task, endpoints)
    for server_name, client in (mcp_clients or {}).items():
        url = getattr(client, "_url", None)
        if not isinstance(url, str) or not url:
            continue
        merged.setdefault(
            server_name,
            url.removesuffix("/mcp"),
        )
    return merged


def _compose_service_host_port(compose_data: dict[str, Any], service_name: str) -> int | None:
    """Extract mapped host port for a docker-compose service, if available."""
    services = compose_data.get("services", {})
    if not isinstance(services, dict):
        return None
    service = services.get(service_name)
    if not isinstance(service, dict):
        return None
    ports = service.get("ports", [])
    if not isinstance(ports, list):
        return None
    for entry in ports:
        if isinstance(entry, str) and ":" in entry:
            host = entry.split(":", 1)[0].strip().strip('"').strip("'")
            if host.isdigit():
                return int(host)
    return None


def _build_services_available_section(
    config: EnvConfig,
    *,
    daytona: bool,
    config_path: str,
    endpoints: dict[str, str],
) -> str:
    """Build website access notes appended to the task prompt.

    This mirrors the instructions we typically embed in generated task prompts so
    the built-in agent has direct UI endpoints/credentials when needed.
    """
    lines: list[str] = []
    if daytona:
        for tool in config.tools:
            endpoint = endpoints.get(tool)
            if endpoint:
                lines.append(f"* {tool}: {endpoint}")
        if not lines:
            return ""
        return "Tool endpoints available to you:\n" + "\n".join(lines)
    compose_path = Path(config_path).parent / "docker-compose.yml"
    compose_data: dict[str, Any] = {}
    if compose_path.exists():
        loaded = yaml.safe_load(compose_path.read_text())
        if isinstance(loaded, dict):
            compose_data = loaded

    lines.clear()
    for tool in config.tools:
        meta = _TOOL_WEB_SERVICE_META.get(tool)
        if not meta:
            continue
        compose_svc = meta["compose_service"]
        default_port = meta["default_port"]
        host_port = _compose_service_host_port(compose_data, str(compose_svc))
        if host_port is None:
            host_port = cast(int, default_port)
        line = f"* {meta['label']}: http://localhost:{host_port}{meta['credentials']}"
        lines.append(line)
    if not lines:
        return ""
    return "Services available to you as websites:\n" + "\n".join(lines)


def _get_daytona_client(daytona_api_key: str | None = None) -> Any:
    """Construct Daytona SDK client lazily so local-only use has no hard dependency."""
    try:
        daytona_mod = importlib.import_module("daytona")
        Daytona = daytona_mod.Daytona
        DaytonaConfig = daytona_mod.DaytonaConfig
    except Exception as exc:  # pragma: no cover - import path depends on env
        click.echo(
            click.style(
                f"Daytona SDK is required for Daytona mode: {exc}",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    api_key = resolve_daytona_api_key(daytona_api_key)
    if not api_key:
        click.echo(
            click.style(
                "Daytona API key is required for Daytona mode via --daytona-api-key, "
                "config, SIMLAB_DAYTONA_API_KEY, or DAYTONA_API_KEY.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)
    return Daytona(DaytonaConfig(api_key=api_key))


def _sandbox_status(sandbox: Any) -> str:
    """Best-effort status extraction across Daytona SDK versions."""
    for attr in ("state", "status"):
        value = getattr(sandbox, attr, None)
        if value is None:
            continue
        if isinstance(value, str):
            return value.lower()
        value_name = getattr(value, "value", None)
        if isinstance(value_name, str):
            return value_name.lower()
        return str(value).lower()
    return "unknown"


def _is_active_sandbox_status(status: str) -> bool:
    return status in {"running", "started", "active", "ready"}


def _invoke_daytona_resume(daytona: Any, sandbox: Any, sandbox_id: str) -> bool:
    """Try multiple SDK-compatible resume/start methods."""
    candidates = [
        (sandbox, "resume", ()),
        (sandbox, "start", ()),
        (daytona, "resume", (sandbox_id,)),
        (daytona, "start", (sandbox_id,)),
    ]
    for obj, method_name, args in candidates:
        method = getattr(obj, method_name, None)
        if method is None:
            continue
        try:
            method(*args)
            return True
        except TypeError:
            # Some SDK variants expose daytona.start(sandbox=...) etc.; skip safely.
            continue
        except Exception:
            continue
    return False


def _resume_daytona_sandbox_interactively(
    daytona: Any, sandbox: Any, sandbox_id: str, status: str
) -> Any:
    """Prompt user to resume a stopped/paused sandbox and wait until active."""
    should_resume = False
    try:
        should_resume = click.confirm(
            f"Daytona sandbox is '{status}'. Resume it now?",
            default=True,
        )
    except (click.Abort, EOFError):
        should_resume = False

    if not should_resume:
        click.echo(
            click.style(
                "Run `simlab env up <env-name> --daytona` before tasks run/seed.",
                fg="yellow",
            ),
            err=True,
        )
        raise SystemExit(1)

    click.echo("Resuming Daytona sandbox...", err=True)
    if not _invoke_daytona_resume(daytona, sandbox, sandbox_id):
        click.echo(
            click.style(
                "Unable to resume sandbox via Daytona SDK. "
                "Run `simlab env up <env-name> --daytona`.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    deadline = time.time() + 60
    while time.time() < deadline:
        refreshed = daytona.get(sandbox_id)
        refreshed_status = _sandbox_status(refreshed)
        if _is_active_sandbox_status(refreshed_status):
            click.echo(click.style(f"Sandbox resumed ({refreshed_status}).", fg="green"), err=True)
            return refreshed
        time.sleep(2)

    click.echo(
        click.style(
            "Timed out waiting for Daytona sandbox to become active. "
            "Try `simlab env up <env-name> --daytona`.",
            fg="red",
        ),
        err=True,
    )
    raise SystemExit(1)


def _get_daytona_endpoints(
    config_path: str,
    daytona_api_key: str | None = None,
    *,
    allow_resume: bool = True,
) -> dict[str, str]:
    """Get tool endpoint URLs from a running Daytona sandbox via Python SDK."""
    config_file = Path(config_path)
    if not config_file.exists():
        click.echo(click.style(f"Config not found: {config_file}", fg="red"), err=True)
        raise SystemExit(1)

    data = yaml.safe_load(config_file.read_text())
    config = EnvConfig(**data)
    local_endpoints = get_tool_endpoints(config, config_path=config_file)
    env_dir = config_file.parent
    mcp_config = load_mcp_servers_from_env_dir(env_dir)
    if not _requires_daytona_sandbox(config, mcp_config):
        return local_endpoints

    state_file = Path(config_path).parent / "daytona-state.json"
    if not state_file.exists():
        click.echo(
            click.style(f"No Daytona state at {state_file}. Is the env running?", fg="red"),
            err=True,
        )
        raise SystemExit(1)
    state = json.loads(state_file.read_text())
    sandbox_id = state.get("sandbox_id")
    if not sandbox_id:
        click.echo(click.style("Invalid Daytona state: missing sandbox_id", fg="red"), err=True)
        raise SystemExit(1)

    daytona = _get_daytona_client(daytona_api_key=daytona_api_key)
    sandbox = daytona.get(sandbox_id)
    status = _sandbox_status(sandbox)
    resumed = False
    if not _is_active_sandbox_status(status):
        if not allow_resume:
            click.echo(
                click.style(
                    f"Daytona sandbox is '{status}' and --skip-env-setup prevents auto-resume. "
                    "Start it manually with `simlab env up <env-name> --daytona`.",
                    fg="red",
                ),
                err=True,
            )
            raise SystemExit(1)
        sandbox = _resume_daytona_sandbox_interactively(daytona, sandbox, sandbox_id, status)
        resumed = True
    if resumed:
        get_profiled_service_names, _ = get_env_runtime_helpers()
        preseed_svc_names = get_profiled_service_names(config, "preseed", config_file)
        DaytonaRunner = get_daytona_runner_class()
        runner = DaytonaRunner(daytona_api_key=daytona_api_key)
        try:
            runner.restart_sandbox_services(
                sandbox,
                preseed_svc_names=preseed_svc_names,
            )
        except RuntimeError as exc:
            click.echo(
                click.style(
                    "Unable to restart services in the resumed Daytona sandbox.",
                    fg="red",
                ),
                err=True,
            )
            click.echo(click.style(str(exc), fg="red"), err=True)
            click.echo(
                click.style(
                    f"Run `simlab env up {config.name} --daytona` to rebuild and restart it.",
                    fg="yellow",
                ),
                err=True,
            )
            raise SystemExit(1) from exc

    registry = ToolRegistry()
    registry.load_all()

    urls: dict[str, str] = {}
    for tool_name in config.tools:
        tool = registry.get_tool(tool_name)
        if tool is None:
            continue
        if tool.tool_server_port is not None:
            preview = sandbox.get_preview_link(tool.tool_server_port)
            urls[tool_name] = preview.url
        elif tool.tool_server_url:
            urls[tool_name] = tool.tool_server_url

    # When env has command-based MCP servers, gateway runs in the sandbox; add its URL.
    if mcp_config and get_mcp_command_servers(mcp_config):
        preview = sandbox.get_preview_link(get_mcp_gateway_host_port(env_dir))
        urls[ComposeEngine.MCP_GATEWAY_SERVICE_NAME] = preview.url.rstrip("/") + "/mcp"
    return urls


def _requires_daytona_sandbox(
    config: EnvConfig,
    mcp_config: dict[str, Any] | None,
) -> bool:
    """Return True when the env needs a Daytona sandbox for tool execution."""
    registry = ToolRegistry()
    registry.load_all()
    for tool_name in config.tools:
        tool = registry.get_tool(tool_name)
        if tool is not None and tool.tool_server_port is not None:
            return True
    return bool(mcp_config and get_mcp_command_servers(mcp_config))


def _endpoint_has_tools(base_url: str) -> bool:
    """Return True if the endpoint responds to /tools with a tools list."""
    req = urllib.request.Request(f"{base_url}/tools")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("tools"), list)


def _endpoint_is_reachable(url: str) -> bool:
    """Return True when an HTTP endpoint responds at all."""
    req = urllib.request.Request(url)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=5):  # noqa: S310
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, TimeoutError):
        return False


def _any_endpoint_reachable(endpoints: dict[str, str]) -> bool:
    return any(_endpoint_has_tools(url) for url in endpoints.values())


def _reachable_endpoints(endpoints: dict[str, str]) -> dict[str, bool]:
    """Return per-endpoint reachability map via /tools."""
    return {name: _endpoint_has_tools(url) for name, url in endpoints.items()}


def _daytona_state_file_for_config(config_path: str | None) -> Path | None:
    """Return the Daytona state file path for an env config, if applicable."""
    if not config_path:
        return None
    return Path(config_path).parent / "daytona-state.json"


def _require_reachable_endpoints(
    *,
    endpoints: dict[str, str],
    action: str,
    using_daytona: bool,
    config_path: str | None = None,
    wait: bool = False,
    timeout: int = 120,
    poll_interval: int = 5,
    log_prefix: str = "",
) -> None:
    """Fail fast when endpoints are not reachable before seed/run.

    When *wait* is ``True``, poll until all endpoints respond or *timeout*
    seconds elapse (useful after ``docker compose up -d`` in a fresh sandbox).
    After polling (or immediately when ``wait=False``), raise ``SystemExit(1)``
    if **zero** endpoints are reachable. Skips the MCP gateway's /tools check,
    but still requires the gateway itself to respond when it is the only
    configured endpoint.
    """
    if not endpoints:
        click.echo(click.style(f"No endpoints resolved for {action}.", fg="red"), err=True)
        raise SystemExit(1)

    prefix = f"{log_prefix} " if log_prefix else ""

    if wait:
        deadline = time.monotonic() + timeout
        while True:
            reachability = _reachable_endpoints(endpoints)
            check = {
                k: v for k, v in reachability.items() if k != ComposeEngine.MCP_GATEWAY_SERVICE_NAME
            }
            if not check or all(check.values()):
                return
            remaining = [n for n, ok in check.items() if not ok]
            if time.monotonic() >= deadline:
                click.echo(
                    click.style(
                        f"{prefix}Timed out waiting for endpoints: {', '.join(remaining)}",
                        fg="yellow",
                    ),
                    err=True,
                )
                break  # fall through to fail-fast check below
            click.echo(f"{prefix}Waiting for: {', '.join(remaining)}...")
            time.sleep(poll_interval)
    else:
        reachability = _reachable_endpoints(endpoints)

    # MCP gateway does not expose /tools; exclude it from reachability requirement
    check = {k: v for k, v in reachability.items() if k != ComposeEngine.MCP_GATEWAY_SERVICE_NAME}
    if any(check.values()):
        return
    gateway_url = endpoints.get(ComposeEngine.MCP_GATEWAY_SERVICE_NAME)
    if not check and gateway_url and _endpoint_is_reachable(gateway_url):
        return

    mode = "daytona" if using_daytona else "local"
    click.echo(
        click.style(
            f"None of the resolved {mode} endpoints are reachable for {action}.",
            fg="red",
        ),
        err=True,
    )
    for name, url in endpoints.items():
        click.echo(f"  - {name}: {url}", err=True)
    if using_daytona:
        click.echo(
            click.style(
                "Daytona endpoints were requested, but the sandbox is not reachable. "
                "Check the sandbox state or restart it with: simlab env up <env-name> --daytona",
                fg="yellow",
            ),
            err=True,
        )
    else:
        click.echo(
            click.style(
                "Local tool servers are not reachable. "
                "Start the environment with simlab env up <env-name> and inspect "
                "docker compose ps / docker compose logs if a service failed to start.",
                fg="yellow",
            ),
            err=True,
        )
        state_file = _daytona_state_file_for_config(config_path)
        if state_file is not None and state_file.exists():
            click.echo(
                click.style(
                    "A Daytona state file exists for this environment at "
                    f"{state_file}. If you intended to use the remote sandbox, "
                    "rerun with --daytona.",
                    fg="yellow",
                ),
                err=True,
            )
    raise SystemExit(1)


def _resolve_endpoints(
    *,
    config_path: str,
    config: EnvConfig,
    daytona_requested: bool,
    daytona_api_key: str | None = None,
    allow_resume: bool = True,
) -> tuple[dict[str, str], bool]:
    """Resolve endpoints for the explicitly requested execution backend."""
    if daytona_requested:
        return (
            _get_daytona_endpoints(
                config_path, daytona_api_key=daytona_api_key, allow_resume=allow_resume
            ),
            True,
        )

    return get_tool_endpoints(config, config_path=Path(config_path)), False


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
        data = yaml.safe_load(path.read_text()) or {}
        config = EnvConfig(**data)
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
        rows = _local_task_rows(_load_tasks(bundle_dir), include_test=include_test)
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
    data = yaml.safe_load(Path(config_path).read_text()) or {}
    config = EnvConfig(**data)
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
    resolved_model = (model or global_cfg.agent_model or "").strip()
    resolved_provider = (provider or global_cfg.agent_provider or "openai").strip() or "openai"
    resolved_api_key = resolve_agent_api_key(
        api_key,
        provider=resolved_provider,
        config=global_cfg,
    )
    resolved_base_url = (base_url or global_cfg.agent_base_url or "").strip() or None
    return resolved_model, resolved_provider, resolved_api_key, resolved_base_url


def _find_rubric_file(bundle_dir: Path | None, task_id: str) -> Path | None:
    """Locate rubric markdown for a task in the bundle's rubrics/ directory."""
    if bundle_dir is None:
        return None
    rubric_path = bundle_dir / "rubrics" / f"{task_id}.md"
    if rubric_path.is_file():
        return rubric_path
    # Try with hyphens replaced by underscores and vice versa
    alt_id = task_id.replace("-", "_")
    alt_path = bundle_dir / "rubrics" / f"{alt_id}.md"
    if alt_path.is_file():
        return alt_path
    alt_id = task_id.replace("_", "-")
    alt_path = bundle_dir / "rubrics" / f"{alt_id}.md"
    if alt_path.is_file():
        return alt_path
    return None


def _maybe_run_rubric_judge(
    *,
    task_data: dict[str, Any],
    bundle_dir: Path | None,
    messages: list[dict[str, Any]],
    global_cfg: Any,
) -> dict[str, Any] | None:
    """Run the rubric-based LLM judge if a rubric is available and config allows it."""
    meta = task_data.get("meta", {})
    task_id = meta.get("task_id", "")
    task_description = task_data.get("task", "")

    rubric_path = _find_rubric_file(bundle_dir, task_id)
    if rubric_path is None:
        return None

    model = (global_cfg.verifier_model or "").strip()
    if not model:
        click.echo("  Rubric found but no verifier_model configured — skipping rubric judge.")
        return None

    provider = (global_cfg.verifier_provider or "").strip() or None
    api_key = (global_cfg.verifier_api_key or "").strip() or None
    base_url_val = (global_cfg.verifier_base_url or "").strip() or None

    rubric_markdown = rubric_path.read_text(encoding="utf-8")
    click.echo(click.style("\nRunning rubric judge...", bold=True))
    click.echo(f"  Rubric: {rubric_path}")
    click.echo(f"  Model:  {provider}/{model}" if provider else f"  Model:  {model}")

    run_rubric_judge = get_rubric_judge_runner()
    result = run_rubric_judge(
        task_description=task_description,
        rubric_markdown=rubric_markdown,
        messages=messages,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url_val,
    )

    if result.error:
        click.echo(click.style(f"  Rubric judge error: {result.error}", fg="yellow"))
    else:
        color = "green" if result.score >= 0.6 else "red"
        click.echo(
            click.style(
                f"  Rubric verdict: {result.verdict} (score={result.score:.2f}, "
                f"confidence={result.confidence:.2f})",
                fg=color,
            )
        )
        if result.failed_criteria:
            click.echo(f"  Failed criteria: {', '.join(result.failed_criteria)}")

    return result.to_dict()


def _apply_verifier_env_overrides(global_cfg: Any) -> tuple[dict[str, str | None], dict[str, str]]:
    env_updates = {
        "SIMLAB_VERIFIER_MODEL": (global_cfg.verifier_model or "").strip(),
        "SIMLAB_VERIFIER_PROVIDER": (global_cfg.verifier_provider or "").strip(),
        "SIMLAB_VERIFIER_BASE_URL": (global_cfg.verifier_base_url or "").strip(),
        "SIMLAB_VERIFIER_API_KEY": (global_cfg.verifier_api_key or "").strip(),
    }
    original: dict[str, str | None] = {}
    applied = {key: value for key, value in env_updates.items() if value}
    for key, value in applied.items():
        original[key] = os.environ.get(key)
        os.environ[key] = value
    return original, applied


def _restore_env(original: dict[str, str | None], applied: dict[str, str]) -> None:
    for key in applied:
        previous = original.get(key)
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


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
    data = yaml.safe_load(Path(config_path).read_text()) or {}
    config = EnvConfig(**data)
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
    config_path = str(env_dir / "env.yaml")
    config_file = Path(config_path)
    data = yaml.safe_load(config_file.read_text()) or {}
    config = EnvConfig(**data)
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


def _print_parallel_summary(summary: Any) -> None:
    """Display a table summarizing parallel rollout results."""
    click.echo(click.style("\n=== Parallel Rollout Summary ===\n", bold=True))
    click.echo(f"  Task:      {summary.task_id}")
    click.echo(f"  Rollouts:  {summary.rollout_count}")
    click.echo(f"  Duration:  {summary.total_duration_seconds:.1f}s")
    click.echo()

    passed = [r for r in summary.results if r.error is None and r.verification_passed is not False]
    failed = [r for r in summary.results if r.error is not None or r.verification_passed is False]
    rewards = [r.reward for r in summary.results if r.reward is not None]

    for r in summary.results:
        status = (
            click.style("PASS", fg="green")
            if r.verification_passed
            else (
                click.style("FAIL", fg="red")
                if r.verification_passed is False
                else click.style("ERR ", fg="red")
                if r.error
                else click.style("N/A ", fg="yellow")
            )
        )
        reward_str = f"{r.reward:.1f}" if r.reward is not None else "-"
        err_str = f"  error={r.error}" if r.error else ""
        click.echo(
            f"  rollout {r.rollout_idx}: {status}  "
            f"reward={reward_str}  steps={r.steps_taken}  "
            f"time={r.duration_seconds:.0f}s{err_str}"
        )

    click.echo()
    click.echo(f"  Passed:    {len(passed)}/{summary.rollout_count}")
    if failed:
        click.echo(click.style(f"  Failed:    {len(failed)}", fg="red"))
    if rewards:
        avg = sum(rewards) / len(rewards)
        click.echo(f"  Avg reward: {avg:.2f}")
    if summary.output_dir:
        click.echo(f"  Output:    {summary.output_dir}")
    click.echo()


@tasks.command()
@click.option(
    "--env",
    "env_name",
    required=True,
    help="Environment name (for config and endpoints).",
)
@click.option("--task", "task_id", required=True, help="Task ID to run.")
@click.option("--include-test", is_flag=True, help="Include test-category tasks.")
@click.option(
    "--tasks-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Path to a local task bundle (default: fetch task from API using env template).",
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
    env_name: str,
    task_id: str,
    include_test: bool,
    tasks_dir: str | None,
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
    env_dir = resolve_env_dir(env_name, ctx=ctx)
    config_path = str(env_dir / "env.yaml")
    config_file = Path(config_path)
    data = yaml.safe_load(config_file.read_text()) or {}
    config = EnvConfig(**data)
    base_url_api = resolve_scenario_manager_api_url(
        config_path=config_file,
        config=config,
        base_url=resolve_scenario_manager_api_url(config=global_cfg),
    )
    scenario_manager_api_key = resolve_collinear_api_key(config=global_cfg)
    backend_id: str | None = None
    bundle_dir: Path | None = Path(tasks_dir) if tasks_dir else None
    if bundle_dir is not None:
        task_data, profiles, _local_task_file = _load_local_task(bundle_dir, task_id)
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

        from simlab.runtime.env_lifecycle import _validate_daytona_coding_assets
        from simlab.runtime.parallel_daytona import ParallelDaytonaOrchestrator

        _validate_daytona_coding_assets(config, env_dir)

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

        registry = ToolRegistry()
        registry.load_all()
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
        preseed_svc_names = get_profiled_service_names(config, profile="preseed")
        seed_svc_names = get_profiled_service_names(config, profile="seed")

        orchestrator = ParallelDaytonaOrchestrator(
            rollout_count=rollout_count,
            max_parallel=max_parallel,
            daytona_api_key=global_cfg.daytona_api_key,
        )
        summary = orchestrator.execute(
            task_id=task_data.get("meta", {}).get("task_id", task_id),
            task_data=task_data,
            profiles=profiles,
            compose_dir=env_dir,
            tool_ports=tool_ports,
            extra_tool_urls=extra_tool_urls,
            preseed_svc_names=preseed_svc_names,
            seed_svc_names=seed_svc_names,
            config=config,
            config_path=config_path,
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            max_steps=max_steps,
            agent_import_path=agent_import_path,
            agent_timeout_seconds=agent_timeout_seconds,
            no_seed=no_seed,
            bundle_dir=bundle_dir,
            global_cfg=global_cfg,
            backend_id=backend_id,
            base_url_api=base_url_api,
            scenario_manager_api_key=scenario_manager_api_key,
        )
        _print_parallel_summary(summary)
        has_failures = any(
            r.error is not None or r.verification_passed is False for r in summary.results
        )
        if has_failures:
            raise SystemExit(1)
        return

    meta = task_data.get("meta", {})
    display = meta.get("display_name", meta.get("task_id", task_id))

    # --- Phase 0: Auto-start environment if needed ---
    managed_env = False  # True only when *this* run started the env
    has_local_services = env_has_local_services(env_dir)

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
                click.echo(click.style("\nStarting environment...", bold=True))
                if daytona:
                    ensure_env_started_daytona(
                        env_dir,
                        config,
                        env_dir / "env.yaml",
                        daytona_api_key=global_cfg.daytona_api_key,
                        verbose=verbose,
                    )
                    if not no_seed:
                        run_env_seed_daytona(
                            env_dir,
                            config,
                            env_dir / "env.yaml",
                            daytona_api_key=global_cfg.daytona_api_key,
                        )
                else:
                    ensure_env_started_local(
                        env_dir,
                        config,
                        env_dir / "env.yaml",
                    )
                    if not no_seed:
                        run_env_seed_local(env_dir, config, env_dir / "env.yaml")
                click.echo(click.style("Environment started.", fg="green"))

        # --- Phases 1-3: Seed, run agent, verify ---
        _run_single_rollout(
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
        )
    finally:
        # --- Phase 4: Teardown ---
        if managed_env and not keep_alive:
            click.echo(click.style("\nTearing down environment...", bold=True))
            try:
                if daytona:
                    env_down_daytona(env_dir, daytona_api_key=global_cfg.daytona_api_key)
                else:
                    env_down_local(env_dir)
            except (Exception, SystemExit):
                click.echo(
                    click.style("Warning: environment teardown failed.", fg="yellow"),
                    err=True,
                )


def _run_single_rollout(
    *,
    env_dir: Path,
    config: EnvConfig,
    config_path: str,
    global_cfg: Any,
    task_data: dict[str, Any],
    task_id: str,
    profiles: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    display: str,
    model: str,
    provider: str,
    api_key: str | None,
    base_url: str | None,
    max_steps: int,
    agent_import_path: str | None,
    agent_timeout_seconds: float,
    no_seed: bool,
    verbose: bool,
    daytona: bool,
    tasks_dir: str | None,
    bundle_dir: Path | None,
    backend_id: str | None,
    base_url_api: str,
    scenario_manager_api_key: str | None,
    managed_env: bool,
    keep_alive: bool,
    skip_env_setup: bool,
) -> None:
    """Execute a single task rollout: resolve endpoints, seed, run agent, verify."""
    # --- Phase 1: Resolve endpoints ---
    endpoints, using_daytona = _resolve_endpoints(
        config_path=config_path,
        config=config,
        daytona_requested=daytona,
        daytona_api_key=global_cfg.daytona_api_key,
        allow_resume=not skip_env_setup,
    )
    # --- Load MCP servers (if any) and build MCP client handles ---
    mcp_config = load_mcp_servers_from_env_dir(env_dir)
    mcp_clients = _build_mcp_clients(mcp_config, endpoints)
    needed_run_endpoints = needed_task_endpoints(
        task_data,
        endpoints,
        include_tool_servers=True,
    )
    if needed_run_endpoints:
        _require_reachable_endpoints(
            endpoints=needed_run_endpoints,
            action="task run",
            using_daytona=using_daytona,
            config_path=config_path,
            wait=using_daytona,
        )
    _require_mcp_tools_available(mcp_clients)

    # --- Phase 2: Seeding ---
    if not no_seed:
        # Env-level mutable-state reset — skip if we just did it during startup
        if not managed_env:
            run_env_seed_services(
                config,
                config_path,
                using_daytona=using_daytona,
                daytona_api_key=global_cfg.daytona_api_key,
            )

        # Task-level provisioning
        _provision_task_group_channels(
            task_data,
            profiles,
            config,
            config_path,
            using_daytona=using_daytona,
            daytona_api_key=global_cfg.daytona_api_key,
        )
        _provision_task_calendar_users(
            task_data,
            config,
            config_path,
            using_daytona=using_daytona,
            daytona_api_key=global_cfg.daytona_api_key,
        )
        _ensure_task_calendar_accounts(task_data, profiles, endpoints, config)

        # Task data seeding
        emails = task_data.get("seed_emails", [])
        cal_events = task_data.get("seed_calendar_events", [])
        group_channels = task_data.get("seed_group_channels", [])
        if emails or cal_events or group_channels:
            click.echo(click.style(f"\nSeeding task: {display}\n", bold=True))
            ok, fail = _seed_task_data(task_data, profiles, endpoints)
            if fail:
                click.echo(
                    click.style(f"\n  Seeding had {fail} failure(s).", fg="yellow"),
                )
            else:
                click.echo(click.style(f"\n  {ok} item(s) seeded.\n", fg="green"))

    # --- Phase 3: Build environment + run ---
    rewritten = _rewrite_tool_server_urls(task_data, endpoints)
    tool_namespace_endpoints = _effective_tool_servers(rewritten, endpoints)
    if not tool_namespace_endpoints and not mcp_clients:
        click.echo(
            click.style(
                "Task has no valid tool_servers and no MCP servers. "
                "Add tools to the env or use --mcp-servers at env init.",
                fg="red",
            )
        )
        raise SystemExit(1)
    if not agent_import_path and not api_key:
        click.echo(
            click.style(
                "Reference agent requires --agent-api-key or SIMLAB_AGENT_API_KEY (or OPENAI_API_KEY for OpenAI). "
                "Custom agents (--agent-import-path) use their own credentials.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    click.echo(click.style(f"\nRunning task: {display}", bold=True))
    click.echo(f"  Model:            {model} ({provider})")
    click.echo(f"  Max steps:        {max_steps}")
    click.echo(f"  Agent import:     {agent_import_path or 'builtin:ReferenceAgent'}")
    click.echo(f"  Agent timeout (s): {agent_timeout_seconds:g}")
    click.echo(f"  Endpoint mode:    {'daytona' if using_daytona else 'local'}")
    click.echo(f"  Managed env:      {managed_env}")
    if verbose:
        click.echo(f"  Tool servers:     {', '.join(sorted(tool_namespace_endpoints))}")
    click.echo()

    unified_tool_env_cls, run_with_agent_contract = get_agent_runtime_helpers()
    environment = unified_tool_env_cls(
        tool_servers=tool_namespace_endpoints,
        mcp_clients=mcp_clients or None,
    )
    instruction = rewritten.get("task", "")
    skills_section = _build_skills_guidance_section(
        _load_skills_markdown(config=config, bundle_dir=bundle_dir)
    )
    if skills_section:
        instruction = f"{instruction}\n\n{skills_section}"
    services_section = _build_services_available_section(
        config,
        daytona=using_daytona,
        config_path=config_path,
        endpoints=endpoints,
    )
    if services_section:
        instruction = f"{instruction}\n\n{services_section}"

    artifacts = run_with_agent_contract(
        task_id=meta.get("task_id", task_id),
        instruction=instruction,
        model=model,
        provider=provider,
        max_steps=max_steps,
        environment=environment,
        agent_import_path=agent_import_path,
        timeout_seconds=agent_timeout_seconds,
        api_key=api_key,
        base_url=base_url,
    )
    artifacts.metadata.setdefault("cli_runtime", {})
    artifacts.metadata["cli_runtime"].update(
        {
            "base_url": base_url,
            "api_key_provided": bool(api_key),
            "agent_import_path": agent_import_path,
            "tool_servers": tool_namespace_endpoints,
            "managed_env": managed_env,
            "keep_alive": keep_alive,
        }
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"agent_run_{meta.get('task_id', task_id)}_{ts}"
    run_dir = Path("output") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "artifacts.json"
    artifacts.dump(output_path)

    click.echo(f"Saved run artifacts: {output_path}")
    run_error = artifacts.error
    if run_error:
        emit_cli_event(
            "tasks_run_failed",
            {
                "failure_stage": "agent_run",
                "task_source": "local_bundle" if tasks_dir else "scenario_manager_api",
                "mode": "daytona" if using_daytona else "local",
                "seed_requested": not no_seed,
                "tool_server_count": len(tool_namespace_endpoints),
                "had_custom_agent": bool(agent_import_path),
                "max_steps": max_steps,
                "managed_env": managed_env,
                "keep_alive": keep_alive,
            },
        )
        click.echo(click.style(f"Run failed: {run_error}", fg="yellow"), err=True)

    evaluators = task_data.get("verifiers") or task_data.get("evaluators")
    reward: float | None = None
    verification_passed: bool | None = None
    if evaluators:
        build_verifier_artifacts, infer_scenario_from_evaluator, run_verifier = (
            get_verifier_runtime_helpers()
        )

        verifier_scenario = infer_scenario_from_evaluator(evaluators[0]) or backend_id or ""
        # Verifier adapter expects a task file path: write task_data to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(task_data, f, indent=2)
            task_file = Path(f.name)
        try:
            adapter = build_verifier_artifacts(
                artifacts,
                task_file,
                _verifier_tool_servers(rewritten, endpoints, mcp_clients),
            )
            verifier_results: list[dict[str, Any]] = []
            click.echo(click.style("\nRunning verifiers...", bold=True))
            original_env, applied_env = _apply_verifier_env_overrides(global_cfg)
            try:
                for i, ev in enumerate(evaluators, 1):
                    if ev.get("func") != "python_module" or not ev.get("module"):
                        continue
                    mod_path = ev["module"]
                    click.echo(f"  Verifier {i}/{len(evaluators)}: {mod_path}")
                    result = run_verifier(
                        mod_path,
                        adapter,
                        verifier_scenario,
                        scenario_manager_base_url=base_url_api,
                        scenario_manager_api_key=scenario_manager_api_key,
                        local_verifier_path=(
                            _get_local_verifier_file_path(bundle_dir, mod_path)
                            if bundle_dir is not None
                            else None
                        ),
                        verifier_cache_root=env_dir / "verifiers",
                    )
                    verifier_results.append(
                        {
                            "module": mod_path,
                            "success": result.success,
                            "message": result.message or "",
                            "output": result.output or "",
                        }
                    )
                    if result.output or result.message:
                        click.echo(result.output or result.message)
                    if not result.success:
                        click.echo(click.style(f"  Verifier {i} failed", fg="red"))
                    else:
                        click.echo(click.style(f"  Verifier {i} passed", fg="green"))
            finally:
                _restore_env(original_env, applied_env)
            all_passed = all(r["success"] for r in verifier_results)

            # --- Rubric judge (optional) ---
            rubric_result_dict = _maybe_run_rubric_judge(
                task_data=task_data,
                bundle_dir=bundle_dir,
                messages=list(artifacts.messages),
                global_cfg=global_cfg,
            )

            # --- Compute reward ---
            reward = 1.0 if all_passed else 0.0
            verification_passed = all_passed

            # Write Harbor-style verifier reward files under this run's directory
            verifier_dir = run_dir / "verifier"
            verifier_dir.mkdir(parents=True, exist_ok=True)
            (verifier_dir / "reward.txt").write_text(
                "1" if all_passed else "0",
                encoding="utf-8",
            )
            reward_payload: dict[str, Any] = {
                "reward": reward,
                "verifier_results": verifier_results,
            }
            if rubric_result_dict is not None:
                reward_payload["rubric_result"] = rubric_result_dict
            (verifier_dir / "reward.json").write_text(
                json.dumps(reward_payload, indent=2),
                encoding="utf-8",
            )
            click.echo(f"Reward: {reward:.1f}")
            if rubric_result_dict:
                click.echo(f"Rubric score: {rubric_result_dict.get('score', 'N/A')}")
            click.echo(f"Verifier reward: {verifier_dir / 'reward.txt'}")
            if not all_passed:
                emit_cli_event(
                    "tasks_run_failed",
                    {
                        "failure_stage": "verifier",
                        "task_source": "local_bundle" if tasks_dir else "scenario_manager_api",
                        "mode": "daytona" if using_daytona else "local",
                        "seed_requested": not no_seed,
                        "tool_server_count": len(tool_namespace_endpoints),
                        "had_custom_agent": bool(agent_import_path),
                        "max_steps": max_steps,
                        "verifier_count": len(evaluators),
                        "managed_env": managed_env,
                        "keep_alive": keep_alive,
                    },
                )
                click.echo(click.style("\nVerification failed.", fg="red"), err=True)
                raise SystemExit(1)
            click.echo(click.style("\nVerification passed.", fg="green"))
        finally:
            if task_file is not None:
                task_file.unlink(missing_ok=True)
    elif run_error:
        raise SystemExit(1)

    click.echo(click.style("Run completed.", fg="green"))
    emit_cli_event(
        "tasks_run_completed",
        {
            "task_source": "local_bundle" if tasks_dir else "scenario_manager_api",
            "mode": "daytona" if using_daytona else "local",
            "seed_requested": not no_seed,
            "tool_server_count": len(tool_namespace_endpoints),
            "had_custom_agent": bool(agent_import_path),
            "max_steps": max_steps,
            "steps_taken": artifacts.steps_taken,
            "tool_calls_made": len(artifacts.tool_calls),
            "verifier_count": len(evaluators or []),
            "verifier_passed": verification_passed,
            "reward": reward,
            "managed_env": managed_env,
            "keep_alive": keep_alive,
        },
    )


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
