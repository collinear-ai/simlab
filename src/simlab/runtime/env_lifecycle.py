"""Environment lifecycle: detection, startup, seeding, health polling, teardown."""

from __future__ import annotations

import itertools
import json
import queue
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from datetime import timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import click
import yaml

from simlab.cli.progress import StepContext
from simlab.cli.progress import StepProgress
from simlab.cli.progress import StepProgressReporter
from simlab.composer.engine import ComposeEngine
from simlab.composer.engine import EnvConfig
from simlab.config import resolve_daytona_api_key
from simlab.env_registry import build_registry
from simlab.runtime.compose_ps import parse_ps_output
from simlab.runtime.daytona_runner import DaytonaNotFoundError
from simlab.seeder import query_tool_server

# ---------------------------------------------------------------------------
# Compose introspection
# ---------------------------------------------------------------------------


def _compose_has_build_contexts(compose_file: Path) -> bool:
    """Return True if the compose file defines any service with a build context."""
    existing = yaml.safe_load(compose_file.read_text())
    if not existing or "services" not in existing:
        return False
    return any("build" in svc for svc in existing["services"].values())


def _compose_has_services(compose_file: Path) -> bool:
    """Return True if the compose file defines at least one service."""
    try:
        existing = yaml.safe_load(compose_file.read_text())
    except (yaml.YAMLError, OSError):
        return False
    services = existing.get("services") if isinstance(existing, dict) else None
    return isinstance(services, dict) and bool(services)


# ---------------------------------------------------------------------------
# Profiled service helpers
# ---------------------------------------------------------------------------


def get_profiled_service_names(
    config: EnvConfig,
    profile: str,
    config_path: Path | None = None,
    tool_names: list[str] | None = None,
) -> list[str]:
    """Return compose service names associated with a seed/preseed profile."""
    return _get_profiled_service_names(
        config,
        profile=profile,
        config_path=config_path,
        tool_names=tool_names,
    )


def run_profiled_services_local(
    out_dir: Path,
    svc_names: list[str],
    profile: str,
    env_overrides: dict[str, str] | None = None,
    quiet: bool = False,
    step_ctx: object | None = None,
) -> None:
    """Run profiled containers locally via docker compose."""
    _run_profiled_services_local(
        out_dir,
        svc_names,
        profile,
        env_overrides=env_overrides,
        quiet=quiet,
        step_ctx=step_ctx,
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
    registry = build_registry(env_dir=config_path.parent if config_path is not None else None)
    names: list[str] = []
    for tool_name in tool_names or config.tools:
        tool = registry.get_tool(tool_name)
        if not tool:
            continue
        service_defs = tool.preseed_services if profile == "preseed" else tool.seed_services
        if service_defs:
            names.extend(service_defs.keys())
    return names


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
        tty_args = ["-T"] if quiet else []
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
                *tty_args,
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


# ---------------------------------------------------------------------------
# Tool ports / Daytona runner
# ---------------------------------------------------------------------------


def _get_tool_ports(config: EnvConfig, config_path: Path | None = None) -> dict[str, int]:
    """Get tool server ports keyed by tool name from loaded catalog entries."""
    registry = build_registry(env_dir=config_path.parent if config_path is not None else None)
    ports: dict[str, int] = {}
    for tool_name in config.tools:
        tool = registry.get_tool(tool_name)
        if tool and tool.tool_server_port is not None:
            ports[tool_name] = tool.tool_server_port
    return ports


def _get_daytona_runner(daytona_api_key: str | None = None):  # noqa: ANN202
    """Import Daytona runner lazily so base CLI works without optional deps."""
    try:
        daytona_runner_module = import_module("simlab.runtime.daytona_runner")
    except ModuleNotFoundError as exc:
        click.echo(
            click.style(
                "Daytona support is unavailable in this installation.\n"
                "Source checkout: uv sync --extra daytona\n"
                'Installed CLI: uv tool install --python 3.13 "simulationlab[daytona]"\n'
                "Or run without --daytona.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1) from exc

    return daytona_runner_module.DaytonaRunner(daytona_api_key=daytona_api_key)


def validate_daytona_coding_assets(config: EnvConfig, config_dir: Path) -> None:
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


# ---------------------------------------------------------------------------
# Seed verification
# ---------------------------------------------------------------------------


def _query_tool_server(
    url: str,
    tool_name: str,
    parameters: dict[str, Any],
) -> Any:
    """Execute a tool action against a tool server."""
    return query_tool_server(url, tool_name, parameters)


def _verify_seed_local(config: EnvConfig, config_path: Path | None = None) -> None:
    """Verify seed data by querying the frappe tool server."""
    registry = build_registry(env_dir=config_path.parent if config_path is not None else None)
    for tool_name in config.tools:
        tool = registry.get_tool(tool_name)
        if not tool or not tool.seed_services:
            continue
        if tool.name == "frappe-hrms":
            url = f"http://localhost:{tool.tool_server_port}"
            _print_frappe_verification(url)


def _verify_seed_daytona(config: EnvConfig, endpoints: dict[str, str], config_path: Path) -> None:
    """Verify seed data by querying Daytona-exposed tool endpoints."""
    registry = build_registry(env_dir=config_path.parent if config_path is not None else None)
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

    emp_resp = _query_tool_server(
        url,
        "frappe_list_resource",
        {"doctype": "Employee", "limit_page_length": 0},
    )
    emp_count = _extract_count(emp_resp)

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

    total_docs = sum(len(v) for v in categories.values())
    if total_docs > 0:
        click.echo(f"\n  Documents:           {total_docs} total")
        for cat_name in [*_DOC_PREFIXES.keys(), "Policies"]:
            items = categories.get(cat_name, [])
            if items:
                click.echo(f"    {cat_name + ':':<23}{len(items)}")

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
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BUILDKIT_STEP_RE = re.compile(r"^#\d+\s+\[(?P<target>[^\]]+)\]\s*(?P<rest>.*)$")
_COMPOSE_ACTIONS = {
    "pulling",
    "pulled",
    "building",
    "creating",
    "created",
    "starting",
    "started",
    "recreated",
    "running",
    "waiting",
}


def _compose_service_names(compose_file: Path) -> set[str]:
    """Return compose service names declared in *compose_file*."""
    try:
        compose_data = yaml.safe_load(compose_file.read_text())
    except (yaml.YAMLError, OSError):
        return set()
    services = compose_data.get("services") if isinstance(compose_data, dict) else None
    if not isinstance(services, dict):
        return set()
    return {str(name) for name in services}


def _local_health_fetcher(compose_file: Path) -> Callable[[], dict[str, str]]:
    """Return a callable that fetches health from local docker compose."""
    expected_services = _compose_service_names(compose_file)

    def _fetch() -> dict[str, str]:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "ps",
                "--all",
                "--format",
                "{{.Name}}\t{{.Status}}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {}
        return parse_ps_output(result.stdout, expected_services=expected_services)

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


def _render_simple_status_line(
    spinner_frame: str,
    message: str,
    elapsed: float,
) -> str:
    """Render a single spinner line for long-running subprocess work."""
    elapsed_str = f"{int(elapsed)}s"
    return "".join(
        [
            click.style(f" {spinner_frame} ", fg="cyan"),
            click.style(message, fg="white"),
            click.style(f"  {elapsed_str}", fg="bright_black"),
        ]
    )


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
    if not sys.stdout.isatty():
        return
    for _ in range(n):
        sys.stdout.write("\033[A\033[2K")
    sys.stdout.flush()


def _poll_health(
    health_fetcher: Callable[[], dict[str, str]],
    timeout: int = 180,
    *,
    quiet: bool = False,
) -> None:
    """Poll services with an animated status display.

    When *quiet* is True, suppress the animated spinner and service table —
    use this when a parent progress context already owns the terminal.
    Failures and timeouts are always reported.

    Also suppresses animated output when stdout is not a tty (pipes, CI).
    """
    interactive = sys.stdout.isatty()
    suppress_animation = quiet or not interactive
    start = time.time()
    spinner = itertools.cycle(_SPINNER_FRAMES)
    lines_printed = 0
    poll_interval = 2.0
    consecutive_ready_polls = 0

    while time.time() - start < timeout:
        elapsed = time.time() - start
        services = health_fetcher()
        current_message = _health_wait_message(services)

        if not suppress_animation:
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
                if not suppress_animation and lines_printed > 0:
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
                consecutive_ready_polls += 1
            else:
                consecutive_ready_polls = 0

            if consecutive_ready_polls >= 2:
                if not suppress_animation and lines_printed > 0:
                    _clear_lines(lines_printed)
                if not suppress_animation:
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


def _run_compose_up_local(
    up_cmd: list[str],
    *,
    has_builds: bool,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``docker compose up`` with an animated progress display.

    When *quiet* is True, the built-in spinner is suppressed — use this
    when a parent progress context (e.g. ``StepProgress``) already owns
    the terminal display.
    """
    start = time.time()
    spinner = itertools.cycle(_SPINNER_FRAMES)
    current_message = "Starting docker compose build..." if has_builds else "Starting services..."
    poll_interval = 0.2
    lines_printed = 0
    interactive = sys.stdout.isatty()

    process = subprocess.Popen(
        up_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    progress_lines: queue.Queue[str] = queue.Queue()
    reader_threads = [
        threading.Thread(
            target=_drain_subprocess_stream,
            args=(process.stdout, stdout_chunks, progress_lines),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_subprocess_stream,
            args=(process.stderr, stderr_chunks, progress_lines),
            daemon=True,
        ),
    ]
    for reader_thread in reader_threads:
        reader_thread.start()

    if not interactive:
        click.echo(current_message)
        click.echo()

    while process.poll() is None:
        now = time.time()
        progress_message = _latest_compose_progress_message(progress_lines)
        if progress_message is not None:
            current_message = progress_message

        if not quiet and interactive:
            if lines_printed > 0:
                _clear_lines(lines_printed)

            click.echo(_render_simple_status_line(next(spinner), current_message, now - start))
            click.echo()
            lines_printed = 2
        time.sleep(poll_interval)

    process.wait()
    for reader_thread in reader_threads:
        reader_thread.join()
    if not quiet and lines_printed > 0:
        _clear_lines(lines_printed)

    return subprocess.CompletedProcess(
        args=up_cmd,
        returncode=process.returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


def _drain_subprocess_stream(
    stream: Any,
    sink: list[str],
    progress_lines: queue.Queue[str] | None = None,
) -> None:
    """Drain a subprocess stream in the background to avoid pipe backpressure."""
    if stream is None:
        return
    remainder = ""
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            sink.append(chunk)
            if progress_lines is None:
                continue
            remainder += chunk
            lines = remainder.splitlines(keepends=True)
            remainder = lines.pop() if lines and not lines[-1].endswith(("\n", "\r")) else ""
            for line in lines:
                progress_lines.put(line)
        if progress_lines is not None and remainder.strip():
            progress_lines.put(remainder)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def _latest_compose_progress_message(progress_lines: queue.Queue[str]) -> str | None:
    """Return the most recent user-facing compose progress message, if any."""
    latest: str | None = None
    while True:
        try:
            raw_line = progress_lines.get_nowait()
        except queue.Empty:
            return latest
        message = _summarize_compose_progress(raw_line)
        if message is not None:
            latest = message


def _summarize_compose_progress(raw_line: str) -> str | None:
    """Convert raw compose output into a concise progress message."""
    text = _normalize_compose_output_line(raw_line)
    if not text:
        return None

    if text.lower().startswith("attaching to "):
        return "Attaching to container logs..."

    action_parts = text.rsplit(" ", maxsplit=1)
    if len(action_parts) == 2 and action_parts[1].lower() in _COMPOSE_ACTIONS:
        service, action = action_parts
        return f"{action.capitalize()} {service}..."

    buildkit_match = _BUILDKIT_STEP_RE.match(text)
    if buildkit_match:
        target = buildkit_match.group("target")
        rest = buildkit_match.group("rest").strip()
        target_parts = target.split()
        service = target_parts[0]
        step = target_parts[1] if len(target_parts) > 1 and "/" in target_parts[1] else None
        if rest.endswith(" DONE"):
            rest = rest.removesuffix(" DONE").strip()
        if not rest:
            return f"Building {service}..."
        prefix = f"Building {service}"
        if step:
            prefix += f" ({step})"
        return f"{prefix}: {rest}"

    if text.startswith("=> "):
        return text[3:]

    interesting_terms = (
        "pull",
        "build",
        "create",
        "start",
        "extract",
        "export",
        "resolve",
        "load",
        "download",
    )
    if any(term in text.lower() for term in interesting_terms):
        return text
    return None


def _normalize_compose_output_line(raw_line: str) -> str:
    """Strip ANSI escapes and collapse whitespace in compose output."""
    without_ansi = _ANSI_ESCAPE_RE.sub("", raw_line).replace("\r", " ").strip()
    return " ".join(without_ansi.split())


def _health_wait_message(services: dict[str, str]) -> str:
    """Return a deterministic health-poll message from current service state."""
    if not services:
        return "Waiting for docker compose status..."

    starting = sorted(name for name, status in services.items() if status == "starting")
    unknown = sorted(name for name, status in services.items() if status == "unknown")

    if starting:
        if len(starting) == 1:
            return f"Waiting for {starting[0]} health check..."
        return f"Waiting for {len(starting)} services to become healthy..."

    if unknown:
        if len(unknown) == 1:
            return f"Reading status for {unknown[0]}..."
        return f"Reading status for {len(unknown)} services..."

    ready = sum(1 for status in services.values() if status in ("healthy", "running"))
    if ready == len(services):
        return "Confirming services are stable..."

    return "Waiting for services to start..."


# ---------------------------------------------------------------------------
# Public lifecycle functions
# ---------------------------------------------------------------------------


def env_has_local_services(env_dir: Path) -> bool:
    """Return True when the compose file in *env_dir* defines at least one service.

    External-only environments (all tools are URL-based) return False.
    Callers should skip startup / teardown when this returns False.
    """
    compose_file = env_dir / "docker-compose.yml"
    if not compose_file.exists():
        return False
    return _compose_has_services(compose_file)


def is_env_running_local(env_dir: Path) -> bool:
    """Return True when ALL non-profile local Docker Compose services are running.

    Returns False if any expected service is missing or not in a running state,
    so that ``ensure_env_started_local`` can bring up the missing services.
    """
    compose_file = env_dir / "docker-compose.yml"
    if not compose_file.exists():
        return False
    try:
        compose_data = yaml.safe_load(compose_file.read_text())
    except (yaml.YAMLError, OSError):
        return False
    services = compose_data.get("services") if isinstance(compose_data, dict) else None
    if not isinstance(services, dict) or not services:
        return False
    expected = sum(1 for svc in services.values() if not svc.get("profiles"))
    if expected == 0:
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "ps", "--status", "running", "-q"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    running = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
    return running >= expected


def has_any_running_containers(env_dir: Path) -> bool:
    """Return True when ANY Docker Compose service is running.

    Unlike ``is_env_running_local`` (which requires *all* expected services),
    this returns True if even a single container is up.  Used by ``env delete``
    where any running container should block deletion.
    """
    compose_file = env_dir / "docker-compose.yml"
    if not compose_file.exists():
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "ps", "--status", "running", "-q"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def is_env_running_daytona(env_dir: Path, daytona_api_key: str | None = None) -> bool:
    """Return True when a Daytona sandbox for this env is active.

    Pure status check — no side effects.  Reads ``daytona-state.json``,
    fetches the sandbox via the SDK, and returns whether it is in an
    active state.
    """
    state_file = env_dir / "daytona-state.json"
    if not state_file.exists():
        return False
    try:
        state = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    sandbox_id = state.get("sandbox_id")
    if not sandbox_id:
        return False
    api_key = resolve_daytona_api_key(daytona_api_key)
    if not api_key:
        return False
    try:
        daytona_mod = import_module("daytona")
        client = daytona_mod.Daytona(daytona_mod.DaytonaConfig(api_key=api_key))
        sandbox = client.get(sandbox_id)
    except Exception:
        return False
    return _daytona_sandbox_status(sandbox) in {"running", "started", "active", "ready"}


def detect_env_status(env_dir: Path) -> str:
    """Return ``'running'``, ``'stopped'``, or ``'error'`` for an environment."""
    env_yaml = env_dir / "env.yaml"
    if not env_yaml.is_file():
        return "error"
    try:
        yaml.safe_load(env_yaml.read_text(encoding="utf-8"))
    except Exception:
        return "error"
    if has_any_running_containers(env_dir):
        return "running"
    return "stopped"


def get_env_created_date(env_dir: Path) -> str:
    """Return the creation date of ``env.yaml`` as ``YYYY-MM-DD``."""
    env_yaml = env_dir / "env.yaml"
    try:
        mtime = env_yaml.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    except OSError:
        return "unknown"


def ensure_daytona_sandbox_ready(env_dir: Path, daytona_api_key: str | None = None) -> bool:
    """Make sure the Daytona sandbox is usable, resuming or cleaning up as needed.

    Returns True if an existing sandbox is now active (already running or
    successfully resumed).  Returns False if no sandbox exists or recovery
    failed.

    On False, the state file is removed so the caller can create a fresh
    sandbox when we can prove the referenced sandbox no longer exists (or the
    state file is invalid).  If Daytona API calls fail in a way that leaves
    sandbox existence uncertain, the state file is preserved so callers do not
    lose the cleanup pointer.
    """
    state_file = env_dir / "daytona-state.json"
    if not state_file.exists():
        return False
    try:
        state = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        state_file.unlink(missing_ok=True)
        return False
    sandbox_id = state.get("sandbox_id")
    if not sandbox_id:
        state_file.unlink(missing_ok=True)
        return False
    api_key = resolve_daytona_api_key(daytona_api_key)
    if not api_key:
        return False

    try:
        daytona_mod = import_module("daytona")
        client = daytona_mod.Daytona(daytona_mod.DaytonaConfig(api_key=api_key))
    except Exception:
        return False

    try:
        sandbox = client.get(sandbox_id)
    except DaytonaNotFoundError:
        state_file.unlink(missing_ok=True)
        return False
    except Exception:
        return False

    status = _daytona_sandbox_status(sandbox)
    if status in {"running", "started", "active", "ready"}:
        return True

    if status in {"paused", "stopped"}:
        click.echo(f"  Daytona sandbox is '{status}', attempting resume...")
        if _try_resume_daytona_sandbox(client, sandbox, sandbox_id):
            return True
        click.echo(
            click.style("  Could not resume sandbox; will create a new one.", fg="yellow"),
            err=True,
        )

    # Remove state file only if deletion succeeds; preserve on failure.
    try:
        client.delete(sandbox)
        state_file.unlink(missing_ok=True)
    except Exception:
        click.echo(
            click.style(
                "  Warning: could not delete stale sandbox; keeping state file for manual cleanup.",
                fg="yellow",
            ),
            err=True,
        )
    return False


def _daytona_sandbox_status(sandbox: Any) -> str:
    """Best-effort status string from a Daytona sandbox object."""
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


def _try_resume_daytona_sandbox(client: Any, sandbox: Any, sandbox_id: str) -> bool:
    """Attempt to resume a paused/stopped sandbox, waiting up to 60 s."""
    resumed = False
    for obj, method_name, args in [
        (sandbox, "resume", ()),
        (sandbox, "start", ()),
        (client, "resume", (sandbox_id,)),
        (client, "start", (sandbox_id,)),
    ]:
        method = getattr(obj, method_name, None)
        if method is None:
            continue
        try:
            method(*args)
            resumed = True
            break
        except (TypeError, Exception):  # noqa: S112
            continue
    if not resumed:
        return False
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            refreshed = client.get(sandbox_id)  # type: ignore[union-attr]
        except Exception:
            return False
        if _daytona_sandbox_status(refreshed) in {"running", "started", "active", "ready"}:
            click.echo(click.style("  Sandbox resumed.", fg="green"))
            return True
        time.sleep(2)
    return False


def ensure_env_started_local(
    env_dir: Path,
    config: EnvConfig,
    config_path: Path | None = None,
    *,
    quiet: bool = False,
) -> None:
    """Start local Docker services: preseed, compose up, health poll.

    **Precondition:** ``env_has_local_services(env_dir)`` is True.  This
    function does NOT check for services itself — the caller must guard.

    Does NOT run seed-profile services; call ``run_env_seed_local`` separately.

    When *quiet* is True, suppress the built-in compose-up spinner and
    health-poll display — use this when a parent progress context (e.g.
    ``StepProgress``) already owns the terminal.
    """
    compose_file = env_dir / "docker-compose.yml"
    has_builds = _compose_has_build_contexts(compose_file)

    preseed_svc_names = _get_preseed_service_names(config, config_path)
    if preseed_svc_names:
        if not quiet:
            click.echo("  Running preseed services...")
        _run_profiled_services_local(env_dir, preseed_svc_names, profile="preseed", quiet=True)

    up_cmd = ["docker", "compose", "-f", str(compose_file), "up", "-d"]
    if has_builds:
        up_cmd.append("--build")
    result = _run_compose_up_local(up_cmd, has_builds=has_builds, quiet=quiet)
    if result.returncode != 0:
        port_match = None
        stderr = result.stderr or ""
        if stderr:
            port_conflict_patterns = (
                r"Bind for 0\.0\.0\.0:(?P<port>\d+) failed: port is already allocated",
                r"0\.0\.0\.0:(?P<port>\d+)/tcp: .*address already in use",
                r"listen tcp\d* 0\.0\.0\.0:(?P<port>\d+): bind: address already in use",
            )
            for pattern in port_conflict_patterns:
                port_match = re.search(pattern, stderr)
                if port_match:
                    break
        if port_match:
            port = port_match.group("port")
            click.echo(
                click.style(
                    f"Failed to start services because port {port} is already in use.",
                    fg="red",
                ),
                err=True,
            )
            try:
                ps = subprocess.run(
                    ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception:
                ps = None

            offenders: list[str] = []
            if ps is not None and ps.returncode == 0:
                for line in ps.stdout.splitlines():
                    if f":{port}->" not in line and f"::{port}->" not in line:
                        continue
                    offenders.append(line.split("\t", maxsplit=1)[0].strip())

            if offenders:
                click.echo(
                    "Stop the container using this port and rerun. Containers using this port:",
                    err=True,
                )
                for name in offenders:
                    click.echo(f"  - {name}", err=True)
                click.echo("Tip: docker stop <container>", err=True)
            else:
                click.echo(
                    (
                        f"Stop whatever is using port {port} and rerun. "
                        f"Tip: ss -ltnp | rg ':{port}\\b'"
                    ),
                    err=True,
                )
            raise SystemExit(1)

        click.echo(
            click.style("Failed to start services:", fg="red"),
            err=True,
        )
        if result.stderr:
            click.echo(result.stderr.strip(), err=True)
        raise SystemExit(1)

    if not quiet:
        click.echo()
    _poll_health(
        _local_health_fetcher(compose_file),
        timeout=180,
        quiet=quiet,
    )


def ensure_env_started_daytona(
    env_dir: Path,
    config: EnvConfig,
    config_path: Path | None = None,
    *,
    daytona_api_key: str | None = None,
    verbose: bool = False,
    progress: StepProgress | None = None,
) -> dict[str, str]:
    """Create a Daytona sandbox, upload compose, start services, health poll.

    **Precondition:** ``env_has_local_services(env_dir)`` is True.

    Does NOT run seed-profile services; call ``run_env_seed_daytona`` separately.

    Returns the endpoints dict ``{tool_name: public_url}``.
    """
    validate_daytona_coding_assets(config, env_dir)

    runner = _get_daytona_runner(daytona_api_key=daytona_api_key)
    tool_ports = _get_tool_ports(config, config_path)
    preseed_svc_names = _get_preseed_service_names(config, config_path)
    progress_instance = progress or StepProgress(verbose=verbose)
    reporter = StepProgressReporter(progress_instance)

    endpoints = runner.up(
        env_dir, tool_ports, preseed_svc_names=preseed_svc_names, reporter=reporter
    )

    click.echo()
    _poll_health(
        lambda: runner.get_health(env_dir),
        timeout=180,
    )

    return endpoints


def _seed_local(out_dir: Path, seed_svc_names: list[str]) -> None:
    """Run seed containers locally via docker compose."""
    _run_profiled_services_local(out_dir, seed_svc_names, profile="seed")


def run_env_seed_local(
    env_dir: Path,
    config: EnvConfig,
    config_path: Path | None = None,
    *,
    quiet: bool = False,
) -> None:
    """Run seed-profile services locally and verify."""
    seed_svc_names = _get_seed_service_names(config, config_path)
    if not seed_svc_names:
        return
    if not quiet:
        click.echo("  Running environment seed services...")
    _run_profiled_services_local(env_dir, seed_svc_names, profile="seed", quiet=True)
    _verify_seed_local(config, config_path)


def run_env_seed_daytona(
    env_dir: Path,
    config: EnvConfig,
    config_path: Path | None = None,
    *,
    daytona_api_key: str | None = None,
    endpoints: dict[str, str] | None = None,
) -> None:
    """Run seed-profile services in the Daytona sandbox and verify."""
    seed_svc_names = _get_seed_service_names(config, config_path)
    if not seed_svc_names:
        return
    click.echo("  Running seed services...")
    runner = _get_daytona_runner(daytona_api_key=daytona_api_key)
    runner.seed(env_dir, seed_svc_names)
    if endpoints:
        _verify_seed_daytona(config, endpoints, config_path or env_dir / "env.yaml")


def env_down_local(env_dir: Path) -> None:
    """Stop and remove local Docker Compose services.

    Raises ``SystemExit(1)`` when the compose file is missing or
    ``docker compose down`` fails, so callers can distinguish success
    from failure.
    """
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

    click.echo(click.style("Services stopped.", fg="green"))


def env_purge_docker_local(env_dir: Path) -> bool:
    """Remove Docker resources (containers, networks, volumes) for this environment.

    Runs ``docker compose down -v --remove-orphans`` to ensure a clean slate.
    Returns True if cleanup succeeded (or no compose file exists), False if
    Docker reported errors.
    """
    compose_file = env_dir / "docker-compose.yml"
    if not compose_file.exists():
        return True

    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "down",
            "-v",
            "--remove-orphans",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(
            click.style("Docker cleanup failed:", fg="red"),
            err=True,
        )
        click.echo(result.stderr, err=True)
        return False
    return True


def env_down_daytona(
    env_dir: Path,
    daytona_api_key: str | None = None,
    progress: StepProgress | None = None,
) -> None:
    """Delete the Daytona sandbox for this environment."""
    runner = _get_daytona_runner(daytona_api_key=daytona_api_key)
    reporter = StepProgressReporter(progress) if progress is not None else None
    runner.down(env_dir, reporter=reporter)
