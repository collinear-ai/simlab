"""CLI telemetry helpers for event capture and API request context."""

from __future__ import annotations

import atexit
import json
import platform
import queue
import sys
import threading
import time
import uuid
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from functools import wraps
from pathlib import Path
from typing import Any
from typing import cast

import click
import requests

from simlab import __version__
from simlab.config import SIMLAB_TELEMETRY_DISABLE_ENV_VARS
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_scenario_manager_api_url
from simlab.config import telemetry_disabled
from simlab.config import telemetry_state_path

SIMLAB_COMMAND_HEADER = "X-SimLab-Command"
SIMLAB_INSTALL_HEADER = "X-SimLab-Install-Id"
SIMLAB_SESSION_HEADER = "X-SimLab-Session-Id"
SIMLAB_VERSION_HEADER = "X-SimLab-Version"

SIMLAB_TELEMETRY_API_PATH = "/telemetry/cli-events"
SIMLAB_TELEMETRY_TIMEOUT_SECONDS = 2.0
SIMLAB_TELEMETRY_EXIT_FLUSH_SECONDS = 5.0
SIMLAB_SESSION_TIMEOUT_SECONDS = 30 * 60
SIMLAB_TELEMETRY_META_KEY = "simlab.telemetry"
SIMLAB_TELEMETRY_NOTICE = (
    "SimLab sends usage telemetry and includes your Collinear API key when configured. "
    f"Set {SIMLAB_TELEMETRY_DISABLE_ENV_VARS[0]}=1 to disable."
)

request_headers_var: ContextVar[dict[str, str] | None] = ContextVar(
    "simlab_request_headers",
    default=None,
)
capture_config_var: ContextVar[TelemetryCaptureConfig | None] = ContextVar(
    "simlab_capture_config",
    default=None,
)


@dataclass(frozen=True)
class TelemetryCaptureConfig:
    """Describes whether one command should emit telemetry."""

    enabled: bool
    api_key: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class QueuedTelemetryRequest:
    """Describes one telemetry POST that should be sent in the background."""

    url: str
    payload: dict[str, Any]
    headers: dict[str, str]
    timeout_seconds: float


class BackgroundRequestPoster:
    """Posts telemetry requests on a daemon thread so commands stay non-blocking."""

    def __init__(
        self,
        *,
        thread_name: str,
        max_queue_size: int = 64,
        worker_count: int = 3,
    ) -> None:
        """Initialize the background poster and its bounded in-memory queue."""
        self.thread_name = thread_name
        self.worker_count = worker_count
        self.queue: queue.Queue[QueuedTelemetryRequest] = queue.Queue(maxsize=max_queue_size)
        self.lock = threading.Lock()
        self.threads: list[threading.Thread] = []

    def submit(self, request: QueuedTelemetryRequest) -> None:
        """Queue one telemetry request for background delivery."""
        self.ensure_started()
        try:
            self.queue.put_nowait(request)
        except queue.Full:
            return

    def ensure_started(self) -> None:
        """Start the daemon worker threads once, on first use."""
        active_threads = [thread for thread in self.threads if thread.is_alive()]
        if len(active_threads) >= self.worker_count:
            self.threads = active_threads
            return
        with self.lock:
            active_threads = [thread for thread in self.threads if thread.is_alive()]
            if len(active_threads) >= self.worker_count:
                self.threads = active_threads
                return
            while len(active_threads) < self.worker_count:
                worker_index = len(active_threads) + 1
                thread = threading.Thread(
                    target=self.run_forever,
                    name=f"{self.thread_name}-{worker_index}",
                    daemon=True,
                )
                thread.start()
                active_threads.append(thread)
            self.threads = active_threads

    def run_forever(self) -> None:
        """Deliver queued telemetry requests until process exit."""
        while True:
            request = self.queue.get()
            try:
                requests.post(
                    request.url,
                    json=request.payload,
                    headers=request.headers,
                    timeout=request.timeout_seconds,
                )
            except requests.RequestException:
                pass
            finally:
                self.queue.task_done()

    def wait_until_idle(self, timeout_seconds: float) -> bool:
        """Wait for queued requests to finish, up to the timeout."""
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() < deadline:
            if self.queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self.queue.unfinished_tasks == 0


background_telemetry_poster = BackgroundRequestPoster(thread_name="simlab-telemetry")


def queue_telemetry_request(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> None:
    """Queue one telemetry POST to run in the background."""
    background_telemetry_poster.submit(
        QueuedTelemetryRequest(
            url=url,
            payload=dict(payload),
            headers=dict(headers),
            timeout_seconds=timeout_seconds,
        )
    )


def flush_background_telemetry_requests(
    timeout_seconds: float = SIMLAB_TELEMETRY_EXIT_FLUSH_SECONDS,
) -> bool:
    """Wait briefly for queued telemetry requests before process exit."""
    return background_telemetry_poster.wait_until_idle(timeout_seconds)


atexit.register(flush_background_telemetry_requests)


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def parse_timestamp(value: object) -> datetime | None:
    """Parse an ISO timestamp if present."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_state(path: Path) -> dict[str, Any]:
    """Read persisted telemetry state from disk."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(path: Path, state: dict[str, Any]) -> bool:
    """Persist telemetry state to disk."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        return False
    return True


def ensure_session(state: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Fill missing session fields and rotate idle sessions."""
    now_value = now or utc_now()
    notice_shown = bool(state.get("notice_shown", False))
    install_id = str(state.get("install_id") or state.get("anonymous_id") or "").strip()
    if not install_id:
        install_id = str(uuid.uuid4())
    session_id = str(state.get("session_id", "")).strip()
    session_started_at = parse_timestamp(state.get("session_started_at"))
    last_seen = parse_timestamp(state.get("last_seen_at"))
    session_expired = (
        last_seen is None
        or (now_value - last_seen).total_seconds() > SIMLAB_SESSION_TIMEOUT_SECONDS
    )
    if not session_id or session_expired:
        session_id = str(uuid.uuid4())
        session_started_at = now_value

    state.clear()
    state["notice_shown"] = notice_shown
    state["install_id"] = install_id
    state["session_id"] = session_id
    state["session_started_at"] = (session_started_at or now_value).isoformat()
    state["last_seen_at"] = now_value.isoformat()
    return state


def root_context(ctx: click.Context | None) -> click.Context | None:
    """Walk to the root Click context."""
    current = ctx
    while current is not None and current.parent is not None:
        current = current.parent
    return current


def command_name_from_ctx(ctx: click.Context | None) -> str:
    """Return the leaf command path without the root program name."""
    if ctx is None:
        return ""
    path = ctx.command_path.strip()
    root = root_context(ctx)
    if root is None or not root.command_path.strip():
        return path
    root_path = root.command_path.strip()
    if root.command.name == "cli" and path.startswith(f"{root_path} "):
        return path[len(root_path) + 1 :]
    return path


def command_group_from_name(command_name: str) -> str:
    """Return the top-level command group."""
    return command_name.split(" ", 1)[0] if command_name else ""


def leaf_command_from_name(command_name: str) -> str:
    """Return the final command segment."""
    return command_name.rsplit(" ", 1)[-1] if command_name else ""


def current_request_headers() -> dict[str, str]:
    """Return per-command headers for Scenario Manager API requests."""
    return dict(request_headers_var.get() or {})


def build_scenario_manager_headers() -> dict[str, str]:
    """Return the current telemetry headers for Scenario Manager API requests."""
    headers = current_request_headers()
    headers.setdefault(SIMLAB_VERSION_HEADER, __version__)
    return headers


def normalize_optional_string(value: str | None) -> str | None:
    """Return a stripped string or None."""
    normalized = (value or "").strip()
    return normalized or None


def normalize_config_path(value: object) -> str | Path | None:
    """Return a config path value only when it is already path-like."""
    if isinstance(value, (str, Path)):
        return value
    return None


def stderr_supports_notice() -> bool:
    """Return whether stderr can show an interactive first-run notice."""
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def always_enabled_capture_config() -> TelemetryCaptureConfig:
    """Return the default capture config for commands that should always emit."""
    if telemetry_disabled():
        return disabled_capture_config()
    return TelemetryCaptureConfig(enabled=True)


def disabled_capture_config() -> TelemetryCaptureConfig:
    """Return a capture config that suppresses telemetry."""
    return TelemetryCaptureConfig(enabled=False)


def current_capture_config() -> TelemetryCaptureConfig | None:
    """Return the active command capture config."""
    return capture_config_var.get()


@contextmanager
def capture_config_context(config: TelemetryCaptureConfig | None) -> Iterator[None]:
    """Set the active command capture config for nested emit_cli_event calls."""
    token = capture_config_var.set(config)
    try:
        yield
    finally:
        capture_config_var.reset(token)


def resolve_scenario_manager_capture_config(
    ctx: click.Context | None,
    *,
    config_path: str | Path | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> TelemetryCaptureConfig:
    """Enable telemetry for authenticated commands against the configured API."""
    root = root_context(ctx)
    if telemetry_disabled(ctx=root):
        return disabled_capture_config()

    resolved_config_path: Path | None = None
    if isinstance(config_path, Path):
        resolved_config_path = config_path
    elif isinstance(config_path, str) and config_path.strip():
        resolved_config_path = Path(config_path)

    resolved_url = resolve_scenario_manager_api_url(
        config_path=resolved_config_path,
        base_url=base_url,
        ctx=root,
    ).rstrip("/")
    resolved_api_key = resolve_collinear_api_key(api_key, ctx=root)
    return TelemetryCaptureConfig(
        enabled=bool(resolved_api_key),
        api_key=resolved_api_key,
        base_url=resolved_url,
    )


class TelemetryService:
    """Thin client that posts CLI events back to the Scenario Manager API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        state_path: Path | None = None,
        timeout_seconds: float = SIMLAB_TELEMETRY_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the CLI telemetry client."""
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip() or None
        self.state_path = state_path or telemetry_state_path()
        self.timeout_seconds = timeout_seconds
        self.version = __version__
        self.state = ensure_session(load_state(self.state_path))
        self.show_notice_once()
        save_state(self.state_path, self.state)

    def show_notice_once(self) -> None:
        """Show the first-run telemetry notice on interactive terminals."""
        if bool(self.state.get("notice_shown")):
            return
        if not stderr_supports_notice():
            return
        click.secho(SIMLAB_TELEMETRY_NOTICE, fg="yellow", err=True)
        self.state["notice_shown"] = True

    def touch(self) -> None:
        """Advance the session heartbeat."""
        ensure_session(self.state)
        save_state(self.state_path, self.state)

    def command_properties(self, command_name: str) -> dict[str, Any]:
        """Return common event properties shared by all commands."""
        self.touch()
        return {
            "command": command_name,
            "command_group": command_group_from_name(command_name),
            "command_name": leaf_command_from_name(command_name),
            "install_id": self.install_id,
            "session_id": self.session_id,
            "simlab_version": self.version,
            "python_version": platform.python_version(),
            "platform_system": platform.system().lower(),
            "platform_release": platform.release(),
        }

    def capture(
        self,
        event: str,
        properties: dict[str, Any] | None = None,
        *,
        api_key_override: str | None = None,
        base_url_override: str | None = None,
    ) -> None:
        """Send a telemetry event to the Scenario Manager API."""
        self.touch()
        payload = {
            "event": event,
            "properties": properties or {},
        }
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        effective_api_key = normalize_optional_string(api_key_override) or self.api_key
        effective_base_url = normalize_optional_string(base_url_override) or self.base_url
        if effective_api_key:
            headers["API-Key"] = effective_api_key
        queue_telemetry_request(
            url=f"{effective_base_url}{SIMLAB_TELEMETRY_API_PATH}",
            payload=payload,
            headers=headers,
            timeout_seconds=self.timeout_seconds,
        )

    @property
    def session_id(self) -> str:
        """Return the current CLI session identifier."""
        return str(self.state.get("session_id", ""))

    @property
    def install_id(self) -> str:
        """Return the current CLI install identifier."""
        return str(self.state.get("install_id", ""))

    @contextmanager
    def command_context(self, command_name: str) -> Iterator[None]:
        """Set per-command headers for outbound Scenario Manager API requests."""
        self.touch()
        headers = {
            SIMLAB_COMMAND_HEADER: command_name,
            SIMLAB_INSTALL_HEADER: self.install_id,
            SIMLAB_SESSION_HEADER: self.session_id,
            SIMLAB_VERSION_HEADER: self.version,
        }
        token = request_headers_var.set(headers)
        try:
            yield
        finally:
            request_headers_var.reset(token)


def build_telemetry_service(ctx: click.Context | None) -> TelemetryService | None:
    """Create a telemetry service from the root CLI config."""
    root = root_context(ctx)
    if telemetry_disabled(ctx=root):
        return None

    base_url = resolve_scenario_manager_api_url(ctx=root)
    return TelemetryService(
        base_url=base_url,
        api_key=resolve_collinear_api_key(ctx=root),
        state_path=telemetry_state_path(ctx=root),
    )


def get_telemetry_from_ctx(ctx: click.Context | None) -> TelemetryService | None:
    """Return the cached telemetry service for the root command context."""
    root = root_context(ctx)
    if root is None:
        return None
    if SIMLAB_TELEMETRY_META_KEY not in root.meta:
        root.meta[SIMLAB_TELEMETRY_META_KEY] = build_telemetry_service(root)
    return cast(TelemetryService | None, root.meta.get(SIMLAB_TELEMETRY_META_KEY))


def capture_cli_event(
    ctx: click.Context | None,
    event: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Capture a command-scoped telemetry event."""
    capture_config = current_capture_config()
    if capture_config is None or not capture_config.enabled:
        return
    telemetry = get_telemetry_from_ctx(ctx)
    if telemetry is None:
        return
    command_name = command_name_from_ctx(ctx)
    event_properties = telemetry.command_properties(command_name)
    if properties:
        event_properties.update(properties)
    telemetry.capture(
        event,
        event_properties,
        api_key_override=capture_config.api_key,
        base_url_override=capture_config.base_url,
    )


def emit_cli_event(event: str, properties: dict[str, Any] | None = None) -> None:
    """Capture a telemetry event for the current Click command."""
    capture_cli_event(click.get_current_context(silent=True), event, properties)


def exit_code_from_exception(exc: BaseException) -> int:
    """Convert Click and Python exit exceptions into exit codes."""
    if isinstance(exc, click.exceptions.Exit):
        return int(exc.exit_code)
    if isinstance(exc, SystemExit):
        code = exc.code
        if isinstance(code, int):
            return code
        return 1
    if isinstance(exc, (KeyboardInterrupt, click.Abort)):
        return 130
    return 1


def error_type_from_exception(exc: BaseException) -> str:
    """Return a compact error category for telemetry."""
    if isinstance(exc, click.exceptions.Exit):
        return "click_exit"
    if isinstance(exc, SystemExit):
        return "system_exit"
    if isinstance(exc, KeyboardInterrupt):
        return "keyboard_interrupt"
    if isinstance(exc, click.Abort):
        return "click_abort"
    return type(exc).__name__


def with_command_telemetry(
    command_name: str,
    resolver: Callable[
        [click.Context | None, tuple[object, ...], dict[str, object]],
        TelemetryCaptureConfig,
    ]
    | None = None,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Wrap one Click command with generic telemetry capture."""

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            ctx = click.get_current_context(silent=True)
            capture_config = (
                resolver(ctx, args, kwargs)
                if resolver is not None
                else always_enabled_capture_config()
            )
            with capture_config_context(capture_config):
                if not capture_config.enabled:
                    return func(*args, **kwargs)

                telemetry = get_telemetry_from_ctx(ctx)
                if telemetry is None:
                    return func(*args, **kwargs)

                started_at = time.monotonic()
                base_properties = telemetry.command_properties(command_name)

                with telemetry.command_context(command_name):
                    telemetry.capture(
                        "cli_command_started",
                        base_properties,
                        api_key_override=capture_config.api_key,
                        base_url_override=capture_config.base_url,
                    )
                    try:
                        result = func(*args, **kwargs)
                    except BaseException as exc:
                        duration_ms = int((time.monotonic() - started_at) * 1000)
                        exit_code = exit_code_from_exception(exc)
                        failed_properties = dict(base_properties)
                        failed_properties.update(
                            {
                                "duration_ms": duration_ms,
                                "exit_code": exit_code,
                                "error_type": error_type_from_exception(exc),
                            }
                        )
                        if exit_code == 0:
                            telemetry.capture(
                                "cli_command_finished",
                                failed_properties,
                                api_key_override=capture_config.api_key,
                                base_url_override=capture_config.base_url,
                            )
                        else:
                            failed_properties["error_category"] = error_type_from_exception(exc)
                            telemetry.capture(
                                "cli_command_failed",
                                failed_properties,
                                api_key_override=capture_config.api_key,
                                base_url_override=capture_config.base_url,
                            )
                        raise

                duration_ms = int((time.monotonic() - started_at) * 1000)
                completed_properties = dict(base_properties)
                completed_properties["duration_ms"] = duration_ms
                completed_properties["exit_code"] = 0
                telemetry.capture(
                    "cli_command_finished",
                    completed_properties,
                    api_key_override=capture_config.api_key,
                    base_url_override=capture_config.base_url,
                )
                return result

        return wrapper

    return decorator
