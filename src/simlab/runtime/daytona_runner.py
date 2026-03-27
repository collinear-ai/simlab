"""Run docker-compose environments inside a Daytona sandbox."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import shlex
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import click
import yaml

from simlab.cli.progress import DefaultReporter
from simlab.config import resolve_daytona_api_key
from simlab.runtime.compose_ps import parse_ps_output

_daytona_not_found_error: type[Exception]

try:
    from daytona import CreateSandboxFromSnapshotParams
    from daytona import CreateSnapshotParams
    from daytona import Daytona
    from daytona import DaytonaConfig
    from daytona import DaytonaNotFoundError as DaytonaSdkNotFoundError
    from daytona import Resources
    from daytona import SessionExecuteRequest
except ImportError as e:
    DAYTONA_IMPORT_ERROR: ImportError | None = e
    CreateSandboxFromSnapshotParams = Any  # type: ignore[misc,assignment]
    CreateSnapshotParams = Any  # type: ignore[misc,assignment]
    Daytona = Any  # type: ignore[misc,assignment]
    DaytonaConfig = Any  # type: ignore[misc,assignment]

    class DaytonaFallbackNotFoundError(Exception):
        """Fallback type used when the Daytona SDK is not installed."""

    Resources = Any  # type: ignore[misc,assignment]
    SessionExecuteRequest = Any  # type: ignore[misc,assignment]
    _daytona_not_found_error = DaytonaFallbackNotFoundError
else:
    DAYTONA_IMPORT_ERROR = None
    _daytona_not_found_error = DaytonaSdkNotFoundError

DaytonaNotFoundError = _daytona_not_found_error

_STATE_FILE = "daytona-state.json"
_SNAPSHOT_NAME = "docker-dind"
_COMPOSE_DIR = "/home/daytona"
_MCP_GATEWAY_CONFIG_FILE = "mcp-gateway-config.json"

_logger = logging.getLogger(__name__)


def _get_daytona(api_key: str | None = None):  # noqa: ANN202
    """Build an authenticated Daytona client, or exit with a helpful message."""
    if DAYTONA_IMPORT_ERROR is not None:
        click.echo(
            click.style(
                "Daytona support requires optional dependency 'daytona'. "
                "Install with: pip install 'simulationlab[daytona]'",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1) from DAYTONA_IMPORT_ERROR

    key = resolve_daytona_api_key(api_key)
    if not key:
        click.echo(
            click.style(
                "Daytona API key is required via --daytona-api-key, config, "
                "SIMLAB_DAYTONA_API_KEY, or DAYTONA_API_KEY.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)
    return Daytona(DaytonaConfig(api_key=key))


# ---------------------------------------------------------------------------
# Module-level helpers (reusable by DaytonaRunner and parallel orchestrator)
# ---------------------------------------------------------------------------


def _prepare_compose_for_remote(
    compose_dir: Path,
) -> tuple[bytes, dict[str, tuple[Path, str]]]:
    """Strip local build contexts and collect contexts to build remotely."""
    compose_path = compose_dir / "docker-compose.yml"
    compose_data = yaml.safe_load(compose_path.read_text())
    services = compose_data.get("services", {}) if isinstance(compose_data, dict) else {}

    build_contexts: dict[str, tuple[Path, str]] = {}
    compose_root = compose_dir.resolve()
    for service_name, service in services.items():
        if not isinstance(service, dict):
            continue

        volumes = service.get("volumes")
        if isinstance(volumes, list):
            rewritten_volumes: list[Any] = []
            for volume in volumes:
                if not isinstance(volume, str):
                    rewritten_volumes.append(volume)
                    continue
                source, sep, remainder = volume.partition(":")
                if not sep or not source.startswith("/"):
                    rewritten_volumes.append(volume)
                    continue
                try:
                    relative_source = Path(source).resolve().relative_to(compose_root)
                except ValueError:
                    rewritten_volumes.append(volume)
                    continue
                remote_source = f"{_COMPOSE_DIR}/{relative_source.as_posix()}"
                rewritten_volumes.append(f"{remote_source}:{remainder}")
            service["volumes"] = rewritten_volumes

        build = service.pop("build", None)
        if build is None:
            continue
        build_context = build.get("context") if isinstance(build, dict) else build
        if not isinstance(build_context, str):
            continue

        context_path = Path(build_context)
        if not context_path.is_absolute():
            context_path = (compose_dir / context_path).resolve()
        image_tag = str(service.get("image", f"{service_name}:latest"))
        service["image"] = image_tag
        if context_path.is_dir():
            build_contexts[service_name] = (context_path, image_tag)
        else:
            click.echo(
                click.style(
                    f"Warning: build context not found for {service_name}: {context_path}",
                    fg="yellow",
                ),
                err=True,
            )

    compose_bytes = yaml.dump(
        compose_data,
        default_flow_style=False,
        sort_keys=False,
    ).encode()
    return compose_bytes, build_contexts


def _upload_directory(sandbox: Any, local_dir: Path, remote_dir: str) -> None:  # noqa: ANN401
    """Upload a local directory recursively into the sandbox."""
    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(local_dir)
        remote_path = f"{remote_dir}/{rel_path.as_posix()}"
        remote_parent = remote_path.rsplit("/", 1)[0]
        sandbox.process.exec(f"mkdir -p {remote_parent}", cwd=_COMPOSE_DIR)
        sandbox.fs.upload_file(file_path.read_bytes(), remote_path)


def _upload_compose_bundle(sandbox: Any, compose_dir: Path) -> None:  # noqa: ANN401
    """Upload bundle-local assets as a single tarball to minimise API calls.

    Packs all files and directories in *compose_dir* (except
    ``docker-compose.yml``, ``.env``, and the Daytona state file) into an
    in-memory gzipped tarball, uploads it in **one** HTTP request, and
    extracts it in the sandbox.  This avoids the connection-reset errors
    that occur when many small files are uploaded individually through
    Daytona's proxy.
    """
    ignored = {"docker-compose.yml", ".env", _STATE_FILE}
    entries = [e for e in compose_dir.iterdir() if e.name not in ignored]
    if not entries:
        return

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", dereference=True) as tar:
        for entry in entries:
            tar.add(str(entry), arcname=entry.name)
    tarball_bytes = buf.getvalue()

    remote_tar = f"{_COMPOSE_DIR}/_bundle.tar.gz"
    sandbox.fs.upload_file(tarball_bytes, remote_tar)
    resp = sandbox.process.exec(
        "tar xzf _bundle.tar.gz && rm -f _bundle.tar.gz",
        cwd=_COMPOSE_DIR,
        timeout=60,
    )
    if resp.exit_code != 0:
        raise RuntimeError(f"Failed to extract bundle tarball: {resp.result}")


def _run_profiled_services_in_sandbox(
    sandbox: Any,  # noqa: ANN401
    svc_names: list[str],
    *,
    profile: str,
    env_overrides: dict[str, str] | None = None,
    log_prefix: str = "",
    quiet: bool = False,
    reporter: Any | None = None,  # noqa: ANN401
) -> str:
    """Run ``docker compose run`` for services in a compose profile.

    When *quiet* is True, output is routed through *reporter* (if provided)
    instead of being printed directly via ``click.echo``.  The *log_prefix*
    is prepended to messages when not in quiet mode.
    """
    output_parts: list[str] = []
    phase_label = "Preseeding" if profile == "preseed" else "Seeding"
    prefix = f"{log_prefix} " if log_prefix else ""

    def _echo(msg: str) -> None:
        if quiet and reporter is not None:
            reporter.detail(msg)
        elif not quiet:
            click.echo(f"{prefix}{msg}")

    for svc_name in svc_names:
        _echo(f"{phase_label}: {svc_name}...")
        override_args = ""
        for key, value in (env_overrides or {}).items():
            override_args += f" -e {shlex.quote(f'{key}={value}')}"
        compose_cmd = (
            f"docker compose --profile {profile} run --rm{override_args} {shlex.quote(svc_name)}"
        )
        resp = sandbox.process.exec(
            compose_cmd,
            cwd=_COMPOSE_DIR,
            timeout=600,
        )
        if resp.exit_code != 0:
            raise RuntimeError(f"{phase_label[:-3]} '{svc_name}' failed:\n{resp.result}")
        output_parts.append(resp.result or "")
        if env_overrides:
            rendered = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in env_overrides.items()
            )
            _echo(f"  Applied overrides: {rendered}")

    if not quiet:
        click.echo(click.style(f"{prefix}{phase_label} complete.", fg="green"))
    return "\n".join(output_parts)


def _ensure_docker_daemon_ready_in_sandbox(
    sandbox: Any,  # noqa: ANN401
    *,
    log_prefix: str = "",
) -> None:
    """Wait for the sandbox Docker daemon and start it manually if needed."""
    prefix = f"{log_prefix} " if log_prefix else ""

    with contextlib.suppress(Exception):
        sandbox.process.create_session("init")

    click.echo(f"{prefix}Waiting for Docker daemon...")
    daemon_ready = False
    resp = None
    for attempt in range(20):
        resp = sandbox.process.execute_session_command(
            "init",
            SessionExecuteRequest(command="docker info"),
            timeout=5,
        )
        if resp.exit_code == 0:
            click.echo(f"{prefix}Docker daemon ready.")
            daemon_ready = True
            break
        if attempt == 0:
            click.echo(f"{prefix}  Docker daemon still starting; retrying...")
        time.sleep(3)

    if not daemon_ready:
        click.echo(f"{prefix}Attempting to start Docker daemon manually...")
        sandbox.process.execute_session_command(
            "init",
            SessionExecuteRequest(command="nohup dockerd &", run_async=True),
        )
        for _ in range(20):
            resp = sandbox.process.execute_session_command(
                "init",
                SessionExecuteRequest(command="docker info"),
                timeout=5,
            )
            if resp.exit_code == 0:
                click.echo(f"{prefix}Docker daemon ready (manual start).")
                daemon_ready = True
                break
            time.sleep(3)

    if daemon_ready:
        return

    stdout = getattr(resp, "stdout", "") if resp else ""
    stderr = getattr(resp, "stderr", "") if resp else ""
    raise RuntimeError(f"Docker daemon failed to start in sandbox. stdout={stdout} stderr={stderr}")


def restart_sandbox_environment(
    sandbox: Any,  # noqa: ANN401
    *,
    preseed_svc_names: list[str] | None = None,
    log_prefix: str = "",
) -> None:
    """Restart compose services inside an already existing sandbox."""
    prefix = f"{log_prefix} " if log_prefix else ""

    _ensure_docker_daemon_ready_in_sandbox(sandbox, log_prefix=log_prefix)

    if preseed_svc_names:
        _run_profiled_services_in_sandbox(
            sandbox,
            preseed_svc_names,
            profile="preseed",
            log_prefix=log_prefix,
        )

    click.echo(f"{prefix}Restarting services...")
    resp = sandbox.process.exec(
        "docker compose up -d",
        cwd=_COMPOSE_DIR,
        timeout=120,
    )
    if resp.exit_code != 0:
        raise RuntimeError(f"docker compose up failed:\n{resp.result}")


def setup_sandbox_environment(
    sandbox: Any,  # noqa: ANN401
    compose_dir: Path,
    tool_ports: dict[str, int],
    preseed_svc_names: list[str] | None = None,
    log_prefix: str = "",
) -> dict[str, str]:
    """Set up a Daytona sandbox: Docker daemon, compose files, images, services.

    This is the core setup logic extracted from ``DaytonaRunner.up()`` so it can
    be reused by both the single-environment flow and the parallel orchestrator.

    Args:
        sandbox: An already-created Daytona sandbox object.
        compose_dir: Local directory containing docker-compose.yml and .env.
        tool_ports: Mapping of tool name to port number.
        preseed_svc_names: Services to run with the ``preseed`` profile before startup.
        log_prefix: Optional prefix for log messages (e.g. ``"[rollout 1/5]"``).

    Returns:
        Mapping of tool name to public URL.

    Raises:
        RuntimeError: If any setup step fails.

    """
    prefix = f"{log_prefix} " if log_prefix else ""

    _ensure_docker_daemon_ready_in_sandbox(sandbox, log_prefix=log_prefix)

    # Upload compose files.  Bundle assets are packed into a single tarball
    # to minimise HTTP requests through Daytona's proxy (many small uploads
    # cause connection-reset errors under concurrent load).
    click.echo(f"{prefix}Uploading docker-compose.yml...")
    compose_content, build_contexts = _prepare_compose_for_remote(compose_dir)
    sandbox.fs.upload_file(compose_content, f"{_COMPOSE_DIR}/docker-compose.yml")

    click.echo(f"{prefix}Uploading compose bundle assets...")
    _upload_compose_bundle(sandbox, compose_dir)

    env_file = compose_dir / ".env"
    if env_file.exists():
        click.echo(f"{prefix}Uploading .env...")
        sandbox.fs.upload_file(env_file.read_bytes(), f"{_COMPOSE_DIR}/.env")

    # Build local-context images in the sandbox.
    for service_name, (local_context, image_tag) in build_contexts.items():
        click.echo(f"{prefix}Uploading build context for {service_name}...")
        remote_dir = f"{_COMPOSE_DIR}/build-contexts/{service_name}"
        sandbox.process.exec(f"mkdir -p {remote_dir}", cwd=_COMPOSE_DIR)
        _upload_directory(sandbox, local_context, remote_dir)
        click.echo(f"{prefix}Building {image_tag}...")
        resp = sandbox.process.exec(
            f"docker build -t {image_tag} {remote_dir}",
            cwd=_COMPOSE_DIR,
            timeout=300,
        )
        if resp.exit_code != 0:
            raise RuntimeError(f"docker build failed for {service_name}:\n{resp.result}")

    # Pull images
    click.echo(f"{prefix}Pulling images...")
    resp = sandbox.process.exec(
        "docker compose pull --ignore-buildable",
        cwd=_COMPOSE_DIR,
        timeout=300,
    )
    if resp.exit_code != 0:
        # Fallback: older docker compose may not support --ignore-buildable.
        resp = sandbox.process.exec(
            "docker compose pull --ignore-pull-failures",
            cwd=_COMPOSE_DIR,
            timeout=300,
        )
        if resp.exit_code != 0:
            raise RuntimeError(f"docker compose pull failed:\n{resp.result}")

    if preseed_svc_names:
        _run_profiled_services_in_sandbox(
            sandbox,
            preseed_svc_names,
            profile="preseed",
            log_prefix=log_prefix,
        )

    # Start services
    click.echo(f"{prefix}Starting services...")
    resp = sandbox.process.exec(
        "docker compose up -d",
        cwd=_COMPOSE_DIR,
        timeout=120,
    )
    if resp.exit_code != 0:
        raise RuntimeError(f"docker compose up failed:\n{resp.result}")

    # Get public URLs for each tool port
    urls: dict[str, str] = {}
    for tool_name, port in tool_ports.items():
        preview = sandbox.get_preview_link(port)
        urls[tool_name] = preview.url

    return urls


def teardown_sandbox(
    daytona_client: Any,  # noqa: ANN401
    sandbox: Any,  # noqa: ANN401
    log_prefix: str = "",
) -> bool:
    """Tear down a sandbox: stop services and delete.

    Returns:
        ``True`` if the sandbox was successfully deleted, ``False`` otherwise.

    """
    prefix = f"{log_prefix} " if log_prefix else ""
    try:
        sandbox.process.exec("docker compose down", cwd=_COMPOSE_DIR, timeout=60)
    except Exception as e:
        _logger.debug("%sCould not stop services: %s", prefix, e)

    try:
        daytona_client.stop(sandbox)
    except Exception as e:
        _logger.debug("%sCould not stop sandbox: %s", prefix, e)

    try:
        daytona_client.delete(sandbox)
    except Exception as e:
        _logger.debug("%sCould not delete sandbox: %s", prefix, e)
        return False
    return True


class DaytonaRunner:
    """Runs docker-compose inside a Daytona sandbox."""

    def __init__(self, daytona_api_key: str | None = None) -> None:
        """Store optional Daytona API key (else read from DAYTONA_API_KEY env)."""
        self._daytona_api_key = daytona_api_key

    def up(
        self,
        compose_dir: Path,
        tool_ports: dict[str, int],
        preseed_svc_names: list[str] | None = None,
        reporter: Any | None = None,  # noqa: ANN401
    ) -> dict[str, str]:
        """Create a Daytona sandbox and start docker-compose services.

        Args:
            compose_dir: Local directory containing docker-compose.yml and .env.
            tool_ports: Mapping of tool name to port number.
            preseed_svc_names: Services to run with the ``preseed`` profile before startup.
            reporter: Optional :class:`ProgressReporter` for structured output.

        Returns:
            Mapping of tool name to public URL.

        """
        rpt = reporter if reporter is not None else DefaultReporter()

        ensure_snapshot_exists(self._daytona_api_key)

        daytona = _get_daytona(self._daytona_api_key)

        # Create sandbox from the docker-dind snapshot
        rpt.start_step("Daytona sandbox created")
        sandbox = daytona.create(
            CreateSandboxFromSnapshotParams(
                snapshot=_SNAPSHOT_NAME,
                public=True,
            ),
        )
        state_file = compose_dir / _STATE_FILE
        state_file.write_text(json.dumps({"sandbox_id": sandbox.id}))
        rpt.end_step(success=True)

        try:
            sandbox.process.create_session("init")

            # Wait for Docker daemon
            rpt.start_step("Docker daemon ready")
            daemon_ready = False
            for attempt in range(20):
                resp = sandbox.process.execute_session_command(
                    "init", SessionExecuteRequest(command="docker info"), timeout=5
                )
                if resp.exit_code == 0:
                    daemon_ready = True
                    break
                if attempt == 0:
                    rpt.detail("Docker daemon still starting; retrying...")
                time.sleep(3)

            if not daemon_ready:
                rpt.detail("Attempting to start Docker daemon manually...")
                sandbox.process.execute_session_command(
                    "init",
                    SessionExecuteRequest(command="nohup dockerd &", run_async=True),
                )
                for _ in range(20):
                    resp = sandbox.process.execute_session_command(
                        "init", SessionExecuteRequest(command="docker info"), timeout=5
                    )
                    if resp.exit_code == 0:
                        daemon_ready = True
                        break
                    time.sleep(3)

            if not daemon_ready:
                rpt.end_step(success=False, error="Docker daemon failed to start in sandbox.")
                raise SystemExit(1)
            rpt.end_step(success=True)

            # Upload compose files
            rpt.start_step("Environment files uploaded")
            compose_content, build_contexts = _prepare_compose_for_remote(compose_dir)
            sandbox.fs.upload_file(compose_content, f"{_COMPOSE_DIR}/docker-compose.yml")
            rpt.detail("Uploaded docker-compose.yml")

            rpt.detail("Uploading compose bundle assets")
            _upload_compose_bundle(sandbox, compose_dir)

            env_file = compose_dir / ".env"
            if env_file.exists():
                sandbox.fs.upload_file(env_file.read_bytes(), f"{_COMPOSE_DIR}/.env")
                rpt.detail("Uploaded .env")
            self._upload_support_files(sandbox, compose_dir)
            rpt.end_step(success=True)

            # Build local-context images in the sandbox.
            if build_contexts:
                rpt.start_step("Images built")
                for service_name, (local_context, image_tag) in build_contexts.items():
                    rpt.detail(f"Uploading build context for {service_name}...")
                    remote_dir = f"{_COMPOSE_DIR}/build-contexts/{service_name}"
                    sandbox.process.exec(f"mkdir -p {remote_dir}", cwd=_COMPOSE_DIR)
                    _upload_directory(sandbox, local_context, remote_dir)
                    rpt.detail(f"Building {image_tag}...")
                    resp = sandbox.process.exec(
                        f"docker build -t {image_tag} {remote_dir}",
                        cwd=_COMPOSE_DIR,
                        timeout=300,
                    )
                    if resp.exit_code != 0:
                        rpt.end_step(
                            success=False,
                            error=f"docker build failed for {service_name}:\n{resp.result}",
                        )
                        raise SystemExit(1)
                rpt.end_step(success=True)
            # Pull images
            rpt.start_step("Images pulled")
            resp = sandbox.process.exec(
                "docker compose pull --ignore-buildable",
                cwd=_COMPOSE_DIR,
                timeout=300,
            )
            if resp.exit_code != 0:
                resp = sandbox.process.exec(
                    "docker compose pull --ignore-pull-failures",
                    cwd=_COMPOSE_DIR,
                    timeout=300,
                )
                if resp.exit_code != 0:
                    rpt.end_step(
                        success=False,
                        error=f"docker compose pull failed:\n{resp.result}",
                    )
                    raise SystemExit(1)
            rpt.end_step(success=True)

            if preseed_svc_names:
                rpt.start_step("Environment preseeded")
                try:
                    _run_profiled_services_in_sandbox(
                        sandbox,
                        preseed_svc_names,
                        profile="preseed",
                        quiet=True,
                        reporter=rpt,
                    )
                except RuntimeError as e:
                    rpt.end_step(success=False, error=str(e))
                    with contextlib.suppress(Exception):
                        daytona.delete(sandbox)
                        state_file.unlink(missing_ok=True)
                    raise SystemExit(1) from e
                rpt.end_step(success=True)

            # Start services
            rpt.start_step("Services started")
            resp = sandbox.process.exec(
                "docker compose up -d",
                cwd=_COMPOSE_DIR,
                timeout=120,
            )
            if resp.exit_code != 0:
                rpt.end_step(
                    success=False,
                    error=f"docker compose up failed:\n{resp.result}",
                )
                raise SystemExit(1)
            rpt.end_step(success=True)

            # Get public URLs for each tool port
            urls: dict[str, str] = {}
            for tool_name, port in tool_ports.items():
                preview = sandbox.get_preview_link(port)
                urls[tool_name] = preview.url

            state_file.write_text(json.dumps({"sandbox_id": sandbox.id}))

            return urls

        except Exception:
            with contextlib.suppress(Exception):
                daytona.delete(sandbox)
                state_file.unlink(missing_ok=True)
            raise

    def get_urls(
        self,
        compose_dir: Path,
        tool_ports: dict[str, int],
    ) -> dict[str, str]:
        """Get public URLs for a running Daytona sandbox."""
        state_file = compose_dir / _STATE_FILE
        if not state_file.exists():
            click.echo(
                click.style(
                    f"No Daytona state at {state_file}. Is the environment running?",
                    fg="red",
                ),
                err=True,
            )
            raise SystemExit(1)

        state = json.loads(state_file.read_text())
        sandbox_id = state["sandbox_id"]
        daytona = _get_daytona(self._daytona_api_key)
        sandbox = daytona.get(sandbox_id)

        urls: dict[str, str] = {}
        for tool_name, port in tool_ports.items():
            preview = sandbox.get_preview_link(port)
            urls[tool_name] = preview.url
        return urls

    def restart_sandbox_services(
        self,
        sandbox: Any,  # noqa: ANN401
        *,
        preseed_svc_names: list[str] | None = None,
        log_prefix: str = "",
    ) -> None:
        """Restart compose services inside a resumed Daytona sandbox."""
        restart_sandbox_environment(
            sandbox,
            preseed_svc_names=preseed_svc_names,
            log_prefix=log_prefix,
        )

    def down(self, compose_dir: Path) -> None:
        """Tear down the Daytona sandbox."""
        state_file = compose_dir / _STATE_FILE
        if not state_file.exists():
            click.echo(
                click.style(
                    f"No Daytona state found at {state_file}. Is the environment running?",
                    fg="red",
                ),
                err=True,
            )
            raise SystemExit(1)

        state = json.loads(state_file.read_text())
        sandbox_id = state["sandbox_id"]
        daytona = _get_daytona(self._daytona_api_key)

        click.echo("Stopping services and deleting sandbox...")
        deleted = False
        sandbox_gone = False
        try:
            sandbox = daytona.get(sandbox_id)
            deleted = teardown_sandbox(daytona, sandbox)
        except DaytonaNotFoundError:
            # Sandbox genuinely gone (deleted out-of-band or expired).
            sandbox_gone = True
        except Exception as e:
            # Transient API error — sandbox may still be alive, keep state file.
            click.echo(
                click.style(f"Warning: could not reach Daytona API: {e}", fg="yellow"),
                err=True,
            )

        if deleted or sandbox_gone:
            state_file.unlink(missing_ok=True)
            click.echo(click.style("Daytona sandbox torn down.", fg="green"))
        else:
            click.echo(
                click.style(
                    f"Warning: sandbox {sandbox_id} may not have been deleted. "
                    f"State file kept at {state_file} for manual cleanup.",
                    fg="yellow",
                ),
                err=True,
            )

    def seed(
        self,
        compose_dir: Path,
        seed_svc_names: list[str],
        env_overrides: dict[str, str] | None = None,
    ) -> str:
        """Run seed containers inside the Daytona sandbox.

        Returns:
            Combined output from all seed containers.

        """
        return self.run_profiled_services(
            compose_dir,
            seed_svc_names,
            profile="seed",
            env_overrides=env_overrides,
        )

    def run_profiled_services(
        self,
        compose_dir: Path,
        svc_names: list[str],
        *,
        profile: str,
        env_overrides: dict[str, str] | None = None,
    ) -> str:
        """Run profiled compose services inside an existing Daytona sandbox."""
        state_file = compose_dir / _STATE_FILE
        if not state_file.exists():
            click.echo(
                click.style(
                    f"No Daytona state at {state_file}. Is the env running?",
                    fg="red",
                ),
                err=True,
            )
            raise SystemExit(1)

        state = json.loads(state_file.read_text())
        sandbox_id = state["sandbox_id"]
        daytona = _get_daytona(self._daytona_api_key)
        sandbox = daytona.get(sandbox_id)

        try:
            return _run_profiled_services_in_sandbox(
                sandbox,
                svc_names,
                profile=profile,
                env_overrides=env_overrides,
            )
        except RuntimeError as e:
            click.echo(click.style(str(e), fg="red"), err=True)
            raise SystemExit(1) from e

    def get_health(self, compose_dir: Path) -> dict[str, str]:
        """Get service health from inside the Daytona sandbox."""
        state_file = compose_dir / _STATE_FILE
        if not state_file.exists():
            return {}

        state = json.loads(state_file.read_text())
        sandbox_id = state["sandbox_id"]

        try:
            daytona = _get_daytona(self._daytona_api_key)
            sandbox = daytona.get(sandbox_id)
            resp = sandbox.process.exec(
                'docker compose ps --format "{{.Name}}\t{{.Status}}"',
                cwd=_COMPOSE_DIR,
            )
            if resp.exit_code != 0:
                return {}
            return parse_ps_output(resp.result)
        except Exception:
            return {}

    def _prepare_compose_for_remote(
        self,
        compose_dir: Path,
    ) -> bytes:
        """Return compose content for remote execution."""
        compose_bytes, _ = _prepare_compose_for_remote(compose_dir)
        return compose_bytes

    def _upload_directory(self, sandbox: Any, local_dir: Path, remote_dir: str) -> None:  # noqa: ANN401
        """Upload a local directory recursively into the sandbox."""
        for file_path in local_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(local_dir)
            remote_path = f"{remote_dir}/{rel_path.as_posix()}"
            remote_parent = remote_path.rsplit("/", 1)[0]
            sandbox.process.exec(f"mkdir -p {remote_parent}", cwd=_COMPOSE_DIR)
            sandbox.fs.upload_file(file_path.read_bytes(), remote_path)

    def _upload_compose_bundle(self, sandbox: Any, compose_dir: Path) -> None:  # noqa: ANN401
        """Upload bundle-local assets referenced by docker-compose relative paths."""
        ignored = {"docker-compose.yml", ".env", _STATE_FILE}
        for entry in compose_dir.iterdir():
            if entry.name in ignored:
                continue
            remote_path = f"{_COMPOSE_DIR}/{entry.name}"
            if entry.is_dir():
                sandbox.process.exec(f"mkdir -p {remote_path}", cwd=_COMPOSE_DIR)
                self._upload_directory(sandbox, entry, remote_path)
                continue
            sandbox.fs.upload_file(entry.read_bytes(), remote_path)

    def _upload_support_files(self, sandbox: Any, compose_dir: Path) -> None:  # noqa: ANN401
        """Upload extra compose-sidecar files required by mounted services."""
        gateway_config = compose_dir / _MCP_GATEWAY_CONFIG_FILE
        if gateway_config.exists():
            click.echo(f"Uploading {_MCP_GATEWAY_CONFIG_FILE}...")
            sandbox.fs.upload_file(
                gateway_config.read_bytes(),
                f"{_COMPOSE_DIR}/{_MCP_GATEWAY_CONFIG_FILE}",
            )

    def _run_profiled_services_in_sandbox(
        self,
        sandbox: Any,  # noqa: ANN401
        svc_names: list[str],
        *,
        profile: str,
        env_overrides: dict[str, str] | None = None,
        quiet: bool = False,
        reporter: Any | None = None,  # noqa: ANN401
    ) -> str:
        """Run ``docker compose run`` for services in a compose profile.

        When *quiet* is True, output is routed through *reporter* (if provided)
        instead of being printed directly via ``click.echo``.
        """
        output_parts: list[str] = []
        phase_label = "Preseeding" if profile == "preseed" else "Seeding"

        def _echo(msg: str) -> None:
            if quiet and reporter is not None:
                reporter.detail(msg)
            elif not quiet:
                click.echo(msg)

        for svc_name in svc_names:
            _echo(f"{phase_label}: {svc_name}...")
            override_args = ""
            for key, value in (env_overrides or {}).items():
                override_args += f" -e {shlex.quote(f'{key}={value}')}"
            compose_cmd = (
                f"docker compose --profile {profile} run --rm"
                f"{override_args} {shlex.quote(svc_name)}"
            )
            resp = sandbox.process.exec(
                compose_cmd,
                cwd=_COMPOSE_DIR,
                timeout=600,
            )
            if resp.exit_code != 0:
                click.echo(
                    click.style(
                        f"{phase_label[:-3]} '{svc_name}' failed:\n{resp.result}",
                        fg="red",
                    ),
                    err=True,
                )
                raise SystemExit(1)
            output_parts.append(resp.result or "")
            if env_overrides:
                rendered = " ".join(
                    f"{key}={shlex.quote(value)}" for key, value in env_overrides.items()
                )
                _echo(f"  Applied overrides: {rendered}")

        if not quiet:
            click.echo(click.style(f"{phase_label} complete.", fg="green"))
        return "\n".join(output_parts)


def ensure_snapshot_exists(api_key: str | None = None) -> None:
    """Create the docker-dind snapshot if it doesn't already exist."""
    daytona = _get_daytona(api_key)

    # Check if snapshot already exists
    try:
        daytona.snapshot.get(_SNAPSHOT_NAME)
    except Exception as e:
        logging.getLogger(__name__).debug("Snapshot get (will create): %s", e)
    else:
        click.echo(click.style(f"Snapshot '{_SNAPSHOT_NAME}' already exists.", fg="green"))
        return

    click.echo(f"Creating snapshot '{_SNAPSHOT_NAME}' from docker:28.3.3-dind...")
    click.echo("This may take a few minutes on first run.\n")

    daytona.snapshot.create(
        CreateSnapshotParams(
            name=_SNAPSHOT_NAME,
            image="docker:28.3.3-dind",
            resources=Resources(cpu=4, memory=8, disk=10),
        ),
        on_logs=sys.stdout.write,
    )
    click.echo(click.style(f"\nSnapshot '{_SNAPSHOT_NAME}' created successfully.", fg="green"))
