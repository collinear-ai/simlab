"""Run docker-compose environments inside a Daytona sandbox."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import click
import yaml

from simlab.config import resolve_daytona_api_key

try:
    from daytona import CreateSandboxFromSnapshotParams
    from daytona import CreateSnapshotParams
    from daytona import Daytona
    from daytona import DaytonaConfig
    from daytona import Resources
    from daytona import SessionExecuteRequest
except ImportError as e:
    DAYTONA_IMPORT_ERROR: ImportError | None = e
    CreateSandboxFromSnapshotParams = Any  # type: ignore[misc,assignment]
    CreateSnapshotParams = Any  # type: ignore[misc,assignment]
    Daytona = Any  # type: ignore[misc,assignment]
    DaytonaConfig = Any  # type: ignore[misc,assignment]
    Resources = Any  # type: ignore[misc,assignment]
    SessionExecuteRequest = Any  # type: ignore[misc,assignment]
else:
    DAYTONA_IMPORT_ERROR = None

from simlab.runtime.compose_ps import parse_ps_output

_STATE_FILE = "daytona-state.json"
_SNAPSHOT_NAME = "docker-dind"
_COMPOSE_DIR = "/home/daytona"


def _get_daytona(api_key: str | None = None):  # noqa: ANN202
    """Build an authenticated Daytona client, or exit with a helpful message."""
    if DAYTONA_IMPORT_ERROR is not None:
        click.echo(
            click.style(
                "Daytona support requires optional dependency 'daytona'. "
                "Install with: pip install 'simlab[daytona]'",
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
    ) -> dict[str, str]:
        """Create a Daytona sandbox and start docker-compose services.

        Args:
            compose_dir: Local directory containing docker-compose.yml and .env.
            tool_ports: Mapping of tool name to port number.
            preseed_svc_names: Services to run with the ``preseed`` profile before startup.

        Returns:
            Mapping of tool name to public URL.

        """
        ghcr_token = os.environ.get("GHCR_TOKEN", "")
        ghcr_username = os.environ.get("GHCR_USERNAME", "sachin-patro")

        daytona = _get_daytona(self._daytona_api_key)

        # Create sandbox from the docker-dind snapshot (resources are set on the snapshot)
        click.echo("Creating Daytona sandbox...")
        sandbox = daytona.create(
            CreateSandboxFromSnapshotParams(
                snapshot=_SNAPSHOT_NAME,
                public=True,
            ),
        )
        state_file = compose_dir / _STATE_FILE
        state_file.write_text(json.dumps({"sandbox_id": sandbox.id}))

        try:
            # Use session API so we get separate stdout/stderr
            sandbox.process.create_session("init")

            # Wait for Docker daemon (started automatically by snapshot entrypoint)
            click.echo("Waiting for Docker daemon...")
            daemon_ready = False
            for attempt in range(20):
                resp = sandbox.process.execute_session_command(
                    "init", SessionExecuteRequest(command="docker info"), timeout=5
                )
                if resp.exit_code == 0:
                    click.echo("Docker daemon ready.")
                    daemon_ready = True
                    break
                if attempt == 0:
                    click.echo("  Docker daemon still starting; retrying...")
                time.sleep(3)

            if not daemon_ready:
                # Try starting dockerd manually as a fallback
                click.echo("Attempting to start Docker daemon manually...")
                sandbox.process.execute_session_command(
                    "init",
                    SessionExecuteRequest(command="nohup dockerd &", run_async=True),
                )
                for _ in range(20):
                    resp = sandbox.process.execute_session_command(
                        "init", SessionExecuteRequest(command="docker info"), timeout=5
                    )
                    if resp.exit_code == 0:
                        click.echo("Docker daemon ready (manual start).")
                        daemon_ready = True
                        break
                    time.sleep(3)

            if not daemon_ready:
                click.echo(
                    click.style("Docker daemon failed to start in sandbox.", fg="red"),
                    err=True,
                )
                click.echo(f"  stdout: {resp.stdout}", err=True)
                click.echo(f"  stderr: {resp.stderr}", err=True)
                raise SystemExit(1)

            # Upload compose files
            click.echo("Uploading docker-compose.yml...")
            compose_content, build_contexts = self._prepare_compose_for_remote(compose_dir)
            sandbox.fs.upload_file(compose_content, f"{_COMPOSE_DIR}/docker-compose.yml")

            env_file = compose_dir / ".env"
            if env_file.exists():
                click.echo("Uploading .env...")
                sandbox.fs.upload_file(env_file.read_bytes(), f"{_COMPOSE_DIR}/.env")

            # Log in to GHCR inside sandbox
            if ghcr_token:
                click.echo("Logging into GHCR...")
                resp = sandbox.process.exec(
                    f"echo {ghcr_token} | docker login ghcr.io -u {ghcr_username} --password-stdin",
                    cwd=_COMPOSE_DIR,
                )
                if resp.exit_code != 0:
                    click.echo(
                        click.style(f"GHCR login failed: {resp.result}", fg="red"),
                        err=True,
                    )
                    raise SystemExit(1)
            else:
                click.echo(
                    click.style(
                        "Warning: GHCR_TOKEN not set — private images will fail to pull.",
                        fg="yellow",
                    ),
                    err=True,
                )

            # Build local-context images in the sandbox.
            for service_name, (local_context, image_tag) in build_contexts.items():
                click.echo(f"Uploading build context for {service_name}...")
                remote_dir = f"{_COMPOSE_DIR}/build-contexts/{service_name}"
                sandbox.process.exec(f"mkdir -p {remote_dir}", cwd=_COMPOSE_DIR)
                self._upload_directory(sandbox, local_context, remote_dir)
                click.echo(f"Building {image_tag}...")
                resp = sandbox.process.exec(
                    f"docker build -t {image_tag} {remote_dir}",
                    cwd=_COMPOSE_DIR,
                    timeout=300,
                )
                if resp.exit_code != 0:
                    click.echo(
                        click.style(
                            f"docker build failed for {service_name}:\n{resp.result}", fg="red"
                        ),
                        err=True,
                    )
                    raise SystemExit(1)

            # Pull images
            click.echo("Pulling images...")
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
                    click.echo(
                        click.style(f"docker compose pull failed:\n{resp.result}", fg="red"),
                        err=True,
                    )
                    raise SystemExit(1)

            if preseed_svc_names:
                self._run_profiled_services_in_sandbox(
                    sandbox,
                    preseed_svc_names,
                    profile="preseed",
                )

            # Start services
            click.echo("Starting services...")
            resp = sandbox.process.exec(
                "docker compose up -d",
                cwd=_COMPOSE_DIR,
                timeout=120,
            )
            if resp.exit_code != 0:
                click.echo(
                    click.style(f"docker compose up failed:\n{resp.result}", fg="red"),
                    err=True,
                )
                raise SystemExit(1)

            # Get public URLs for each tool port
            urls: dict[str, str] = {}
            for tool_name, port in tool_ports.items():
                preview = sandbox.get_preview_link(port)
                urls[tool_name] = preview.url

            # Refresh state for later teardown
            state_file.write_text(json.dumps({"sandbox_id": sandbox.id}))

            return urls

        except Exception:
            # Cleanup on failure
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

        click.echo("Stopping services in sandbox...")
        try:
            sandbox = daytona.get(sandbox_id)
            sandbox.process.exec("docker compose down", cwd=_COMPOSE_DIR, timeout=60)
        except Exception as e:
            click.echo(
                click.style(f"Warning: could not stop services: {e}", fg="yellow"),
                err=True,
            )

        click.echo("Deleting sandbox...")
        try:
            sandbox = daytona.get(sandbox_id)
            daytona.delete(sandbox)
        except Exception as e:
            click.echo(
                click.style(f"Warning: could not delete sandbox: {e}", fg="yellow"),
                err=True,
            )

        state_file.unlink(missing_ok=True)
        click.echo(click.style("Daytona sandbox torn down.", fg="green"))

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

        return self._run_profiled_services_in_sandbox(
            sandbox,
            svc_names,
            profile=profile,
            env_overrides=env_overrides,
        )

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
    ) -> tuple[bytes, dict[str, tuple[Path, str]]]:
        """Strip local build contexts and collect contexts to build remotely."""
        compose_path = compose_dir / "docker-compose.yml"
        compose_data = yaml.safe_load(compose_path.read_text())
        services = compose_data.get("services", {}) if isinstance(compose_data, dict) else {}

        build_contexts: dict[str, tuple[Path, str]] = {}
        for service_name, service in services.items():
            if not isinstance(service, dict):
                continue
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

    def _run_profiled_services_in_sandbox(
        self,
        sandbox: Any,  # noqa: ANN401
        svc_names: list[str],
        *,
        profile: str,
        env_overrides: dict[str, str] | None = None,
    ) -> str:
        """Run ``docker compose run`` for services in a compose profile."""
        output_parts: list[str] = []
        phase_label = "Preseeding" if profile == "preseed" else "Seeding"
        for svc_name in svc_names:
            click.echo(f"{phase_label}: {svc_name}...")
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
                click.echo(f"  Applied overrides: {rendered}")

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
