from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import simlab.runtime.daytona_runner as daytona_runner_mod
import yaml
from rich.console import Console
from simlab.cli.progress import StepProgress
from simlab.cli.progress import StepProgressReporter
from simlab.runtime.daytona_runner import DaytonaRunner


class _FakeFs:
    def __init__(self) -> None:
        self.uploads: list[tuple[bytes, str]] = []

    def upload_file(self, content: bytes, remote_path: str) -> None:
        self.uploads.append((content, remote_path))


def test_upload_support_files_includes_gateway_config(tmp_path: Path) -> None:
    compose_dir = tmp_path / "env"
    compose_dir.mkdir()
    config_path = compose_dir / "mcp-gateway-config.json"
    config_path.write_text(json.dumps({"servers": []}), encoding="utf-8")
    sandbox = SimpleNamespace(fs=_FakeFs())

    DaytonaRunner()._upload_support_files(sandbox, compose_dir)

    assert sandbox.fs.uploads == [
        (config_path.read_bytes(), "/home/daytona/mcp-gateway-config.json")
    ]


def test_upload_support_files_skips_missing_gateway_config(tmp_path: Path) -> None:
    compose_dir = tmp_path / "env"
    compose_dir.mkdir()
    sandbox = SimpleNamespace(fs=_FakeFs())

    DaytonaRunner()._upload_support_files(sandbox, compose_dir)

    assert sandbox.fs.uploads == []


def test_get_health_handles_hyphenated_project_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compose_dir = tmp_path / "env"
    compose_dir.mkdir()
    (compose_dir / "docker-compose.yml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "harbor-openhands-agent-server": {},
                    "harbor-coding-env": {},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (compose_dir / "daytona-state.json").write_text(
        json.dumps({"sandbox_id": "sbx-1"}),
        encoding="utf-8",
    )

    class FakeProcess:
        def exec(self, command: str, *, cwd: str | None = None) -> SimpleNamespace:
            assert command == 'docker compose ps --format "{{.Name}}\t{{.Status}}"'
            assert cwd == "/home/daytona"
            return SimpleNamespace(
                exit_code=0,
                result=(
                    "baseline-env-harbor-openhands-agent-server-1\tUp 5 seconds (healthy)\n"
                    "baseline-env-harbor-coding-env-1\tUp 5 seconds (health: starting)\n"
                ),
            )

    class FakeDaytona:
        def get(self, sandbox_id: str) -> SimpleNamespace:
            assert sandbox_id == "sbx-1"
            return SimpleNamespace(process=FakeProcess())

    monkeypatch.setattr(daytona_runner_mod, "_get_daytona", lambda _api_key=None: FakeDaytona())

    health = DaytonaRunner().get_health(compose_dir)

    assert health == {
        "harbor-openhands-agent-server": "healthy",
        "harbor-coding-env": "starting",
    }


def test_prepare_compose_for_remote_rewrites_local_build_and_mounts(tmp_path: Path) -> None:
    compose_dir = tmp_path / "env"
    compose_dir.mkdir()
    gateway_dir = compose_dir / "gateway"
    gateway_dir.mkdir()
    (gateway_dir / "Dockerfile").write_text("FROM busybox\n", encoding="utf-8")
    gateway_config = compose_dir / "mcp-gateway-config.json"
    gateway_config.write_text('{"servers":[]}\n', encoding="utf-8")
    compose_path = compose_dir / "docker-compose.yml"
    compose_path.write_text(
        """
services:
  mcp-gateway:
    build:
      context: ./gateway
      dockerfile: Dockerfile
    volumes:
      - /placeholder
    ports:
      - "8080:8080"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    compose["services"]["mcp-gateway"]["volumes"] = [
        f"{gateway_config}:/config/mcp-gateway-config.json:ro"
    ]
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False), encoding="utf-8")

    rendered, build_contexts = daytona_runner_mod._prepare_compose_for_remote(compose_dir)
    compose = yaml.safe_load(rendered.decode("utf-8"))

    service = compose["services"]["mcp-gateway"]
    assert "build" not in service
    assert service["image"] == "mcp-gateway:latest"
    assert service["volumes"] == [
        "/home/daytona/mcp-gateway-config.json:/config/mcp-gateway-config.json:ro"
    ]
    assert build_contexts["mcp-gateway"] == (
        gateway_dir.resolve(),
        "mcp-gateway:latest",
        "Dockerfile",
    )


def test_prepare_compose_for_remote_rejects_external_build_context(tmp_path: Path) -> None:
    compose_dir = tmp_path / "env"
    compose_dir.mkdir()
    external_dir = tmp_path / "external-build"
    external_dir.mkdir()
    (external_dir / "Dockerfile").write_text("FROM busybox\n", encoding="utf-8")
    (compose_dir / "docker-compose.yml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "harbor-main": {
                        "build": {
                            "context": external_dir.as_posix(),
                            "dockerfile": "Dockerfile",
                        }
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="inside the environment bundle"):
        daytona_runner_mod._prepare_compose_for_remote(compose_dir)


def test_prepare_compose_for_remote_preserves_custom_dockerfile(tmp_path: Path) -> None:
    compose_dir = tmp_path / "env"
    compose_dir.mkdir()
    harbor_dir = compose_dir / "harbor-environment"
    harbor_dir.mkdir()
    (harbor_dir / "Dockerfile").write_text("FROM busybox\n", encoding="utf-8")
    (harbor_dir / "Dockerfile.simlab-wrapper").write_text("FROM busybox\n", encoding="utf-8")
    (compose_dir / "docker-compose.yml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "harbor-openhands-agent-server": {
                        "build": {
                            "context": "./harbor-environment",
                            "dockerfile": "Dockerfile.simlab-wrapper",
                        }
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _, build_contexts = daytona_runner_mod._prepare_compose_for_remote(compose_dir)

    assert build_contexts["harbor-openhands-agent-server"] == (
        harbor_dir.resolve(),
        "harbor-openhands-agent-server:latest",
        "Dockerfile.simlab-wrapper",
    )


def test_restart_sandbox_services_runs_preseed_then_compose_up(monkeypatch) -> None:  # noqa: ANN001
    session_commands: list[str] = []
    exec_commands: list[str] = []

    class FakeProcess:
        def create_session(self, name: str) -> None:
            session_commands.append(f"create:{name}")

        def execute_session_command(
            self,
            session_name: str,
            request: SimpleNamespace,
            timeout: int | None = None,
        ) -> SimpleNamespace:
            _ = timeout
            session_commands.append(f"{session_name}:{request.command}")
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        def exec(
            self,
            command: str,
            *,
            cwd: str | None = None,
            timeout: int | None = None,
        ) -> SimpleNamespace:
            _ = cwd, timeout
            exec_commands.append(command)
            return SimpleNamespace(exit_code=0, result="ok")

    sandbox = SimpleNamespace(process=FakeProcess())
    preseed_calls: list[tuple[list[str], str]] = []

    def fake_session_execute_request(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        daytona_runner_mod,
        "SessionExecuteRequest",
        fake_session_execute_request,
    )

    def fake_run_profiled_services(
        _sandbox: object,
        services: list[str],
        *,
        profile: str,
        **_kwargs: object,
    ) -> str:
        preseed_calls.append((list(services), profile))
        return ""

    monkeypatch.setattr(
        daytona_runner_mod,
        "_run_profiled_services_in_sandbox",
        fake_run_profiled_services,
    )

    DaytonaRunner().restart_sandbox_services(
        sandbox,
        preseed_svc_names=["email-preseed"],
    )

    assert session_commands[:2] == ["create:init", "init:docker info"]
    assert preseed_calls == [(["email-preseed"], "preseed")]
    assert exec_commands == ["docker compose up -d"]


def test_down_uses_reporter_without_trampling_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compose_dir = tmp_path / "env"
    compose_dir.mkdir()
    (compose_dir / "daytona-state.json").write_text(
        json.dumps({"sandbox_id": "sbx-1"}),
        encoding="utf-8",
    )

    sandbox = SimpleNamespace(id="sbx-1")

    class FakeDaytona:
        def get(self, sandbox_id: str) -> SimpleNamespace:
            assert sandbox_id == "sbx-1"
            return sandbox

    monkeypatch.setattr(daytona_runner_mod, "_get_daytona", lambda _api_key=None: FakeDaytona())
    monkeypatch.setattr(daytona_runner_mod, "teardown_sandbox", lambda *_args: True)

    out = io.StringIO()
    err = io.StringIO()
    progress = StepProgress(
        verbose=False,
        console=Console(file=out, force_terminal=False, highlight=False),
        err_console=Console(file=err, force_terminal=False, highlight=False),
    )

    DaytonaRunner().down(compose_dir, reporter=StepProgressReporter(progress))

    output = out.getvalue()
    assert "[done] Daytona sandbox torn down" in output
    assert "Stopping services and deleting sandbox..." not in output
    assert err.getvalue() == ""


def test_collect_compose_debug_output_includes_status_and_logs() -> None:
    commands: list[str] = []

    class FakeProcess:
        def exec(
            self,
            command: str,
            *,
            cwd: str | None = None,
            timeout: int | None = None,
        ) -> SimpleNamespace:
            _ = cwd, timeout
            commands.append(command)
            if command == "docker compose ps --all":
                return SimpleNamespace(exit_code=0, result="svc up")
            if command == "docker compose logs --no-color --tail=200":
                return SimpleNamespace(exit_code=0, result="container log")
            return SimpleNamespace(exit_code=1, result="")

    sandbox = SimpleNamespace(process=FakeProcess())

    output = daytona_runner_mod._collect_compose_debug_output(sandbox)

    assert "Compose status:\nsvc up" in output
    assert "Compose logs:\ncontainer log" in output
    assert commands == [
        "docker compose ps --all",
        "docker compose logs --no-color --tail=200",
    ]
