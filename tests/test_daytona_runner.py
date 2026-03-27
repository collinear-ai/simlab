from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import simlab.runtime.daytona_runner as daytona_runner_mod
import yaml
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

    rendered = DaytonaRunner()._prepare_compose_for_remote(compose_dir)
    compose = yaml.safe_load(rendered.decode("utf-8"))

    service = compose["services"]["mcp-gateway"]
    assert "build" not in service
    assert service["image"] == "mcp-gateway:latest"
    assert service["volumes"] == [
        "/home/daytona/mcp-gateway-config.json:/config/mcp-gateway-config.json:ro"
    ]


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
