from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner
from simlab.agents import loader as agents_loader
from simlab.agents.base import ToolCall
from simlab.agents.base import ToolCallResult
from simlab.cli import tasks as tasks_cli
from simlab.runtime import rollout_runner
from simlab.runtime.adapters.harbor.prepare import parse_harbor_task
from simlab.runtime.adapters.harbor.prepare import prepare_harbor_run
from simlab.runtime.adapters.harbor.trajectory import build_atif_trajectory
from simlab.runtime.adapters.harbor.urls import compose_service_host_port
from simlab.runtime.adapters.harbor.urls import rewrite_mcp_config_for_runtime
from simlab.runtime.adapters.harbor.verifier import ComposeExecResult
from simlab.runtime.adapters.harbor.verifier import run_compose_exec
from simlab.runtime.adapters.harbor.verifier import run_harbor_verifier


def _write_harbor_task(
    base: Path,
    *,
    with_mcp: bool = False,
    exec_form_command: bool = False,
    passthrough_env: bool = False,
    workdir: str = "/app",
) -> Path:
    task_dir = base / ("hello-mcp" if with_mcp else "hello-world")
    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    environment_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text(
        "Create /app/hello.txt with the expected content.",
        encoding="utf-8",
    )
    (tests_dir / "test.sh").write_text(
        "#!/usr/bin/env bash\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n",
        encoding="utf-8",
    )
    docker_workdir = json.dumps(workdir) if " " in workdir else workdir
    (environment_dir / "Dockerfile").write_text(
        f"FROM ubuntu:24.04\nWORKDIR {docker_workdir}\n",
        encoding="utf-8",
    )

    task_toml = [
        'version = "1.0"',
        "",
        "[metadata]",
        'difficulty = "easy"',
        'category = "programming"',
        "",
        "[verifier]",
        "timeout_sec = 120.0",
        "",
        "[agent]",
        "timeout_sec = 120.0",
        "",
        "[environment]",
        "build_timeout_sec = 600.0",
        "cpus = 1",
        "memory_mb = 2048",
        "storage_mb = 10240",
    ]

    if with_mcp:
        main_environment = (
            '    environment:\n      - "HARBOR_HOST_TOKEN"\n      - "HARBOR_MODE=1"\n'
            if passthrough_env
            else ""
        )
        sidecar_environment = (
            '    environment:\n      - "MCP_TOKEN"\n      - "SIDECAR_MODE=enabled"\n'
            if passthrough_env
            else ""
        )
        sidecar_command = (
            '    command:\n      - "python3"\n      - "-m"\n      - "http.server"\n      - "8000"\n'
            if exec_form_command
            else ""
        )
        compose_text = (
            "services:\n"
            "  main:\n" + main_environment + "    depends_on:\n"
            "      mcp-server:\n"
            "        condition: service_healthy\n"
            "  mcp-server:\n"
            "    build:\n"
            "      context: ./mcp-server\n" + sidecar_environment + "    expose:\n"
            '      - "8000"\n'
            "    healthcheck:\n"
            '      test: ["CMD", "python3", "-c", "print(1)"]\n'
        )
        compose_text = compose_text.replace(
            "    expose:\n",
            f"{sidecar_command}    expose:\n",
            1,
        )
        (environment_dir / "docker-compose.yaml").write_text(compose_text, encoding="utf-8")
        (environment_dir / "mcp-server").mkdir(parents=True, exist_ok=True)
        (environment_dir / "mcp-server" / "Dockerfile").write_text(
            'FROM python:3.12-slim\nCMD ["python3", "-m", "http.server", "8000"]\n',
            encoding="utf-8",
        )
        task_toml.extend(
            [
                "",
                "[verifier.env]",
                'MODEL_NAME = "test-model"',
                "",
                "[solution.env]",
                'OPENAI_API_KEY = "${OPENAI_API_KEY}"',
                "",
                "[[environment.mcp_servers]]",
                'name = "mcp-server"',
                'transport = "streamable-http"',
                'url = "http://mcp-server:8000/mcp"',
            ]
        )

    (task_dir / "task.toml").write_text("\n".join(task_toml) + "\n", encoding="utf-8")
    return task_dir


def _load_trajectory_fixture(name: str) -> dict[str, object]:
    fixture_path = Path(__file__).parent / "fixtures" / "harbor_trajectory" / name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _make_success_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        version="0.1",
        task_id="hello-world",
        task="Create /app/hello.txt with the expected content.",
        model="test-model",
        provider="openai",
        created_at="2026-03-31T12:00:00Z",
        messages=[
            {
                "role": "user",
                "content": "Create /app/hello.txt with the expected content.",
                "timestamp": "2026-03-31T12:00:00Z",
            },
            {
                "role": "assistant",
                "content": {
                    "content": "I'll create the file.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "harbor-main__write_file",
                                "arguments": '{"path":"/app/hello.txt","content":"hello"}',
                            },
                        }
                    ],
                },
                "timestamp": "2026-03-31T12:00:01Z",
            },
            {
                "role": "tool",
                "content": {
                    "tool_call_id": "call_1",
                    "tool_server": "harbor-main",
                    "tool_name": "write_file",
                    "summary": "Created /app/hello.txt",
                    "is_error": False,
                },
                "timestamp": "2026-03-31T12:00:02Z",
            },
            {
                "role": "assistant",
                "content": "Done.",
                "timestamp": "2026-03-31T12:00:03Z",
            },
        ],
        tool_calls=[
            ToolCall(
                tool_server="harbor-main",
                tool_name="write_file",
                parameters={"path": "/app/hello.txt", "content": "hello"},
            )
        ],
        tool_results=[
            ToolCallResult(
                observation={"text": "Created /app/hello.txt", "path": "/app/hello.txt"},
                is_error=False,
            )
        ],
        metadata={
            "cli_runtime": {
                "agent_import_path": "custom.agent:FixtureAgent",
            },
            "rollout_metrics": {
                "token_usage": {
                    "prompt_tokens_total": 11,
                    "completion_tokens_total": 7,
                },
                "cost": {"estimated_cost_usd": 0.123},
            },
        },
        final_observation="Done.",
        error=None,
        steps_taken=1,
        max_steps=5,
        log_probs=[-0.1, -0.2],
    )


def _make_timeout_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        version="0.1",
        task_id="hello-world",
        task="Create /app/hello.txt with the expected content.",
        model="test-model",
        provider="openai",
        created_at="2026-03-31T12:10:00Z",
        messages=[
            {
                "role": "user",
                "content": "Create /app/hello.txt with the expected content.",
                "timestamp": "2026-03-31T12:10:00Z",
            }
        ],
        tool_calls=[],
        tool_results=[],
        metadata={
            "timeout": True,
            "cli_runtime": {
                "agent_import_path": "custom.agent:SlowAgent",
            },
        },
        final_observation=None,
        error="Rollout timeout exceeded",
        steps_taken=0,
        max_steps=5,
        log_probs=None,
    )


def test_parse_harbor_task_reads_instruction_and_mcp(tmp_path: Path) -> None:
    task_dir = _write_harbor_task(tmp_path, with_mcp=True)

    spec = parse_harbor_task(task_dir)

    assert spec.task_id == "hello-mcp"
    assert spec.instruction_text == "Create /app/hello.txt with the expected content."
    assert spec.workdir == "/app"
    assert spec.agent_timeout_seconds == 120.0
    assert spec.verifier_timeout_seconds == 120.0
    assert spec.solution_env == {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}
    assert spec.verifier_env == {"MODEL_NAME": "test-model"}
    assert len(spec.mcp_servers) == 1
    assert spec.mcp_servers[0].name == "mcp-server"
    depends_on = spec.main_service_overrides.get("depends_on")
    assert isinstance(depends_on, dict)
    mcp_server = depends_on.get("mcp-server")
    assert isinstance(mcp_server, dict)
    assert mcp_server["condition"] == "service_healthy"


def test_prepare_harbor_run_generates_env_and_bundle(tmp_path: Path) -> None:
    task_dir = _write_harbor_task(tmp_path, with_mcp=True, exec_form_command=True)

    prepared = prepare_harbor_run(task_dir, workspace_root=tmp_path / "generated")

    env_yaml = yaml.safe_load((prepared.env_dir / "env.yaml").read_text(encoding="utf-8"))
    assert "rollout_format" not in env_yaml
    assert env_yaml["tools"] == ["harbor-main"]
    assert (prepared.env_dir / "custom-tools" / "harbor-main.yaml").is_file()
    assert (prepared.env_dir / "mcp-servers.json").is_file()
    dockerfile = prepared.env_dir / "harbor-environment" / "Dockerfile.simlab-wrapper"
    assert dockerfile.is_file()
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    assert "/opt/openhands-venv/bin/python" in dockerfile_text
    assert "COPY .simlab/tests /tests" in dockerfile_text
    assert "/root/.local/bin/env" in dockerfile_text
    assert "command -v bash" in dockerfile_text
    assert "cp -a /app/. /opt/harbor-seed/workdir/" in dockerfile_text
    assert 'ENV OPENHANDS_SEED_DIR="/opt/harbor-seed/workdir"' in dockerfile_text
    entrypoint_text = (
        prepared.env_dir / "harbor-environment" / ".simlab" / "openhands-entrypoint.sh"
    ).read_text(encoding="utf-8")
    assert 'seed_dir="${OPENHANDS_SEED_DIR:-/opt/harbor-seed/workdir}"' in entrypoint_text
    assert 'cp -an "${seed_dir}/." "${workspace_dir}/"' in entrypoint_text
    assert (prepared.env_dir / "harbor-environment" / ".simlab" / "tests" / "test.sh").is_file()
    compose = yaml.safe_load((prepared.env_dir / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "harbor-openhands-agent-server" in compose["services"]
    assert "harbor-coding-env" in compose["services"]
    assert "mcp-server" in compose["services"]
    assert compose["services"]["mcp-server"]["command"] == ["python3", "-m", "http.server", "8000"]
    volumes = compose["services"]["harbor-openhands-agent-server"]["volumes"]
    assert "./harbor-tests:/tests:ro" not in volumes

    task_payload = json.loads(
        (prepared.bundle_dir / "tasks" / f"{prepared.task_id}.json").read_text(encoding="utf-8")
    )
    assert task_payload["task"] == "Create /app/hello.txt with the expected content."
    assert task_payload["tool_servers"][0]["name"] == "harbor-main"
    assert task_payload["harbor_verifier"]["service"] == "harbor-openhands-agent-server"
    assert task_payload["harbor_verifier"]["timeout_sec"] == 120.0


def test_build_atif_trajectory_matches_success_golden() -> None:
    trajectory = build_atif_trajectory(
        _make_success_artifacts(),
        run_id="run-success",
        verification_passed=True,
        reward=1.0,
        reward_payload={"reward": 1.0},
    )

    assert trajectory == _load_trajectory_fixture("success.json")


def test_build_atif_trajectory_matches_verifier_failure_golden() -> None:
    trajectory = build_atif_trajectory(
        _make_success_artifacts(),
        run_id="run-verifier-failure",
        verification_passed=False,
        reward=0.0,
        reward_payload={"reward": 0.0, "reason": "missing output"},
    )

    assert trajectory == _load_trajectory_fixture("verifier_failure.json")


def test_build_atif_trajectory_matches_timeout_golden() -> None:
    trajectory = build_atif_trajectory(
        _make_timeout_artifacts(),
        run_id="run-timeout",
        verification_passed=False,
        reward=0.0,
        reward_payload={"reward": 0.0, "reason": "agent timeout"},
    )

    assert trajectory == _load_trajectory_fixture("timeout.json")


def test_build_atif_trajectory_omits_unknown_task_fields() -> None:
    artifacts = _make_success_artifacts()
    artifacts.task_id = None
    artifacts.task = ""

    trajectory = build_atif_trajectory(
        artifacts,
        run_id="run-missing-task-fields",
        verification_passed=True,
        reward=1.0,
        reward_payload={"reward": 1.0},
    )

    simlab_extra = trajectory["extra"]["simlab"]
    assert "task_id" not in simlab_extra
    assert "task" not in simlab_extra


def test_build_atif_trajectory_preserves_non_rollout_timeout_errors() -> None:
    artifacts = _make_success_artifacts()
    artifacts.error = "Database connection timeout while calling MCP server"

    trajectory = build_atif_trajectory(
        artifacts,
        run_id="run-tool-timeout-error",
        verification_passed=None,
        reward=None,
        reward_payload=None,
    )

    status_step = trajectory["steps"][-1]
    assert status_step["message"] == artifacts.error
    assert status_step["extra"]["status"] == "error"


def test_prepare_harbor_run_preserves_list_environment_and_quotes_workdir(tmp_path: Path) -> None:
    task_dir = _write_harbor_task(
        tmp_path,
        with_mcp=True,
        passthrough_env=True,
        workdir="/app/work space",
    )

    prepared = prepare_harbor_run(task_dir, workspace_root=tmp_path / "generated")

    dockerfile_text = (
        prepared.env_dir / "harbor-environment" / "Dockerfile.simlab-wrapper"
    ).read_text(encoding="utf-8")
    assert 'ENV OPENHANDS_WORKSPACE_DIR="/app/work space"' in dockerfile_text
    assert 'WORKDIR "/app/work space"' in dockerfile_text

    compose = yaml.safe_load((prepared.env_dir / "docker-compose.yml").read_text(encoding="utf-8"))
    main_env = compose["services"]["harbor-openhands-agent-server"]["environment"]
    sidecar_env = compose["services"]["mcp-server"]["environment"]

    assert "HARBOR_HOST_TOKEN" in main_env
    assert "HARBOR_MODE=1" in main_env
    assert "OPENAI_API_KEY=${OPENAI_API_KEY}" in main_env
    assert "SESSION_API_KEY=dev-session-key" in main_env
    assert sidecar_env == ["MCP_TOKEN", "SIDECAR_MODE=enabled"]


def test_rewrite_mcp_config_for_runtime_local(tmp_path: Path) -> None:
    task_dir = _write_harbor_task(tmp_path, with_mcp=True)
    prepared = prepare_harbor_run(task_dir, workspace_root=tmp_path / "generated")
    compose = yaml.safe_load((prepared.env_dir / "docker-compose.yml").read_text(encoding="utf-8"))
    host_port = compose_service_host_port(compose, "mcp-server")

    rewritten = rewrite_mcp_config_for_runtime(
        rollout_runner.load_mcp_servers_from_env_dir(prepared.env_dir),
        env_dir=prepared.env_dir,
        using_daytona=False,
        daytona_client_factory=lambda **_kwargs: None,
    )

    assert host_port is not None
    assert rewritten == {
        "mcpServers": {
            "mcp-server": {
                "url": f"http://localhost:{host_port}/mcp",
            }
        }
    }


def test_rewrite_mcp_config_for_runtime_daytona(tmp_path: Path) -> None:
    task_dir = _write_harbor_task(tmp_path, with_mcp=True)
    prepared = prepare_harbor_run(task_dir, workspace_root=tmp_path / "generated")
    compose = yaml.safe_load((prepared.env_dir / "docker-compose.yml").read_text(encoding="utf-8"))
    host_port = compose_service_host_port(compose, "mcp-server")
    (prepared.env_dir / "daytona-state.json").write_text(
        json.dumps({"sandbox_id": "sbx-1"}),
        encoding="utf-8",
    )

    class FakeSandbox:
        def get_preview_link(self, port: int) -> SimpleNamespace:
            return SimpleNamespace(url=f"https://preview-{port}.example.com")

    class FakeDaytona:
        def get(self, sandbox_id: str) -> FakeSandbox:
            assert sandbox_id == "sbx-1"
            return FakeSandbox()

    rewritten = rewrite_mcp_config_for_runtime(
        rollout_runner.load_mcp_servers_from_env_dir(prepared.env_dir),
        env_dir=prepared.env_dir,
        using_daytona=True,
        daytona_client_factory=lambda **_kwargs: FakeDaytona(),
    )

    assert host_port is not None
    assert rewritten == {
        "mcpServers": {
            "mcp-server": {
                "url": f"https://preview-{host_port}.example.com/mcp",
            }
        }
    }


def test_tasks_run_harbor_uses_instruction_and_writes_reward_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    task_dir = _write_harbor_task(tmp_path, with_mcp=False)
    captured: dict[str, object] = {}

    class FakeArtifacts:
        def __init__(self) -> None:
            self.metadata: dict[str, object] = {}
            self.error: str | None = None
            self.messages: list[dict[str, object]] = [
                {
                    "role": "user",
                    "content": "Create /app/hello.txt with the expected content.",
                    "timestamp": "2026-03-31T12:00:00Z",
                },
                {
                    "role": "assistant",
                    "content": "Done.",
                    "timestamp": "2026-03-31T12:00:01Z",
                },
            ]
            self.steps_taken = 0
            self.tool_calls: list[object] = []
            self.tool_results: list[object] = []
            self.version = "0.1"
            self.task_id = "hello-world"
            self.task = "Create /app/hello.txt with the expected content."
            self.model = "test-model"
            self.provider = "openai"
            self.created_at = "2026-03-31T12:00:00Z"
            self.final_observation = "Done."
            self.log_probs = None

        def dump(self, path: Path) -> None:
            path.write_text("{}", encoding="utf-8")

    class FakeEnvironment:
        def __init__(
            self,
            *,
            tool_servers: dict[str, str],
            mcp_clients: object | None = None,
        ) -> None:
            captured["tool_servers"] = tool_servers
            captured["mcp_clients"] = mcp_clients

    def fake_run_with_agent_contract(**kwargs):
        captured["instruction"] = kwargs["instruction"]
        captured["timeout_seconds"] = kwargs["timeout_seconds"]
        return FakeArtifacts()

    monkeypatch.setattr(
        tasks_cli,
        "get_global_config_from_ctx",
        lambda _ctx: SimpleNamespace(daytona_api_key="daytona-key"),
    )
    monkeypatch.setattr(
        tasks_cli,
        "resolve_scenario_manager_api_url",
        lambda *args, **kwargs: "https://api.example.com",
    )
    monkeypatch.setattr(tasks_cli, "resolve_collinear_api_key", lambda *args, **kwargs: "ck-test")
    monkeypatch.setattr(
        tasks_cli,
        "_resolve_agent_runtime_settings",
        lambda *args, **kwargs: ("test-model", "openai", None, None),
    )
    monkeypatch.setattr(tasks_cli, "env_has_local_services", lambda _env_dir: False)
    monkeypatch.setattr(agents_loader, "load_agent_class", lambda _path: None)
    monkeypatch.setattr(
        rollout_runner,
        "resolve_endpoints",
        lambda **_kwargs: ({"harbor-main": "http://localhost:18020"}, False),
    )
    monkeypatch.setattr(rollout_runner, "require_reachable_endpoints", lambda **_kwargs: None)
    monkeypatch.setattr(rollout_runner, "require_mcp_tools_available", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rollout_runner,
        "get_agent_runtime_helpers",
        lambda: (FakeEnvironment, fake_run_with_agent_contract),
    )
    monkeypatch.setattr(
        rollout_runner.harbor_verifier_runtime,
        "run_harbor_verifier",
        lambda **_kwargs: (
            True,
            1.0,
            {
                "reward": 1.0,
                "checks": [
                    {
                        "check": "hello_file_exists",
                        "passed": True,
                        "description": "Found /app/hello.txt",
                        "weight": 1,
                        "points": 1,
                    }
                ],
                "verifier_results": [
                    {
                        "module": "harbor_test_sh",
                        "success": True,
                        "message": "Harbor test.sh passed",
                        "output": json.dumps(
                            {
                                "checks": [
                                    {
                                        "check": "hello_file_exists",
                                        "passed": True,
                                        "description": "Found /app/hello.txt",
                                        "weight": 1,
                                        "points": 1,
                                    }
                                ]
                            }
                        ),
                    }
                ],
                "harbor_test_sh": {
                    "service": "harbor-openhands-agent-server",
                    "workdir": "/workspace",
                    "script_path": "/tests/test.sh",
                    "exit_code": 0,
                    "output": "",
                },
            },
            "",
        ),
    )
    monkeypatch.setattr(tasks_cli, "emit_cli_event", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        tasks_cli.tasks,
        [
            "run",
            "--harbor",
            str(task_dir),
            "--agent-import-path",
            "custom.agent:Agent",
            "--agent-model",
            "test-model",
        ],
        env={"SIMLAB_DISABLE_TELEMETRY": "1"},
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["instruction"] == "Create /app/hello.txt with the expected content."
    assert captured["timeout_seconds"] == 120.0
    assert captured["tool_servers"] == {"harbor-main": "http://localhost:18020"}
    run_dirs = sorted((tmp_path / "output").glob("agent_run_*"))
    assert len(run_dirs) == 1
    reward_txt = run_dirs[0] / "verifier" / "reward.txt"
    reward_json = run_dirs[0] / "verifier" / "reward.json"
    trajectory_json = run_dirs[0] / "agent" / "trajectory.json"
    artifacts_json = run_dirs[0] / "artifacts.json"
    assert reward_txt.read_text(encoding="utf-8").strip() == "1"
    assert json.loads(reward_json.read_text(encoding="utf-8"))["reward"] == 1.0
    assert not artifacts_json.exists()
    trajectory = json.loads(trajectory_json.read_text(encoding="utf-8"))
    assert trajectory["schema_version"] == "ATIF-v1.4"
    assert trajectory["steps"][0]["source"] == "user"
    assert "1/1 passed" in result.output


def test_tasks_run_harbor_keep_alive_retains_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    task_dir = _write_harbor_task(tmp_path, with_mcp=False)

    class FakeArtifacts:
        def __init__(self) -> None:
            self.metadata: dict[str, object] = {}
            self.error: str | None = None
            self.messages: list[dict[str, object]] = []
            self.steps_taken = 0
            self.tool_calls: list[object] = []

        def dump(self, path: Path) -> None:
            path.write_text("{}", encoding="utf-8")

    class FakeEnvironment:
        def __init__(
            self,
            *,
            tool_servers: dict[str, str],
            mcp_clients: object | None = None,
        ) -> None:
            _ = tool_servers, mcp_clients

    monkeypatch.setattr(
        tasks_cli,
        "get_global_config_from_ctx",
        lambda _ctx: SimpleNamespace(daytona_api_key="daytona-key"),
    )
    monkeypatch.setattr(
        tasks_cli,
        "resolve_scenario_manager_api_url",
        lambda *args, **kwargs: "https://api.example.com",
    )
    monkeypatch.setattr(tasks_cli, "resolve_collinear_api_key", lambda *args, **kwargs: "ck-test")
    monkeypatch.setattr(
        tasks_cli,
        "_resolve_agent_runtime_settings",
        lambda *args, **kwargs: ("test-model", "openai", None, None),
    )
    monkeypatch.setattr(tasks_cli, "env_has_local_services", lambda _env_dir: False)
    monkeypatch.setattr(agents_loader, "load_agent_class", lambda _path: None)
    monkeypatch.setattr(
        rollout_runner,
        "resolve_endpoints",
        lambda **_kwargs: ({"harbor-main": "http://localhost:18020"}, False),
    )
    monkeypatch.setattr(rollout_runner, "require_reachable_endpoints", lambda **_kwargs: None)
    monkeypatch.setattr(rollout_runner, "require_mcp_tools_available", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rollout_runner,
        "get_agent_runtime_helpers",
        lambda: (FakeEnvironment, lambda **_kwargs: FakeArtifacts()),
    )
    monkeypatch.setattr(
        rollout_runner.harbor_verifier_runtime,
        "run_harbor_verifier",
        lambda **_kwargs: (
            True,
            1.0,
            {
                "reward": 1.0,
                "verifier_results": [
                    {
                        "module": "harbor_test_sh",
                        "success": True,
                        "message": "Harbor test.sh passed",
                        "output": json.dumps({"name": "harbor_test_sh", "passed": True}),
                    }
                ],
            },
            "",
        ),
    )
    monkeypatch.setattr(tasks_cli, "emit_cli_event", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        tasks_cli.tasks,
        [
            "run",
            "--harbor",
            str(task_dir),
            "--keep-alive",
            "--agent-import-path",
            "custom.agent:Agent",
            "--agent-model",
            "test-model",
        ],
        env={"SIMLAB_DISABLE_TELEMETRY": "1"},
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Harbor workspace retained:" in result.output
    retained_roots = list((tmp_path / "output" / "harbor_runs").glob("hello-world_*"))
    assert len(retained_roots) == 1
    assert (retained_roots[0] / "env" / "env.yaml").is_file()


def test_tasks_run_harbor_failed_verifier_still_prints_summary_card(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    task_dir = _write_harbor_task(tmp_path, with_mcp=False)

    class FakeArtifacts:
        def __init__(self) -> None:
            self.metadata: dict[str, object] = {}
            self.error: str | None = None
            self.messages: list[dict[str, object]] = []
            self.steps_taken = 0
            self.tool_calls: list[object] = []

        def dump(self, path: Path) -> None:
            path.write_text("{}", encoding="utf-8")

    class FakeEnvironment:
        def __init__(
            self,
            *,
            tool_servers: dict[str, str],
            mcp_clients: object | None = None,
        ) -> None:
            _ = tool_servers, mcp_clients

    monkeypatch.setattr(
        tasks_cli,
        "get_global_config_from_ctx",
        lambda _ctx: SimpleNamespace(daytona_api_key="daytona-key"),
    )
    monkeypatch.setattr(
        tasks_cli,
        "resolve_scenario_manager_api_url",
        lambda *args, **kwargs: "https://api.example.com",
    )
    monkeypatch.setattr(tasks_cli, "resolve_collinear_api_key", lambda *args, **kwargs: "ck-test")
    monkeypatch.setattr(
        tasks_cli,
        "_resolve_agent_runtime_settings",
        lambda *args, **kwargs: ("test-model", "openai", None, None),
    )
    monkeypatch.setattr(tasks_cli, "env_has_local_services", lambda _env_dir: False)
    monkeypatch.setattr(agents_loader, "load_agent_class", lambda _path: None)
    monkeypatch.setattr(
        rollout_runner,
        "resolve_endpoints",
        lambda **_kwargs: ({"harbor-main": "http://localhost:18020"}, False),
    )
    monkeypatch.setattr(rollout_runner, "require_reachable_endpoints", lambda **_kwargs: None)
    monkeypatch.setattr(rollout_runner, "require_mcp_tools_available", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rollout_runner,
        "get_agent_runtime_helpers",
        lambda: (FakeEnvironment, lambda **_kwargs: FakeArtifacts()),
    )
    monkeypatch.setattr(
        rollout_runner.harbor_verifier_runtime,
        "run_harbor_verifier",
        lambda **_kwargs: (
            False,
            0.0,
            {
                "reward": 0.0,
                "checks": [
                    {
                        "check": "hello_file_exists",
                        "passed": False,
                        "description": "Missing /app/hello.txt",
                        "weight": 1,
                        "points": 0,
                    }
                ],
                "verifier_results": [
                    {
                        "module": "harbor_test_sh",
                        "success": False,
                        "message": "Harbor test.sh failed (exit_code=1)",
                        "output": json.dumps(
                            {
                                "checks": [
                                    {
                                        "check": "hello_file_exists",
                                        "passed": False,
                                        "description": "Missing /app/hello.txt",
                                        "weight": 1,
                                        "points": 0,
                                    }
                                ]
                            }
                        ),
                    }
                ],
                "harbor_test_sh": {
                    "service": "harbor-openhands-agent-server",
                    "workdir": "/workspace",
                    "script_path": "/tests/test.sh",
                    "exit_code": 1,
                    "output": "file missing",
                },
            },
            "",
        ),
    )
    monkeypatch.setattr(tasks_cli, "emit_cli_event", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        tasks_cli.tasks,
        [
            "run",
            "--harbor",
            str(task_dir),
            "--agent-import-path",
            "custom.agent:Agent",
            "--agent-model",
            "test-model",
        ],
        env={"SIMLAB_DISABLE_TELEMETRY": "1"},
        catch_exceptions=False,
    )

    assert result.exit_code == 1, result.output
    assert "Verification failed." in result.stderr
    assert "Preserved Harbor workspace:" in result.stderr
    assert "Rollout Summary" in result.stderr
    assert "FAIL" in result.stderr
    assert "Next: simlab eval" in result.stderr


def test_tasks_run_harbor_cli_timeout_override_wins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    task_dir = _write_harbor_task(tmp_path, with_mcp=False)
    captured: dict[str, object] = {}

    class FakeArtifacts:
        def __init__(self) -> None:
            self.metadata: dict[str, object] = {}
            self.error: str | None = None
            self.messages: list[dict[str, object]] = []
            self.steps_taken = 0
            self.tool_calls: list[object] = []

        def dump(self, path: Path) -> None:
            path.write_text("{}", encoding="utf-8")

    class FakeEnvironment:
        def __init__(
            self,
            *,
            tool_servers: dict[str, str],
            mcp_clients: object | None = None,
        ) -> None:
            _ = tool_servers, mcp_clients

    monkeypatch.setattr(
        tasks_cli,
        "get_global_config_from_ctx",
        lambda _ctx: SimpleNamespace(daytona_api_key="daytona-key"),
    )
    monkeypatch.setattr(
        tasks_cli,
        "resolve_scenario_manager_api_url",
        lambda *args, **kwargs: "https://api.example.com",
    )
    monkeypatch.setattr(tasks_cli, "resolve_collinear_api_key", lambda *args, **kwargs: "ck-test")
    monkeypatch.setattr(
        tasks_cli,
        "_resolve_agent_runtime_settings",
        lambda *args, **kwargs: ("test-model", "openai", None, None),
    )
    monkeypatch.setattr(tasks_cli, "env_has_local_services", lambda _env_dir: False)
    monkeypatch.setattr(agents_loader, "load_agent_class", lambda _path: None)
    monkeypatch.setattr(
        rollout_runner,
        "resolve_endpoints",
        lambda **_kwargs: ({"harbor-main": "http://localhost:18020"}, False),
    )
    monkeypatch.setattr(rollout_runner, "require_reachable_endpoints", lambda **_kwargs: None)
    monkeypatch.setattr(rollout_runner, "require_mcp_tools_available", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rollout_runner,
        "get_agent_runtime_helpers",
        lambda: (
            FakeEnvironment,
            lambda **kwargs: (
                captured.update({"timeout_seconds": kwargs["timeout_seconds"]}) or FakeArtifacts()
            ),
        ),
    )
    monkeypatch.setattr(
        rollout_runner.harbor_verifier_runtime,
        "run_harbor_verifier",
        lambda **_kwargs: (
            True,
            1.0,
            {
                "reward": 1.0,
                "verifier_results": [
                    {
                        "module": "harbor_test_sh",
                        "success": True,
                        "message": "Harbor test.sh passed",
                        "output": json.dumps({"name": "harbor_test_sh", "passed": True}),
                    }
                ],
            },
            "",
        ),
    )
    monkeypatch.setattr(tasks_cli, "emit_cli_event", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        tasks_cli.tasks,
        [
            "run",
            "--harbor",
            str(task_dir),
            "--agent-import-path",
            "custom.agent:Agent",
            "--agent-model",
            "test-model",
            "--agent-timeout-seconds",
            "42",
        ],
        env={"SIMLAB_DISABLE_TELEMETRY": "1"},
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["timeout_seconds"] == 42.0


def test_run_harbor_verifier_uses_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_compose_exec(**kwargs):
        if kwargs["command"].startswith("mkdir -p /logs/verifier"):
            captured["timeout"] = kwargs["timeout"]
        return ComposeExecResult(exit_code=0, output="")

    monkeypatch.setattr(
        "simlab.runtime.adapters.harbor.verifier.run_compose_exec",
        fake_run_compose_exec,
    )
    monkeypatch.setattr(
        "simlab.runtime.adapters.harbor.verifier.read_compose_file",
        lambda **kwargs: "1" if kwargs["file_path"].endswith("reward.txt") else '{"reward": 1}',
    )

    passed, reward, payload, output = run_harbor_verifier(
        env_dir=tmp_path,
        verifier_config={
            "service": "harbor-openhands-agent-server",
            "workdir": "/workspace",
            "script_path": "/tests/test.sh",
            "timeout_sec": 321.0,
            "env": {},
        },
        using_daytona=False,
        daytona_client_factory=lambda **_kwargs: None,
    )

    assert passed is True
    assert reward == 1.0
    assert payload["reward"] == 1.0
    assert payload["verifier_results"][0]["module"] == "harbor_test_sh"
    assert output == ""
    assert captured["timeout"] == 321.0


def test_run_compose_exec_returns_clean_result_on_local_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_compose_exec(
        env_dir=tmp_path,
        service="harbor-openhands-agent-server",
        command="/tests/test.sh",
        env_overrides={},
        using_daytona=False,
        daytona_client_factory=lambda **_kwargs: None,
        timeout=12.0,
    )

    assert result == ComposeExecResult(
        exit_code=1,
        output="Command timed out after 12s",
    )


def test_run_compose_exec_returns_clean_result_on_daytona_timeout(tmp_path: Path) -> None:
    (tmp_path / "daytona-state.json").write_text(
        json.dumps({"sandbox_id": "sbx-1"}),
        encoding="utf-8",
    )

    class FakeTimeoutError(Exception):
        pass

    class FakeProcess:
        def exec(self, *_args, **_kwargs):
            raise FakeTimeoutError("execution timed out")

    class FakeSandbox:
        process = FakeProcess()

    class FakeDaytona:
        def get(self, sandbox_id: str) -> FakeSandbox:
            assert sandbox_id == "sbx-1"
            return FakeSandbox()

    result = run_compose_exec(
        env_dir=tmp_path,
        service="harbor-openhands-agent-server",
        command="/tests/test.sh",
        env_overrides={},
        using_daytona=True,
        daytona_client_factory=lambda **_kwargs: FakeDaytona(),
        timeout=34.2,
    )

    assert result == ComposeExecResult(
        exit_code=1,
        output="Command timed out after 35s",
    )


def test_run_compose_exec_coerces_daytona_timeout_to_int(tmp_path: Path) -> None:
    (tmp_path / "daytona-state.json").write_text(
        json.dumps({"sandbox_id": "sbx-1"}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeProcess:
        def exec(self, *_args, **kwargs):
            captured["timeout"] = kwargs["timeout"]
            return SimpleNamespace(exit_code=0, result="")

    class FakeSandbox:
        process = FakeProcess()

    class FakeDaytona:
        def get(self, sandbox_id: str) -> FakeSandbox:
            assert sandbox_id == "sbx-1"
            return FakeSandbox()

    result = run_compose_exec(
        env_dir=tmp_path,
        service="harbor-openhands-agent-server",
        command="/tests/test.sh",
        env_overrides={},
        using_daytona=True,
        daytona_client_factory=lambda **_kwargs: FakeDaytona(),
        timeout=34.2,
    )

    assert result == ComposeExecResult(exit_code=0, output="")
    assert captured["timeout"] == 35
    assert isinstance(captured["timeout"], int)
