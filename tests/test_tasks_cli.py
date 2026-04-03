from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import simlab.cli.tasks as tasks_module
from click.testing import CliRunner
from simlab.agents.base import RunArtifacts
from simlab.api.schemas import ScenarioTask
from simlab.api.schemas import ScenarioTasksResponse
from simlab.cli.tasks import tasks
from simlab.composer.engine import EnvConfig
from simlab.verifiers import VerifierResult


def _write_env_config(env_root: Path, env_name: str = "my-env") -> Path:
    """Create environments/<env_name>/env.yaml with template. Returns path to env.yaml."""
    env_dir = env_root / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    config_path = env_dir / "env.yaml"
    config_path.write_text("name: " + env_name + "\ntemplate: customer_support\ntools: [email]\n")
    return config_path


def _write_local_bundle(
    bundle_dir: Path,
    *,
    task_id: str = "generated-task",
    include_verifier: bool = False,
) -> None:
    (bundle_dir / "tasks").mkdir(parents=True, exist_ok=True)
    task = {
        "meta": {
            "task_id": task_id,
            "display_name": "Generated Task",
            "difficulty": "medium",
            "category": "workflow",
        },
        "task": "Complete the generated task.",
        "apps": ["email"],
        "tool_servers": [
            {
                "name": "email-env",
                "tool_server_url": "http://localhost:8040",
            }
        ],
        "seed_emails": [],
        "seed_calendar_events": [],
        "npcs": [],
        "verifiers": [],
    }
    if include_verifier:
        task["verifiers"] = [
            {
                "func": "python_module",
                "module": "collinear.scenarios.customer_support.verifiers.generated_task",
            }
        ]
        (bundle_dir / "verifiers").mkdir(parents=True, exist_ok=True)
        (bundle_dir / "verifiers" / "generated_task.py").write_text(
            "def verify(run_artifacts):\n    return True\n"
        )
    (bundle_dir / "tasks" / f"{task_id}.json").write_text(json.dumps(task), encoding="utf-8")


def test_tasks_list_include_test_flag_passes_through(tmp_path: Path) -> None:
    _write_env_config(tmp_path)
    runner = CliRunner()
    with (
        patch("simlab.cli.tasks.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.tasks.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.resolve_template_to_backend_id.return_value = "customer_support"
        mocked_client.list_scenario_tasks.return_value = ScenarioTasksResponse(
            scenario_id="customer_support", tasks=[]
        )
        result = runner.invoke(
            tasks,
            ["list", "--env", "my-env", "--include-test"],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )
    assert result.exit_code == 0, result.output
    mocked_client.list_scenario_tasks.assert_called_once_with(
        "customer_support",
        include_hidden=True,
        include_test=True,
    )


def test_tasks_info_include_test_flag_passes_through(tmp_path: Path) -> None:
    _write_env_config(tmp_path)
    runner = CliRunner()
    fake_task = ScenarioTask(
        task_id="test-calendar-integration",
        name="Test Calendar",
        description="desc",
        difficulty="easy",
        category="test",
        tool_servers=[],
    )
    with (
        patch("simlab.cli.tasks.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.tasks.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.resolve_template_to_backend_id.return_value = "customer_support"
        mocked_client.list_scenario_tasks.return_value = ScenarioTasksResponse(
            scenario_id="customer_support",
            tasks=[fake_task],
        )
        result = runner.invoke(
            tasks,
            [
                "info",
                "--env",
                "my-env",
                "--task",
                "test-calendar-integration",
                "--include-test",
            ],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )
    assert result.exit_code == 0, result.output
    mocked_client.list_scenario_tasks.assert_called_once_with(
        "customer_support",
        include_hidden=True,
        include_test=True,
    )


def test_tasks_info_known_test_task_id_resolves_without_include_test_flag(tmp_path: Path) -> None:
    _write_env_config(tmp_path)
    runner = CliRunner()
    fake_task = ScenarioTask(
        task_id="test-calendar-integration",
        name="Test Calendar",
        description="desc",
        difficulty="easy",
        category="test",
        tool_servers=[],
    )
    with (
        patch("simlab.cli.tasks.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.tasks.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.resolve_template_to_backend_id.return_value = "customer_support"
        mocked_client.list_scenario_tasks.side_effect = [
            ScenarioTasksResponse(scenario_id="customer_support", tasks=[]),
            ScenarioTasksResponse(scenario_id="customer_support", tasks=[fake_task]),
        ]
        result = runner.invoke(
            tasks,
            [
                "info",
                "--env",
                "my-env",
                "--task",
                "test-calendar-integration",
            ],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )
    assert result.exit_code == 0, result.output
    assert mocked_client.list_scenario_tasks.call_count == 2
    first_call, second_call = mocked_client.list_scenario_tasks.call_args_list
    assert first_call.kwargs == {"include_hidden": True, "include_test": False}
    assert second_call.kwargs == {"include_hidden": True, "include_test": True}


def test_tasks_list_supports_local_bundle_without_config(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "generated-tasks"
    _write_local_bundle(bundle_dir)
    runner = CliRunner()

    result = runner.invoke(
        tasks,
        [
            "list",
            "--tasks-dir",
            str(bundle_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "generated-task" in result.output
    assert "Generated Task" in result.output


def test_load_skills_markdown_uses_inline_config_guidance_even_with_bundle_dir(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "generated-tasks"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    config = EnvConfig(
        tools=["coding"],
        scenario_guidance_md="# Inline Guidance\nUse the inline guidance.\n",
    )

    result = tasks_module._load_skills_markdown(config=config, bundle_dir=bundle_dir)

    assert result == "# Inline Guidance\nUse the inline guidance."


def test_tasks_list_fails_fast_on_invalid_local_task_json(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "generated-tasks"
    tasks_dir = bundle_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / "broken.json").write_text("{invalid json\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        tasks,
        [
            "list",
            "--tasks-dir",
            str(bundle_dir),
        ],
    )

    assert result.exit_code == 1
    assert "Invalid JSON in" in result.output
    assert "broken.json" in result.output


def test_load_local_task_keeps_payload_and_file_aligned_for_partial_matches(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "generated-tasks"
    tasks_dir = bundle_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    partial_match_task = {
        "meta": {
            "task_id": "foo-extra",
            "display_name": "Partial Match Task",
            "difficulty": "medium",
            "category": "workflow",
        },
        "task": "Handle the partial match task.",
        "apps": ["email"],
        "tool_servers": [],
        "seed_emails": [],
        "seed_calendar_events": [],
        "npcs": [],
        "verifiers": [],
    }
    exact_match_task = {
        "meta": {
            "task_id": "foo",
            "display_name": "Exact Match Task",
            "difficulty": "medium",
            "category": "workflow",
        },
        "task": "Handle the exact match task.",
        "apps": ["email"],
        "tool_servers": [],
        "seed_emails": [],
        "seed_calendar_events": [],
        "npcs": [],
        "verifiers": [],
    }

    (tasks_dir / "a-task.json").write_text(json.dumps(partial_match_task), encoding="utf-8")
    (tasks_dir / "z-task.json").write_text(json.dumps(exact_match_task), encoding="utf-8")

    task_dict, _, task_file = tasks_module._load_local_task(bundle_dir, "foo")

    assert task_dict["meta"]["task_id"] == "foo"
    assert task_file == tasks_dir / "z-task.json"


def test_tasks_run_supports_local_bundle_without_template(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "generated-tasks"
    _write_local_bundle(bundle_dir, include_verifier=True)
    env_name = "local-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: local-env\n"
        "template: customer_support\n"
        "tools: [email]\n"
        "scenario_guidance_md: |\n"
        "  # Bundle Guidance\n"
        "  Always look for specialized skills first.\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    with (
        patch(
            "simlab.cli.tasks.get_global_config_from_ctx",
            return_value=SimpleNamespace(
                scenario_manager_api_url="https://api.example.com",
                api_key="api-token",
                daytona_api_key=None,
                agent_model=None,
                agent_provider=None,
                agent_api_key=None,
                agent_base_url=None,
                verifier_model=None,
                verifier_provider=None,
                verifier_base_url=None,
                verifier_api_key=None,
            ),
        ),
        patch(
            "simlab.cli.tasks.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        patch("simlab.cli.tasks.ensure_env_artifacts_current"),
        patch("simlab.cli.tasks.ScenarioManagerClient") as mocked_client_cls,
        patch(
            "simlab.cli.tasks._resolve_endpoints",
            return_value=({"email": "http://localhost:8040"}, False),
        ),
        patch("simlab.cli.tasks._require_reachable_endpoints"),
        patch("simlab.cli.tasks._provision_task_calendar_users"),
        patch("simlab.cli.tasks._ensure_task_calendar_accounts"),
        patch("simlab.cli.tasks._build_services_available_section", return_value=""),
        patch(
            "simlab.agents.run_with_agent_contract",
            return_value=RunArtifacts(
                task_id="generated-task",
                task="Complete the generated task.",
                model="gpt-5.2",
                provider="openai",
                max_steps=5,
            ),
        ) as mocked_run_with_agent_contract,
        patch(
            "simlab.verifiers.build_verifier_artifacts",
            return_value=SimpleNamespace(),
        ),
        patch("simlab.verifiers.run_verifier") as mocked_run_verifier,
    ):
        mocked_run_verifier.return_value = VerifierResult(success=True, message="ok")
        result = runner.invoke(
            tasks,
            [
                "run",
                "--env",
                env_name,
                "--tasks-dir",
                str(bundle_dir),
                "--task",
                "generated-task",
                "--agent-model",
                "gpt-5.2",
                "--agent-api-key",
                "openai-key",
            ],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    mocked_client_cls.assert_not_called()
    _, run_kwargs = mocked_run_with_agent_contract.call_args
    assert "Scenario guidance:" in run_kwargs["instruction"]
    assert "Always look for specialized skills first." in run_kwargs["instruction"]
    mocked_run_verifier.assert_called_once()
    _, kwargs = mocked_run_verifier.call_args
    local_verifier_path = kwargs.get("local_verifier_path")
    assert isinstance(local_verifier_path, Path)
    assert local_verifier_path == bundle_dir / "verifiers" / "generated_task.py"


def test_tasks_run_custom_agent_does_not_require_reference_agent_credentials(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "generated-tasks"
    _write_local_bundle(bundle_dir)
    env_name = "local-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: local-env\ntemplate: customer_support\ntools: [email]\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    with (
        patch(
            "simlab.cli.tasks.get_global_config_from_ctx",
            return_value=SimpleNamespace(
                scenario_manager_api_url="https://api.example.com",
                api_key="api-token",
                daytona_api_key=None,
                agent_model=None,
                agent_provider=None,
                agent_api_key=None,
                agent_base_url=None,
                verifier_model=None,
                verifier_provider=None,
                verifier_base_url=None,
                verifier_api_key=None,
            ),
        ),
        patch(
            "simlab.cli.tasks.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        patch("simlab.cli.tasks.ensure_env_artifacts_current"),
        patch(
            "simlab.cli.tasks._resolve_endpoints",
            return_value=({"email": "http://localhost:8040"}, False),
        ),
        patch("simlab.cli.tasks._require_reachable_endpoints"),
        patch("simlab.cli.tasks._provision_task_calendar_users"),
        patch("simlab.cli.tasks._ensure_task_calendar_accounts"),
        patch("simlab.cli.tasks._build_services_available_section", return_value=""),
        patch("simlab.cli.tasks.load_mcp_servers_from_env_dir", return_value={"mcpServers": {}}),
        patch("simlab.cli.tasks._build_mcp_clients", return_value={}),
        patch(
            "simlab.agents.run_with_agent_contract",
            return_value=RunArtifacts(
                task_id="generated-task",
                task="Complete the generated task.",
                model="custom-agent",
                provider="custom-agent",
                max_steps=5,
            ),
        ) as mocked_run_with_agent_contract,
        patch(
            "simlab.verifiers.build_verifier_artifacts",
            return_value=SimpleNamespace(),
        ),
        patch("simlab.verifiers.run_verifier") as mocked_run_verifier,
    ):
        mocked_run_verifier.return_value = VerifierResult(success=True, message="ok")
        result = runner.invoke(
            tasks,
            [
                "run",
                "--env",
                env_name,
                "--tasks-dir",
                str(bundle_dir),
                "--task",
                "generated-task",
                "--agent-import-path",
                "customer.agent:EmailAgent",
            ],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    _, run_kwargs = mocked_run_with_agent_contract.call_args
    assert run_kwargs["model"] == "custom-agent"
    assert run_kwargs["provider"] == "custom-agent"


def test_tasks_run_custom_agent_ignores_global_reference_agent_metadata(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "generated-tasks"
    _write_local_bundle(bundle_dir)
    env_name = "local-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: local-env\ntemplate: customer_support\ntools: [email]\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    with (
        patch(
            "simlab.cli.tasks.get_global_config_from_ctx",
            return_value=SimpleNamespace(
                scenario_manager_api_url="https://api.example.com",
                api_key="api-token",
                daytona_api_key=None,
                agent_model="claude-3-5-sonnet",
                agent_provider="anthropic",
                agent_api_key=None,
                agent_base_url=None,
                verifier_model=None,
                verifier_provider=None,
                verifier_base_url=None,
                verifier_api_key=None,
            ),
        ),
        patch(
            "simlab.cli.tasks.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        patch("simlab.cli.tasks.ensure_env_artifacts_current"),
        patch(
            "simlab.cli.tasks._resolve_endpoints",
            return_value=({"email": "http://localhost:8040"}, False),
        ),
        patch("simlab.cli.tasks._require_reachable_endpoints"),
        patch("simlab.cli.tasks._provision_task_calendar_users"),
        patch("simlab.cli.tasks._ensure_task_calendar_accounts"),
        patch("simlab.cli.tasks._build_services_available_section", return_value=""),
        patch("simlab.cli.tasks.load_mcp_servers_from_env_dir", return_value={"mcpServers": {}}),
        patch("simlab.cli.tasks._build_mcp_clients", return_value={}),
        patch(
            "simlab.agents.run_with_agent_contract",
            return_value=RunArtifacts(
                task_id="generated-task",
                task="Complete the generated task.",
                model="custom-agent",
                provider="custom-agent",
                max_steps=5,
            ),
        ) as mocked_run_with_agent_contract,
        patch(
            "simlab.verifiers.build_verifier_artifacts",
            return_value=SimpleNamespace(),
        ),
        patch("simlab.verifiers.run_verifier") as mocked_run_verifier,
    ):
        mocked_run_verifier.return_value = VerifierResult(success=True, message="ok")
        result = runner.invoke(
            tasks,
            [
                "run",
                "--env",
                env_name,
                "--tasks-dir",
                str(bundle_dir),
                "--task",
                "generated-task",
                "--agent-import-path",
                "customer.agent:EmailAgent",
            ],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    _, run_kwargs = mocked_run_with_agent_contract.call_args
    assert run_kwargs["model"] == "custom-agent"
    assert run_kwargs["provider"] == "custom-agent"


def test_tasks_validate_emits_telemetry(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "generated-tasks"
    _write_local_bundle(bundle_dir)
    runner = CliRunner()

    with patch("simlab.cli.tasks.emit_cli_event") as mocked_emit:
        result = runner.invoke(tasks, ["validate", "--tasks-dir", str(bundle_dir)])

    assert result.exit_code == 0, result.output
    mocked_emit.assert_called_once_with(
        "tasks_validate_completed",
        {
            "task_count": 1,
            "warning_count": 0,
        },
    )


def test_load_skills_markdown_prefers_inline_config_guidance() -> None:
    config = EnvConfig(
        name="local-env",
        tools=["email"],
        scenario_guidance_md="# Inline Guidance\nUse the environment carefully.\n",
    )

    loaded = tasks_module._load_skills_markdown(
        config=config,
        bundle_dir=None,
    )

    assert loaded == "# Inline Guidance\nUse the environment carefully."


def test_tasks_run_passes_mcp_verifier_tool_urls_when_http_tool_servers_are_absent(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "generated-tasks"
    _write_local_bundle(bundle_dir, include_verifier=True)
    task_path = bundle_dir / "tasks" / "generated-task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["tool_servers"] = []
    task_path.write_text(json.dumps(task), encoding="utf-8")

    env_name = "local-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: local-env\ntemplate: customer_support\ntools: []\n",
        encoding="utf-8",
    )

    fake_mcp_client = SimpleNamespace(_url="http://localhost:8091/mcp")
    runner = CliRunner()

    with (
        patch(
            "simlab.cli.tasks.get_global_config_from_ctx",
            return_value=SimpleNamespace(
                scenario_manager_api_url="https://api.example.com",
                api_key="api-token",
                daytona_api_key=None,
                agent_model=None,
                agent_provider=None,
                agent_api_key=None,
                agent_base_url=None,
                verifier_model=None,
                verifier_provider=None,
                verifier_base_url=None,
                verifier_api_key=None,
            ),
        ),
        patch(
            "simlab.cli.tasks.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        patch("simlab.cli.tasks.ensure_env_artifacts_current"),
        patch("simlab.cli.tasks._resolve_endpoints", return_value=({}, False)),
        patch("simlab.cli.tasks._provision_task_calendar_users"),
        patch("simlab.cli.tasks._ensure_task_calendar_accounts"),
        patch("simlab.cli.tasks._build_services_available_section", return_value=""),
        patch(
            "simlab.cli.tasks.load_mcp_servers_from_env_dir",
            return_value={"mcpServers": {"demo": {"url": "http://localhost:8091/mcp"}}},
        ),
        patch("simlab.cli.tasks._build_mcp_clients", return_value={"demo": fake_mcp_client}),
        patch("simlab.cli.tasks._require_mcp_tools_available"),
        patch(
            "simlab.agents.run_with_agent_contract",
            return_value=RunArtifacts(
                task_id="generated-task",
                task="Complete the generated task.",
                model="gpt-5.2",
                provider="openai",
                max_steps=5,
            ),
        ),
        patch(
            "simlab.verifiers.build_verifier_artifacts",
            return_value=SimpleNamespace(),
        ) as mocked_build_verifier_artifacts,
        patch(
            "simlab.verifiers.run_verifier",
            return_value=VerifierResult(success=True, message="ok"),
        ),
    ):
        result = runner.invoke(
            tasks,
            [
                "run",
                "--env",
                env_name,
                "--tasks-dir",
                str(bundle_dir),
                "--task",
                "generated-task",
                "--agent-model",
                "gpt-5.2",
                "--agent-api-key",
                "openai-key",
            ],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    _, _, verifier_tool_servers = mocked_build_verifier_artifacts.call_args.args
    assert verifier_tool_servers == {"demo": "http://localhost:8091"}


def test_tasks_run_writes_atif_when_rollout_format_flag_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    bundle_dir = tmp_path / "generated-tasks"
    _write_local_bundle(bundle_dir)
    env_name = "local-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: local-env\ntemplate: customer_support\ntools: [email]\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    with (
        patch(
            "simlab.cli.tasks.get_global_config_from_ctx",
            return_value=SimpleNamespace(
                scenario_manager_api_url="https://api.example.com",
                api_key="api-token",
                daytona_api_key=None,
                agent_model=None,
                agent_provider=None,
                agent_api_key=None,
                agent_base_url=None,
                verifier_model=None,
                verifier_provider=None,
                verifier_base_url=None,
                verifier_api_key=None,
            ),
        ),
        patch(
            "simlab.cli.tasks.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        patch("simlab.cli.tasks.ensure_env_artifacts_current"),
        patch(
            "simlab.cli.tasks._resolve_endpoints",
            return_value=({"email": "http://localhost:8040"}, False),
        ),
        patch("simlab.cli.tasks._require_reachable_endpoints"),
        patch("simlab.cli.tasks._provision_task_calendar_users"),
        patch("simlab.cli.tasks._ensure_task_calendar_accounts"),
        patch("simlab.cli.tasks._build_services_available_section", return_value=""),
        patch("simlab.cli.tasks.load_mcp_servers_from_env_dir", return_value={"mcpServers": {}}),
        patch("simlab.cli.tasks._build_mcp_clients", return_value={}),
        patch(
            "simlab.agents.run_with_agent_contract",
            return_value=RunArtifacts(
                task_id="generated-task",
                task="Complete the generated task.",
                model="gpt-5.2",
                provider="openai",
                max_steps=5,
            ),
        ),
    ):
        result = runner.invoke(
            tasks,
            [
                "run",
                "--env",
                env_name,
                "--tasks-dir",
                str(bundle_dir),
                "--task",
                "generated-task",
                "--agent-model",
                "gpt-5.2",
                "--agent-api-key",
                "openai-key",
                "--tasks-rollout-format",
                "atif",
            ],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    run_dirs = sorted((tmp_path / "output").glob("agent_run_*"))
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "artifacts.json").exists()
    trajectory = json.loads((run_dirs[0] / "agent" / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["schema_version"] == "ATIF-v1.4"
    assert trajectory["agent"]["model_name"] == "gpt-5.2"


def test_tasks_run_writes_atif_when_env_rollout_format_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    bundle_dir = tmp_path / "generated-tasks"
    _write_local_bundle(bundle_dir)
    env_name = "local-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: local-env\ntemplate: customer_support\ntools: [email]\nrollout_format: atif\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    with (
        patch(
            "simlab.cli.tasks.get_global_config_from_ctx",
            return_value=SimpleNamespace(
                scenario_manager_api_url="https://api.example.com",
                api_key="api-token",
                daytona_api_key=None,
                agent_model=None,
                agent_provider=None,
                agent_api_key=None,
                agent_base_url=None,
                verifier_model=None,
                verifier_provider=None,
                verifier_base_url=None,
                verifier_api_key=None,
            ),
        ),
        patch(
            "simlab.cli.tasks.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        patch("simlab.cli.tasks.ensure_env_artifacts_current"),
        patch(
            "simlab.cli.tasks._resolve_endpoints",
            return_value=({"email": "http://localhost:8040"}, False),
        ),
        patch("simlab.cli.tasks._require_reachable_endpoints"),
        patch("simlab.cli.tasks._provision_task_calendar_users"),
        patch("simlab.cli.tasks._ensure_task_calendar_accounts"),
        patch("simlab.cli.tasks._build_services_available_section", return_value=""),
        patch("simlab.cli.tasks.load_mcp_servers_from_env_dir", return_value={"mcpServers": {}}),
        patch("simlab.cli.tasks._build_mcp_clients", return_value={}),
        patch(
            "simlab.agents.run_with_agent_contract",
            return_value=RunArtifacts(
                task_id="generated-task",
                task="Complete the generated task.",
                model="gpt-5.2",
                provider="openai",
                max_steps=5,
            ),
        ),
    ):
        result = runner.invoke(
            tasks,
            [
                "run",
                "--env",
                env_name,
                "--tasks-dir",
                str(bundle_dir),
                "--task",
                "generated-task",
                "--agent-model",
                "gpt-5.2",
                "--agent-api-key",
                "openai-key",
            ],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    run_dirs = sorted((tmp_path / "output").glob("agent_run_*"))
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "artifacts.json").exists()
    trajectory = json.loads((run_dirs[0] / "agent" / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["schema_version"] == "ATIF-v1.4"
    assert trajectory["agent"]["model_name"] == "gpt-5.2"


def test_tasks_run_writes_atif_when_global_rollout_format_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    bundle_dir = tmp_path / "generated-tasks"
    _write_local_bundle(bundle_dir)
    env_name = "local-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: local-env\ntemplate: customer_support\ntools: [email]\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    with (
        patch(
            "simlab.cli.tasks.get_global_config_from_ctx",
            return_value=SimpleNamespace(
                scenario_manager_api_url="https://api.example.com",
                api_key="api-token",
                daytona_api_key=None,
                agent_model=None,
                agent_provider=None,
                agent_api_key=None,
                agent_base_url=None,
                verifier_model=None,
                verifier_provider=None,
                verifier_base_url=None,
                verifier_api_key=None,
                tasks_rollout_format="atif",
            ),
        ),
        patch(
            "simlab.cli.tasks.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        patch("simlab.cli.tasks.ensure_env_artifacts_current"),
        patch(
            "simlab.cli.tasks._resolve_endpoints",
            return_value=({"email": "http://localhost:8040"}, False),
        ),
        patch("simlab.cli.tasks._require_reachable_endpoints"),
        patch("simlab.cli.tasks._provision_task_calendar_users"),
        patch("simlab.cli.tasks._ensure_task_calendar_accounts"),
        patch("simlab.cli.tasks._build_services_available_section", return_value=""),
        patch("simlab.cli.tasks.load_mcp_servers_from_env_dir", return_value={"mcpServers": {}}),
        patch("simlab.cli.tasks._build_mcp_clients", return_value={}),
        patch(
            "simlab.agents.run_with_agent_contract",
            return_value=RunArtifacts(
                task_id="generated-task",
                task="Complete the generated task.",
                model="gpt-5.2",
                provider="openai",
                max_steps=5,
            ),
        ),
    ):
        result = runner.invoke(
            tasks,
            [
                "run",
                "--env",
                env_name,
                "--tasks-dir",
                str(bundle_dir),
                "--task",
                "generated-task",
                "--agent-model",
                "gpt-5.2",
                "--agent-api-key",
                "openai-key",
            ],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    run_dirs = sorted((tmp_path / "output").glob("agent_run_*"))
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "artifacts.json").exists()
    trajectory = json.loads((run_dirs[0] / "agent" / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["schema_version"] == "ATIF-v1.4"
    assert trajectory["agent"]["model_name"] == "gpt-5.2"
