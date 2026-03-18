from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import simlab.cli.env as env_module
import yaml
from click.testing import CliRunner
from simlab.api.client import ScenarioManagerApiError
from simlab.api.schemas import ScenarioSummary
from simlab.api.schemas import ScenarioToolServer
from simlab.cli.env import env
from simlab.composer.engine import DEFAULT_IMAGE_REGISTRY
from simlab.composer.engine import CodingConfig
from simlab.composer.engine import EnvConfig


class _FakeRegistry:
    def __init__(self, supported_tools: set[str]) -> None:
        self._supported_tools = supported_tools

    def get_tool(self, name: str) -> object | None:
        return object() if name in self._supported_tools else None

    def get_tools(self, names: list[str]) -> list[object]:
        """Return minimal tool-like objects (is_external=True) for engine."""
        out: list[object] = []
        for name in names:
            if name not in self._supported_tools:
                raise KeyError(f"Unknown tool: {name}")
            t = type(
                "_FakeTool",
                (),
                {"name": name, "is_external": True, "tool_server_url": "http://external.example"},
            )()
            out.append(t)
        return out


def test_env_init_template_uses_server_scenario_tools(tmp_path: Path) -> None:
    env_name = "my-env"
    env_dir = tmp_path / "environments" / env_name  # set SIMLAB_ENVIRONMENTS_DIR in invoke
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="financial_services",
            name="Financial Services",
            tool_servers=[
                ScenarioToolServer(name="spreadsheets"),
                ScenarioToolServer(name="missing-tool"),
            ],
            scenario_guidance_md="# Scenario Guidance\nFollow the domain conventions.\n",
        )
    ]

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"spreadsheets"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "financial_services"
        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--template",
                "financial_services",
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert "Starting from template 'financial_services': spreadsheets" in result.output
    assert "Ignoring unsupported tool servers from template 'financial_services': missing-tool" in (
        result.output
    )

    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text())
    assert data["registry"] == DEFAULT_IMAGE_REGISTRY
    assert data["template"] == "financial_services"
    assert data["tools"] == ["spreadsheets"]
    assert data["name"] == env_name
    assert data["scenario_guidance_md"] == "# Scenario Guidance\nFollow the domain conventions.\n"


def test_env_init_scenario_guidance_file_overrides_template_guidance(tmp_path: Path) -> None:
    env_name = "guided-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    guidance_file = tmp_path / "guidance.md"
    guidance_file.write_text("# Local Guidance\nPrefer specialist workflows.\n", encoding="utf-8")
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="financial_services",
            name="Financial Services",
            tool_servers=[ScenarioToolServer(name="spreadsheets")],
            scenario_guidance_md="# Server Guidance\nFollow the server defaults.\n",
        )
    ]

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"spreadsheets"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "financial_services"
        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--template",
                "financial_services",
                "--scenario-guidance-file",
                str(guidance_file),
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    data = yaml.safe_load(out_file.read_text())
    assert data["scenario_guidance_md"] == "# Local Guidance\nPrefer specialist workflows."


def test_env_init_scenario_guidance_file_missing_fails(tmp_path: Path) -> None:
    env_name = "bad-guidance-env"
    runner = CliRunner()

    with patch("simlab.cli.env._get_registry", return_value=_FakeRegistry(set())):
        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--scenario-guidance-file",
                str(tmp_path / "missing.md"),
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 1, result.output
    assert "Scenario guidance file does not exist:" in result.output


def test_env_init_template_maps_service_names_to_registry_tools(tmp_path: Path) -> None:
    env_name = "hr-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="human_resource",
            name="Human Resource",
            tool_servers=[
                ScenarioToolServer(name="frappe-hrms-env"),
                ScenarioToolServer(name="email-env"),
            ],
        )
    ]

    with (
        patch(
            "simlab.cli.env._get_registry",
            return_value=_FakeRegistry({"frappe-hrms", "email"}),
        ),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "human_resource"
        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--template",
                "human_resource",
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert "Starting from template 'human_resource': frappe-hrms, email" in result.output

    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text())
    assert data["registry"] == DEFAULT_IMAGE_REGISTRY
    assert data["tools"] == ["frappe-hrms", "email"]


def test_env_init_template_maps_google_workspace_service_name(tmp_path: Path) -> None:
    env_name = "gws-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="financial_services",
            name="Financial Services",
            tool_servers=[ScenarioToolServer(name="google-workspace-tool-server")],
        )
    ]

    with (
        patch(
            "simlab.cli.env._get_registry",
            return_value=_FakeRegistry({"google-workspace"}),
        ),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "financial_services"
        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--template",
                "financial_services",
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text())
    assert data["registry"] == DEFAULT_IMAGE_REGISTRY
    assert data["tools"] == ["google-workspace"]


def test_env_init_uses_default_registry_when_flag_omitted(tmp_path: Path) -> None:
    env_name = "reg-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="financial_services",
            name="Financial Services",
            tool_servers=[ScenarioToolServer(name="spreadsheets")],
        )
    ]

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"spreadsheets"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "financial_services"
        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--template",
                "financial_services",
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text())
    assert data["registry"] == DEFAULT_IMAGE_REGISTRY


def test_env_init_template_unknown_from_server_fails(tmp_path: Path) -> None:
    env_name = "bad-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    runner = CliRunner()

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"spreadsheets"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = []
        mocked_client.resolve_template_to_backend_id.side_effect = ScenarioManagerApiError(
            0, "Template 'human_resources' not found."
        )
        result = runner.invoke(
            env,
            ["init", env_name, "--template", "human_resources", "--non-interactive"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 1, result.output
    assert "Template 'human_resources' not found." in result.output
    assert not out_file.exists()


def test_env_init_template_uses_global_config_for_api_client(tmp_path: Path) -> None:
    env_name = "cfg-env"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="financial_services",
            name="Financial Services",
            tool_servers=[ScenarioToolServer(name="spreadsheets")],
        )
    ]

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"spreadsheets"})),
        patch("simlab.cli.env.get_global_config_from_ctx") as mocked_cfg,
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://cfg"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_cfg.return_value = SimpleNamespace(
            scenario_manager_api_url="https://cfg",
            collinear_api_key="token-123",
        )
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "financial_services"
        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--template",
                "financial_services",
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    mocked_client_cls.assert_called_once_with(base_url="https://cfg", api_key="token-123")


def test_env_init_auto_builds_compose_files(tmp_path: Path) -> None:
    """env init with no template and no tools exits with 1 (no tools selected)."""
    env_name = "empty-env"
    runner = CliRunner()

    with patch("simlab.cli.env._get_registry", return_value=_FakeRegistry(set())):
        result = runner.invoke(
            env,
            ["init", env_name, "--non-interactive"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 1
    assert not (tmp_path / "environments" / env_name).exists()


def test_env_init_template_auto_builds_compose_files(tmp_path: Path) -> None:
    """env init with --template generates env.yaml + docker-compose.yml in one step."""
    env_name = "compose-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="human_resource",
            name="Human Resource",
            tool_servers=[ScenarioToolServer(name="email-env")],
        )
    ]

    fake_compose_output = MagicMock()
    fake_compose_output.tool_endpoints = {"email": "http://localhost:8040/tools"}
    fake_compose_output.env_file = "# No environment variables required"
    del fake_compose_output.readme  # no longer in ComposeOutput

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"email"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
        patch("simlab.cli.env.ComposeEngine") as mocked_engine_cls,
        patch("simlab.cli.env.write_output") as mocked_write,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "human_resource"
        mocked_engine = mocked_engine_cls.return_value
        mocked_engine.compose.return_value = fake_compose_output

        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--template",
                "human_resource",
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output

    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text())
    assert data["tools"] == ["email"]

    mocked_engine.compose.assert_called_once()
    mocked_write.assert_called_once()
    # write_output is called with (output, env_dir)
    call_args = mocked_write.call_args[0]
    assert call_args[1] == env_dir

    assert "docker-compose.yml" in result.output
    assert "email" in result.output
    assert "env up" in result.output


def test_env_init_coding_template_scaffolds_customization_files(tmp_path: Path) -> None:
    env_name = "coding-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="coding",
            name="Coding",
            tool_servers=[ScenarioToolServer(name="coding-env")],
            scenario_guidance_md="# Server Guidance\nUse coding skills first.\n",
        )
    ]
    fake_compose_output = SimpleNamespace(
        tool_endpoints={"coding": "http://localhost:8020/tools"},
        env_file="# No environment variables required",
    )

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"coding"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
        patch("simlab.cli.env.ComposeEngine") as mocked_engine_cls,
        patch("simlab.cli.env.write_output"),
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "coding"
        mocked_engine_cls.return_value.compose.return_value = fake_compose_output

        result = runner.invoke(
            env,
            ["init", env_name, "--template", "coding", "--non-interactive"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    data = yaml.safe_load(out_file.read_text())
    assert data["tools"] == ["coding"]
    assert data["scenario_guidance_md"] == "# Server Guidance\nUse coding skills first."
    assert data["coding"]["setup_scripts"] == ["./coding/setup/install-tools.sh"]
    assert data["coding"]["skills"] == ["./coding/skills"]
    assert data["coding"]["mounts"] == [
        {
            "source": "./coding/fixtures",
            "target": "/workspace/fixtures",
            "read_only": True,
        }
    ]
    assert (env_dir / "coding" / "setup" / "install-tools.sh").exists()
    assert (env_dir / "coding" / "fixtures" / ".gitkeep").exists()
    assert (env_dir / "coding" / "skills" / "example-skill" / "SKILL.md").exists()
    assert (env_dir / "task-bundle" / "tasks" / "example_task.json").exists()
    assert (env_dir / "task-bundle" / "tasks" / "build_cli_task.json").exists()
    assert (env_dir / "task-bundle" / "tasks" / "parse_csv_task.json").exists()
    task_json = json.loads((env_dir / "task-bundle" / "tasks" / "example_task.json").read_text())
    assert task_json["verifiers"][0]["module"] == "simlab.verifiers.custom_coding"
    assert not (env_dir / "task-bundle" / "verifiers" / "generated_task.py").exists()
    assert (env_dir / "task-bundle" / "verifiers" / "__init__.py").exists()
    assert not (env_dir / "task-bundle" / "skills.md").exists()
    assert (env_dir / "README.md").exists()
    assert str(env_dir / "coding" / "setup" / "install-tools.sh") in result.output
    assert str(env_dir / "task-bundle") in result.output


def test_env_init_force_preserves_existing_env_yaml(tmp_path: Path) -> None:
    """env init ENV_NAME --force with existing env.yaml regenerates compose only."""
    env_name = "preserved-env"
    env_dir = tmp_path / "environments" / env_name
    runner = CliRunner()
    env_dir.mkdir(parents=True)
    env_yaml = env_dir / "env.yaml"
    custom_content = {
        "name": env_name,
        "tools": ["email"],
        "overrides": {"email": {"CUSTOM_VAR": "user-value"}},
        "registry": "ghcr.io/example",
        "template": "human_resource",
    }
    env_yaml.write_text(yaml.dump(custom_content, default_flow_style=False, sort_keys=False))

    fake_compose_output = MagicMock()
    fake_compose_output.tool_endpoints = {"email": "http://localhost:8040/tools"}
    fake_compose_output.env_file = "# No environment variables required"

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"email"})),
        patch("simlab.cli.env.ComposeEngine") as mocked_engine_cls,
        patch("simlab.cli.env.write_output") as mocked_write,
    ):
        mocked_engine = mocked_engine_cls.return_value
        mocked_engine.compose.return_value = fake_compose_output

        result = runner.invoke(
            env,
            ["init", env_name, "--force", "--non-interactive"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert "Regenerating from existing" in result.output

    # env.yaml must be unchanged (manual edits preserved)
    data = yaml.safe_load(env_yaml.read_text())
    assert data["tools"] == ["email"]
    assert data.get("overrides") == {"email": {"CUSTOM_VAR": "user-value"}}
    assert data.get("registry") == "ghcr.io/example"

    mocked_engine.compose.assert_called_once()
    mocked_write.assert_called_once()
    call_args = mocked_write.call_args[0]
    assert call_args[1] == env_dir


def test_env_init_force_updates_scenario_guidance_from_file(tmp_path: Path) -> None:
    env_name = "guided-force-env"
    env_dir = tmp_path / "environments" / env_name
    runner = CliRunner()
    env_dir.mkdir(parents=True)
    env_yaml = env_dir / "env.yaml"
    env_yaml.write_text(
        yaml.dump(
            {
                "name": env_name,
                "tools": ["email"],
                "registry": "ghcr.io/example",
                "scenario_guidance_md": "# Old Guidance\nUse the old instructions.\n",
            },
            default_flow_style=False,
            sort_keys=False,
        )
    )
    guidance_file = tmp_path / "new-guidance.md"
    guidance_file.write_text("# New Guidance\nUse the new instructions.\n", encoding="utf-8")

    fake_compose_output = MagicMock()
    fake_compose_output.tool_endpoints = {"email": "http://localhost:8040/tools"}
    fake_compose_output.env_file = "# No environment variables required"

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"email"})),
        patch("simlab.cli.env.ComposeEngine") as mocked_engine_cls,
        patch("simlab.cli.env.write_output") as mocked_write,
    ):
        mocked_engine = mocked_engine_cls.return_value
        mocked_engine.compose.return_value = fake_compose_output

        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--force",
                "--non-interactive",
                "--scenario-guidance-file",
                str(guidance_file),
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    data = yaml.safe_load(env_yaml.read_text())
    assert data["scenario_guidance_md"] == "# New Guidance\nUse the new instructions."
    mocked_engine.compose.assert_called_once()
    mocked_write.assert_called_once()


def test_env_init_force_backfills_missing_coding_scaffold(tmp_path: Path) -> None:
    env_name = "coding-force-env"
    env_dir = tmp_path / "environments" / env_name
    runner = CliRunner()
    env_dir.mkdir(parents=True)
    (env_dir / "task-bundle" / "verifiers").mkdir(parents=True, exist_ok=True)
    (env_dir / "task-bundle" / "skills.md").write_text("# Old guidance\n", encoding="utf-8")
    (env_dir / "task-bundle" / "verifiers" / "generated_task.py").write_text(
        "def verify(run_artifacts):\n    return True\n",
        encoding="utf-8",
    )
    (env_dir / "env.yaml").write_text(
        yaml.dump(
            {
                "name": env_name,
                "tools": ["coding"],
                "coding": {
                    "setup_scripts": ["./coding/setup/install-tools.sh"],
                    "skills": ["./coding/skills"],
                    "mounts": [
                        {
                            "source": "./coding/fixtures",
                            "target": "/workspace/fixtures",
                            "read_only": True,
                        }
                    ],
                },
                "scenario_guidance_md": "# Existing Guidance\nUse coding skills first.\n",
            },
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    fake_compose_output = SimpleNamespace(
        tool_endpoints={"coding": "http://localhost:8020/tools"},
        env_file="# No environment variables required",
    )

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"coding"})),
        patch("simlab.cli.env.ComposeEngine") as mocked_engine_cls,
        patch("simlab.cli.env.write_output"),
    ):
        mocked_engine_cls.return_value.compose.return_value = fake_compose_output

        result = runner.invoke(
            env,
            ["init", env_name, "--force", "--non-interactive"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert (env_dir / "coding" / "setup" / "install-tools.sh").exists()
    assert (env_dir / "task-bundle" / "skills.md").exists()
    assert (env_dir / "task-bundle" / "verifiers" / "generated_task.py").exists()
    assert (env_dir / "README.md").exists()


def test_validate_daytona_coding_assets_rejects_external_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_dir = tmp_path / "environments" / "coding-env"
    env_dir.mkdir(parents=True, exist_ok=True)
    external_script = tmp_path / "shared" / "install-tools.sh"
    external_script.parent.mkdir(parents=True, exist_ok=True)
    external_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (env_dir / "coding" / "skills").mkdir(parents=True, exist_ok=True)

    config = EnvConfig(
        name="coding-env",
        tools=["coding"],
        coding=CodingConfig(
            setup_scripts=[str(external_script)],
            skills=["./coding/skills"],
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        env_module._validate_daytona_coding_assets(config, env_dir)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Daytona mode only supports coding assets located inside the environment" in captured.err
    assert str(external_script.resolve()) in captured.err


def test_poll_health_raises_when_service_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(SystemExit) as exc_info:
        env_module._poll_health(
            lambda: {
                "openhands-agent-server": "healthy",
                "coding-env": "exited",
            },
            timeout=1,
        )

    assert exc_info.value.code == 1


def test_poll_health_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    now = {"value": 0.0}

    def fake_time() -> float:
        now["value"] += 1.0
        return now["value"]

    monkeypatch.setattr(env_module.time, "time", fake_time)
    monkeypatch.setattr(env_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(SystemExit) as exc_info:
        env_module._poll_health(lambda: {"coding-env": "starting"}, timeout=1)

    assert exc_info.value.code == 1


def test_env_init_emits_telemetry(tmp_path: Path) -> None:
    env_name = "telemetry-env"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="human_resource",
            name="Human Resource",
            tool_servers=[ScenarioToolServer(name="email-env")],
        )
    ]
    fake_compose_output = SimpleNamespace(
        tool_endpoints={"email": "http://localhost:8040/tools"},
        env_file="# No environment variables required",
    )

    with (
        patch("simlab.cli.env._get_registry", return_value=_FakeRegistry({"email"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
        patch("simlab.cli.env.ComposeEngine") as mocked_engine_cls,
        patch("simlab.cli.env.write_output"),
        patch("simlab.cli.env.emit_cli_event") as mocked_emit,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "human_resource"
        mocked_engine_cls.return_value.compose.return_value = fake_compose_output

        result = runner.invoke(
            env,
            ["init", env_name, "--template", "human_resource", "--non-interactive"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    mocked_emit.assert_called_once_with(
        "env_init_completed",
        {
            "template_used": True,
            "selected_tool_count": 1,
            "unsupported_tool_count": 0,
            "non_interactive": True,
        },
    )


def test_env_seed_emits_telemetry_when_no_seed_services(tmp_path: Path) -> None:
    env_name = "seed-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: seed-env\ntools:\n  - email\noverrides: {}\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    with (
        patch("simlab.cli.env._get_seed_service_names", return_value=[]),
        patch("simlab.cli.env.emit_cli_event") as mocked_emit,
    ):
        result = runner.invoke(
            env,
            ["seed", env_name],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    mocked_emit.assert_called_once_with(
        "env_seed_completed",
        {
            "mode": "local",
            "verify_only": False,
            "seed_service_count": 0,
            "tool_count": 1,
        },
    )
