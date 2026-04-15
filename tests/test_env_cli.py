from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import patch

import click
import pytest
import simlab.cli.env as env_module
import simlab.runtime.env_lifecycle as lifecycle_module
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

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._supported_tools)

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


class _ComposeUpCall(TypedDict, total=False):
    up_cmd: list[str]
    has_builds: bool
    health: dict[str, str]
    timeout: int


def _fake_compose_output(
    *,
    tool_endpoints: dict[str, str] | None = None,
    env_file: str = "# No environment variables required",
) -> SimpleNamespace:
    return SimpleNamespace(
        tool_endpoints=tool_endpoints or {},
        env_file=env_file,
    )


def test_env_init_template_uses_server_scenario_tools(tmp_path: Path) -> None:
    env_name = "my-env"
    env_dir = tmp_path / "environments" / env_name  # set SIMLAB_ENVIRONMENTS_DIR in invoke
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="financial_services:1.0.0",
            name="Financial Services",
            tool_servers=[
                ScenarioToolServer(name="spreadsheets"),
                ScenarioToolServer(name="missing-tool"),
            ],
            scenario_guidance_md="# Scenario Guidance\nFollow the domain conventions.\n",
        )
    ]

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"spreadsheets"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=_fake_compose_output()),
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "financial_services:1.0.0"
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
    assert "Starting from template 'financial_services:1.0.0': spreadsheets" in result.output
    assert (
        "Ignoring unsupported tool servers from template 'financial_services:1.0.0': missing-tool"
        in result.output
    )

    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text())
    assert data["registry"] == DEFAULT_IMAGE_REGISTRY
    assert data["template"] == "financial_services:1.0.0"
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
            scenario_id="financial_services:1.0.0",
            name="Financial Services",
            tool_servers=[ScenarioToolServer(name="spreadsheets")],
            scenario_guidance_md="# Server Guidance\nFollow the server defaults.\n",
        )
    ]

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"spreadsheets"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=_fake_compose_output()),
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "financial_services:1.0.0"
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

    with patch("simlab.cli.env.build_registry", return_value=_FakeRegistry(set())):
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
            scenario_id="human_resource:1.0.0",
            name="Human Resource",
            tool_servers=[
                ScenarioToolServer(name="frappe-hrms-env"),
                ScenarioToolServer(name="email-env"),
            ],
        )
    ]

    with (
        patch(
            "simlab.cli.env.build_registry",
            return_value=_FakeRegistry({"frappe-hrms", "email"}),
        ),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "human_resource:1.0.0"
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
    assert "Starting from template 'human_resource:1.0.0': frappe-hrms, email" in result.output

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
            scenario_id="financial_services:1.0.0",
            name="Financial Services",
            tool_servers=[ScenarioToolServer(name="google-workspace-tool-server")],
        )
    ]

    with (
        patch(
            "simlab.cli.env.build_registry",
            return_value=_FakeRegistry({"google-workspace"}),
        ),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "financial_services:1.0.0"
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


def test_env_init_template_maps_erp_service_name(tmp_path: Path) -> None:
    env_name = "erp-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="erp:1.0.0",
            name="ERP",
            tool_servers=[
                ScenarioToolServer(name="erp-env"),
                ScenarioToolServer(name="email-env"),
            ],
        )
    ]

    with (
        patch(
            "simlab.cli.env.build_registry",
            return_value=_FakeRegistry({"erp", "email"}),
        ),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "erp:1.0.0"
        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--template",
                "erp",
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert "Starting from template 'erp:1.0.0': erp, email" in result.output

    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text())
    assert data["registry"] == DEFAULT_IMAGE_REGISTRY
    assert data["tools"] == ["erp", "email"]


def test_env_init_template_maps_crm_service_name(tmp_path: Path) -> None:
    env_name = "crm-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="crm_sales:1.0.0",
            name="CRM Sales",
            tool_servers=[ScenarioToolServer(name="crm-env")],
        )
    ]

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"crm"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "crm_sales:1.0.0"
        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--template",
                "crm_sales",
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text())
    assert data["registry"] == DEFAULT_IMAGE_REGISTRY
    assert data["tools"] == ["crm"]


def test_env_init_uses_default_registry_when_flag_omitted(tmp_path: Path) -> None:
    env_name = "reg-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="financial_services:1.0.0",
            name="Financial Services",
            tool_servers=[ScenarioToolServer(name="spreadsheets")],
        )
    ]

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"spreadsheets"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=_fake_compose_output()),
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "financial_services:1.0.0"
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
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"spreadsheets"})),
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
            scenario_id="financial_services:1.0.0",
            name="Financial Services",
            tool_servers=[ScenarioToolServer(name="spreadsheets")],
        )
    ]

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"spreadsheets"})),
        patch("simlab.cli.env.get_global_config_from_ctx") as mocked_cfg,
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://cfg"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=_fake_compose_output()),
    ):
        mocked_cfg.return_value = SimpleNamespace(
            scenario_manager_api_url="https://cfg",
            collinear_api_key="token-123",
        )
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "financial_services:1.0.0"
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

    with patch("simlab.cli.env.build_registry", return_value=_FakeRegistry(set())):
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

    fake_compose_output = _fake_compose_output(
        tool_endpoints={"email": "http://localhost:8040/tools"}
    )

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"email"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
        patch(
            "simlab.cli.env.regenerate_env_artifacts",
            return_value=fake_compose_output,
        ) as mocked_regenerate,
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

    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text())
    assert data["tools"] == ["email"]

    mocked_regenerate.assert_called_once_with(env_dir)

    assert "docker-compose.yml" in result.output
    assert "email" in result.output
    assert "tasks list" in result.output


def test_env_init_coding_template_scaffolds_customization_files(tmp_path: Path) -> None:
    env_name = "coding-env"
    env_dir = tmp_path / "environments" / env_name
    out_file = env_dir / "env.yaml"
    runner = CliRunner()
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="coding:1.0.0",
            name="Coding",
            tool_servers=[ScenarioToolServer(name="coding-env")],
            scenario_guidance_md="# Server Guidance\nUse coding skills first.\n",
        )
    ]
    fake_compose_output = _fake_compose_output(
        tool_endpoints={"coding": "http://localhost:8020/tools"}
    )

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"coding"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=fake_compose_output),
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "coding:1.0.0"

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
    assert (env_dir / "task-bundle" / "verifiers" / "custom_coding.py").exists()
    assert not (env_dir / "task-bundle" / "skills.md").exists()
    assert (env_dir / "README.md").exists()
    assert str(env_dir / "coding" / "setup" / "install-tools.sh") in result.output
    assert str(env_dir / "task-bundle") in result.output


def test_env_init_with_mcp_servers_persists_and_calls_compose_with_env_dir(
    tmp_path: Path,
) -> None:
    """env init with --mcp-servers writes mcp-servers.json and calls compose with env_dir."""
    env_name = "mcp-env"
    env_dir = tmp_path / "environments" / env_name
    mcp_file = tmp_path / "input-mcp.json"
    mcp_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "notion": {"url": "https://mcp.notion.com/mcp"},
                    "weather": {
                        "command": "uvx",
                        "args": ["mcp-weather"],
                        "env": {"ACCUWEATHER_API_KEY": ""},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    fake_compose_output = _fake_compose_output(env_file="")

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry(set())),
        patch(
            "simlab.cli.env.regenerate_env_artifacts",
            return_value=fake_compose_output,
        ) as mocked_regenerate,
    ):
        result = runner.invoke(
            env,
            [
                "init",
                env_name,
                "--mcp-servers",
                str(mcp_file),
                "--non-interactive",
            ],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    persisted = env_dir / "mcp-servers.json"
    assert persisted.exists()
    data = json.loads(persisted.read_text())
    assert "mcpServers" in data
    assert "notion" in data["mcpServers"]
    assert data["mcpServers"]["notion"]["url"] == "https://mcp.notion.com/mcp"
    assert "weather" in data["mcpServers"]
    mocked_regenerate.assert_called_once_with(env_dir)


def test_env_init_with_mcp_servers_interactive_skips_picker_when_user_declines_extra_tools(
    tmp_path: Path,
) -> None:
    env_name = "mcp-env"
    env_dir = tmp_path / "environments" / env_name
    mcp_file = tmp_path / "input-mcp.json"
    mcp_file.write_text(
        json.dumps({"mcpServers": {"notion": {"url": "https://mcp.notion.com/mcp"}}}),
        encoding="utf-8",
    )
    runner = CliRunner()
    fake_compose_output = _fake_compose_output()

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"email"})),
        patch("simlab.cli.env.click.confirm", return_value=False) as mocked_confirm,
        patch("simlab.cli.env._interactive_select") as mocked_picker,
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=fake_compose_output),
    ):
        result = runner.invoke(
            env,
            ["init", env_name, "--mcp-servers", str(mcp_file)],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert env_dir.exists()
    mocked_confirm.assert_called_once_with(
        "Add catalog tools in addition to the MCP servers?",
        default=False,
    )
    mocked_picker.assert_not_called()


def test_env_init_with_mcp_servers_interactive_can_add_extra_tools(tmp_path: Path) -> None:
    env_name = "mcp-env"
    env_dir = tmp_path / "environments" / env_name
    mcp_file = tmp_path / "input-mcp.json"
    mcp_file.write_text(
        json.dumps({"mcpServers": {"notion": {"url": "https://mcp.notion.com/mcp"}}}),
        encoding="utf-8",
    )
    runner = CliRunner()
    fake_compose_output = _fake_compose_output(
        tool_endpoints={"email": "http://localhost:8040/tools"}
    )

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"email"})),
        patch("simlab.cli.env.click.confirm", return_value=True),
        patch("simlab.cli.env._interactive_select", return_value=["email"]) as mocked_picker,
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=fake_compose_output),
    ):
        result = runner.invoke(
            env,
            ["init", env_name, "--mcp-servers", str(mcp_file)],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert env_dir.exists()
    mocked_picker.assert_called_once()
    data = yaml.safe_load((env_dir / "env.yaml").read_text())
    assert data["tools"] == ["email"]


def test_env_init_interactive_cancel_aborts_before_writing_files(tmp_path: Path) -> None:
    env_name = "cancelled-env"
    mcp_file = tmp_path / "input-mcp.json"
    mcp_file.write_text(
        json.dumps({"mcpServers": {"notion": {"url": "https://mcp.notion.com/mcp"}}}),
        encoding="utf-8",
    )
    runner = CliRunner()

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"email"})),
        patch("simlab.cli.env.click.confirm", return_value=True),
        patch("simlab.cli.env._interactive_select", side_effect=click.Abort()),
    ):
        result = runner.invoke(
            env,
            ["init", env_name, "--mcp-servers", str(mcp_file)],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code != 0
    assert not (tmp_path / "environments" / env_name).exists()


def test_env_init_mcp_servers_invalid_json_fails(tmp_path: Path) -> None:
    """env init with --mcp-servers pointing to invalid JSON exits with error."""
    env_name = "mcp-fail"
    mcp_file = tmp_path / "bad-mcp.json"
    mcp_file.write_text('{"mcpServers": {"x": {}}}', encoding="utf-8")  # missing url/command
    runner = CliRunner()
    with patch("simlab.cli.env.build_registry", return_value=_FakeRegistry(set())):
        result = runner.invoke(
            env,
            ["init", env_name, "--mcp-servers", str(mcp_file), "--non-interactive"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )
    assert result.exit_code == 1
    assert "must have" in result.output or "url" in result.output or "command" in result.output


def test_env_init_mcp_servers_invalid_shape_fails_cleanly(tmp_path: Path) -> None:
    env_name = "mcp-bad-shape"
    mcp_file = tmp_path / "bad-shape-mcp.json"
    mcp_file.write_text('{"mcpServers": []}', encoding="utf-8")
    runner = CliRunner()
    with patch("simlab.cli.env.build_registry", return_value=_FakeRegistry(set())):
        result = runner.invoke(
            env,
            ["init", env_name, "--mcp-servers", str(mcp_file), "--non-interactive"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 1
    assert "mcpServers must be an object" in result.output


def test_env_init_mcp_servers_conflicting_tool_name_fails_before_persist(tmp_path: Path) -> None:
    env_name = "mcp-conflict"
    mcp_file = tmp_path / "conflict-mcp.json"
    mcp_file.write_text(
        json.dumps({"mcpServers": {"email": {"url": "https://example.com/mcp"}}}),
        encoding="utf-8",
    )
    runner = CliRunner()

    with patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"email"})):
        result = runner.invoke(
            env,
            ["init", env_name, "--mcp-servers", str(mcp_file), "--non-interactive"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 1
    assert "conflicts with an existing tool server" in result.output
    assert not (tmp_path / "environments" / env_name / "mcp-servers.json").exists()


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

    fake_compose_output = _fake_compose_output(
        tool_endpoints={"email": "http://localhost:8040/tools"}
    )

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"email"})),
        patch(
            "simlab.cli.env.regenerate_env_artifacts",
            return_value=fake_compose_output,
        ) as mocked_regenerate,
    ):
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

    mocked_regenerate.assert_called_once_with(env_dir)


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

    fake_compose_output = _fake_compose_output(
        tool_endpoints={"email": "http://localhost:8040/tools"}
    )

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"email"})),
        patch(
            "simlab.cli.env.regenerate_env_artifacts",
            return_value=fake_compose_output,
        ) as mocked_regenerate,
    ):
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
    mocked_regenerate.assert_called_once_with(env_dir)


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
    fake_compose_output = _fake_compose_output(
        tool_endpoints={"coding": "http://localhost:8020/tools"}
    )

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"coding"})),
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=fake_compose_output),
    ):
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
        lifecycle_module.validate_daytona_coding_assets(config, env_dir)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Daytona mode only supports coding assets located inside the environment" in captured.err
    assert str(external_script.resolve()) in captured.err


def test_env_init_overwrite_without_mcp_servers_clears_persisted_mcp_config(tmp_path: Path) -> None:
    env_name = "mcp-reset-env"
    env_dir = tmp_path / "environments" / env_name
    runner = CliRunner()
    env_dir.mkdir(parents=True)
    (env_dir / "env.yaml").write_text(
        yaml.dump({"name": env_name, "tools": []}, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    mcp_file = env_dir / "mcp-servers.json"
    mcp_file.write_text(
        json.dumps({"mcpServers": {"notion": {"url": "https://mcp.notion.com/mcp"}}}),
        encoding="utf-8",
    )

    fake_compose_output = _fake_compose_output()

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"email"})),
        patch("simlab.cli.env._interactive_select", return_value=["email"]),
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=fake_compose_output),
    ):
        result = runner.invoke(
            env,
            ["init", env_name],
            input="y\n",
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert not mcp_file.exists()
    assert "Removed persisted MCP servers config" in result.output


def test_env_init_force_without_mcp_servers_clears_persisted_mcp_config(tmp_path: Path) -> None:
    env_name = "mcp-reset-env"
    env_dir = tmp_path / "environments" / env_name
    runner = CliRunner()
    env_dir.mkdir(parents=True)
    (env_dir / "env.yaml").write_text(
        yaml.dump({"name": env_name, "tools": []}, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    mcp_file = env_dir / "mcp-servers.json"
    mcp_file.write_text(
        json.dumps({"mcpServers": {"notion": {"url": "https://mcp.notion.com/mcp"}}}),
        encoding="utf-8",
    )

    fake_compose_output = _fake_compose_output()

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry(set())),
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=fake_compose_output),
    ):
        result = runner.invoke(
            env,
            ["init", env_name, "--force", "--non-interactive"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert not mcp_file.exists()
    assert "Removed persisted MCP servers config" in result.output


def test_env_custom_tools_add_scaffolds_enables_and_regenerates(tmp_path: Path) -> None:
    env_name = "custom-tools-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True)
    env_yaml = env_dir / "env.yaml"
    env_yaml.write_text(
        yaml.dump({"name": env_name, "tools": []}, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    runner = CliRunner()

    with (
        patch("simlab.env_custom_tools.build_registry", return_value=_FakeRegistry(set())),
        patch(
            "simlab.env_custom_tools.regenerate_env_artifacts",
            return_value=_fake_compose_output(),
        ) as mocked_regenerate,
    ):
        result = runner.invoke(
            env,
            ["custom-tools", "add", env_name, "harbor-main"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    tool_file = env_dir / "custom-tools" / "harbor-main.yaml"
    assert tool_file.exists()
    assert "name: harbor-main" in tool_file.read_text(encoding="utf-8")
    data = yaml.safe_load(env_yaml.read_text(encoding="utf-8"))
    assert data["tools"] == ["harbor-main"]
    mocked_regenerate.assert_called_once_with(env_dir)


def test_env_up_with_no_local_services_skips_docker_compose(tmp_path: Path) -> None:
    env_name = "url-only-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True)
    (env_dir / "env.yaml").write_text(
        yaml.dump({"name": env_name, "tools": []}, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    (env_dir / "docker-compose.yml").write_text(
        yaml.dump(
            {
                "services": {},
                "networks": {"simlab": {"driver": "bridge"}},
            },
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    with (
        patch("simlab.runtime.env_lifecycle.subprocess.run") as mocked_run,
        patch("simlab.runtime.env_lifecycle._get_preseed_service_names", return_value=[]),
        patch("simlab.runtime.env_lifecycle._get_seed_service_names", return_value=[]),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["up", env_name],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert "No local services defined" in result.output
    mocked_run.assert_not_called()


def test_env_up_daytona_with_no_services_skips_daytona_startup(tmp_path: Path) -> None:
    env_name = "url-only-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True)
    (env_dir / "env.yaml").write_text(
        yaml.dump({"name": env_name, "tools": []}, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    (env_dir / "docker-compose.yml").write_text(
        yaml.dump(
            {
                "services": {},
                "networks": {"simlab": {"driver": "bridge"}},
            },
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    with (
        patch("simlab.cli.env._up_daytona") as mocked_up_daytona,
        patch("simlab.runtime.env_lifecycle._get_daytona_runner") as mocked_get_daytona_runner,
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["up", env_name, "--daytona"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert "No Daytona services defined" in result.output
    mocked_up_daytona.assert_not_called()
    mocked_get_daytona_runner.assert_not_called()


def test_add_mcp_gateway_endpoint_skips_url_only_mcp_servers(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "mcp-servers.json").write_text(
        json.dumps({"mcpServers": {"notion": {"url": "https://mcp.notion.com/mcp"}}}),
        encoding="utf-8",
    )

    endpoints = env_module._add_mcp_gateway_endpoint(
        {"email": "http://localhost:8040"},
        env_dir=env_dir,
    )

    assert endpoints == {"email": "http://localhost:8040"}


def test_add_mcp_gateway_endpoint_adds_command_server_gateway(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "mcp-servers.json").write_text(
        json.dumps({"mcpServers": {"weather": {"command": "uvx", "args": ["mcp-weather"]}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(env_module, "get_mcp_gateway_host_port", lambda _env_dir: 8081)

    endpoints = env_module._add_mcp_gateway_endpoint({}, env_dir=env_dir)

    assert endpoints == {
        env_module.ComposeEngine.MCP_GATEWAY_SERVICE_NAME: "http://localhost:8081/mcp"
    }


def test_env_down_with_no_current_services_still_attempts_compose_down(tmp_path: Path) -> None:
    env_name = "url-only-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True)
    (env_dir / "env.yaml").write_text(
        yaml.dump({"name": env_name, "tools": []}, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    (env_dir / "docker-compose.yml").write_text(
        yaml.dump(
            {
                "services": {},
                "networks": {"simlab": {"driver": "bridge"}},
            },
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    completed = SimpleNamespace(returncode=0, stderr="")

    with (
        patch("simlab.runtime.env_lifecycle.subprocess.run", return_value=completed) as mocked_run,
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["down", env_name],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )

    assert result.exit_code == 0, result.output
    assert "attempting teardown anyway" in result.output
    mocked_run.assert_called_once_with(
        ["docker", "compose", "-f", str(env_dir / "docker-compose.yml"), "down"],
        capture_output=True,
        text=True,
    )


def test_local_health_fetcher_uses_compose_ps_all(tmp_path: Path) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    completed = SimpleNamespace(returncode=0, stdout="svc\tUp 1 second\n")

    with patch("simlab.runtime.env_lifecycle.subprocess.run", return_value=completed) as mocked_run:
        fetch = lifecycle_module._local_health_fetcher(compose_file)
        assert fetch() == {"svc": "running"}

    mocked_run.assert_called_once_with(
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


def test_local_health_fetcher_preserves_hyphenated_service_names(tmp_path: Path) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "services:\n  harbor-openhands-agent-server: {}\n  harbor-coding-env: {}\n",
        encoding="utf-8",
    )
    completed = SimpleNamespace(
        returncode=0,
        stdout=(
            "env-harbor-openhands-agent-server-1\tUp 5 seconds (healthy)\n"
            "env-harbor-coding-env-1\tUp 5 seconds (health: starting)\n"
        ),
    )

    with patch("simlab.runtime.env_lifecycle.subprocess.run", return_value=completed):
        fetch = lifecycle_module._local_health_fetcher(compose_file)
        assert fetch() == {
            "harbor-openhands-agent-server": "healthy",
            "harbor-coding-env": "starting",
        }


def test_local_health_fetcher_handles_hyphenated_project_names(tmp_path: Path) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "services:\n  harbor-openhands-agent-server: {}\n  harbor-coding-env: {}\n",
        encoding="utf-8",
    )
    completed = SimpleNamespace(
        returncode=0,
        stdout=(
            "baseline-env-harbor-openhands-agent-server-1\tUp 5 seconds (healthy)\n"
            "baseline-env-harbor-coding-env-1\tUp 5 seconds (health: starting)\n"
        ),
    )

    with patch("simlab.runtime.env_lifecycle.subprocess.run", return_value=completed):
        fetch = lifecycle_module._local_health_fetcher(compose_file)
        assert fetch() == {
            "harbor-openhands-agent-server": "healthy",
            "harbor-coding-env": "starting",
        }


def test_run_compose_up_local_shows_spinner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = {"value": 0.0}

    def fake_time() -> float:
        now["value"] += 1.0
        return now["value"]

    class FakeStream:
        def __init__(self, chunks: list[str]) -> None:
            self._chunks = iter(chunks)
            self.read_calls = 0

        def read(self, _size: int) -> str:
            self.read_calls += 1
            return next(self._chunks, "")

        def close(self) -> None:
            return None

    class FakePopen:
        def __init__(self) -> None:
            self.returncode = 0
            self._poll_values = iter([None, 0])
            self.stdout = FakeStream(["compose stdout"])
            self.stderr = FakeStream([""])

        def poll(self) -> int | None:
            value = next(self._poll_values)
            if value is not None:
                assert self.stdout.read_calls > 0
                assert self.stderr.read_calls > 0
                self.returncode = value
            return value

        def wait(self) -> int:
            return self.returncode

    monkeypatch.setattr(lifecycle_module.time, "time", fake_time)
    monkeypatch.setattr(lifecycle_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(lifecycle_module, "_clear_lines", lambda _n: None)
    monkeypatch.setattr(
        lifecycle_module.subprocess,
        "Popen",
        lambda *args, **kwargs: FakePopen(),
    )

    result = lifecycle_module._run_compose_up_local(["docker", "compose", "up"], has_builds=True)

    assert result.returncode == 0
    assert result.stdout == "compose stdout"
    assert "Starting docker compose build..." in capsys.readouterr().out


def test_ensure_env_started_local_uses_compose_spinner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "services:\n  web:\n    build:\n      context: .\n",
        encoding="utf-8",
    )
    called: _ComposeUpCall = {}

    monkeypatch.setattr(
        lifecycle_module, "_get_preseed_service_names", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        lifecycle_module,
        "_run_compose_up_local",
        lambda up_cmd, *, has_builds, quiet=False: (
            called.update({"up_cmd": up_cmd, "has_builds": has_builds})
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(lifecycle_module, "_local_health_fetcher", lambda _compose_file: dict)
    monkeypatch.setattr(
        lifecycle_module,
        "_poll_health",
        lambda health_fetcher, timeout=180, **_kw: called.update(
            {"health": health_fetcher(), "timeout": timeout}
        ),
    )

    lifecycle_module.ensure_env_started_local(tmp_path, EnvConfig(name="test-env"))

    assert called["has_builds"] is True
    assert "--build" in called["up_cmd"]
    assert called["timeout"] == 180
    assert called["health"] == {}


def test_poll_health_raises_when_service_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(SystemExit) as exc_info:
        lifecycle_module._poll_health(
            lambda: {
                "openhands-agent-server": "healthy",
                "coding-env": "exited",
            },
            timeout=1,
        )

    assert exc_info.value.code == 1


def test_poll_health_requires_stable_ready_state(monkeypatch: pytest.MonkeyPatch) -> None:
    now = {"value": 0.0}

    def fake_time() -> float:
        now["value"] += 1.0
        return now["value"]

    states = iter(
        [
            {"mcp-gateway": "running"},
            {"mcp-gateway": "running"},
        ]
    )

    monkeypatch.setattr(lifecycle_module.time, "time", fake_time)
    monkeypatch.setattr(lifecycle_module.time, "sleep", lambda _seconds: None)

    lifecycle_module._poll_health(lambda: next(states), timeout=8)


def test_poll_health_does_not_succeed_on_single_running_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"value": 0.0}

    def fake_time() -> float:
        now["value"] += 1.0
        return now["value"]

    states = iter(
        [
            {"mcp-gateway": "running"},
            {"mcp-gateway": "exited"},
        ]
    )

    monkeypatch.setattr(lifecycle_module.time, "time", fake_time)
    monkeypatch.setattr(lifecycle_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(SystemExit) as exc_info:
        lifecycle_module._poll_health(lambda: next(states), timeout=5)

    assert exc_info.value.code == 1


def test_poll_health_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    now = {"value": 0.0}

    def fake_time() -> float:
        now["value"] += 1.0
        return now["value"]

    monkeypatch.setattr(lifecycle_module.time, "time", fake_time)
    monkeypatch.setattr(lifecycle_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(SystemExit) as exc_info:
        lifecycle_module._poll_health(lambda: {"coding-env": "starting"}, timeout=1)

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
    fake_compose_output = _fake_compose_output(
        tool_endpoints={"email": "http://localhost:8040/tools"}
    )

    with (
        patch("simlab.cli.env.build_registry", return_value=_FakeRegistry({"email"})),
        patch("simlab.cli.env.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.env.ScenarioManagerClient") as mocked_client_cls,
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=fake_compose_output),
        patch("simlab.cli.env.emit_cli_event") as mocked_emit,
    ):
        mocked_client = mocked_client_cls.return_value
        mocked_client.list_scenarios.return_value = fake_scenarios
        mocked_client.resolve_template_to_backend_id.return_value = "human_resource"

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
            "template_name": "human_resource",
            "environment_name": env_name,
            "selected_tool_count": 1,
            "selected_tools": ["email"],
            "unsupported_tool_count": 0,
            "non_interactive": True,
        },
    )


def test_env_init_force_regeneration_emits_telemetry_from_env_yaml(tmp_path: Path) -> None:
    env_name = "regen-telemetry-env"
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: old-name\n"
        "tools:\n"
        "  - email\n"
        "  - rocketchat\n"
        "template: human_resource\n"
        "overrides: {}\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    fake_compose_output = _fake_compose_output(
        tool_endpoints={
            "email": "http://localhost:8040/tools",
            "rocketchat": "http://localhost:8041/tools",
        }
    )

    with (
        patch(
            "simlab.cli.env.build_registry",
            return_value=_FakeRegistry({"email", "rocketchat"}),
        ),
        patch("simlab.cli.env.regenerate_env_artifacts", return_value=fake_compose_output),
        patch("simlab.cli.env.emit_cli_event") as mocked_emit,
    ):
        result = runner.invoke(
            env,
            ["init", env_name, "--force", "--non-interactive"],
            catch_exceptions=False,
            env={
                "SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments"),
                "COLLINEAR_API_KEY": "",
            },
        )

    assert result.exit_code == 0, result.output
    mocked_emit.assert_called_once_with(
        "env_init_completed",
        {
            "template_used": True,
            "template_name": "human_resource",
            "environment_name": env_name,
            "selected_tool_count": 2,
            "selected_tools": ["email", "rocketchat"],
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
        patch("simlab.runtime.env_lifecycle._get_seed_service_names", return_value=[]),
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


# ---------------------------------------------------------------------------
# env list
# ---------------------------------------------------------------------------


def _create_env(envs_root: Path, name: str, tools: list[str]) -> Path:
    """Helper to create a minimal environment directory."""
    env_dir = envs_root / name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        yaml.dump({"name": name, "tools": tools}, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return env_dir


def test_env_list_shows_environments(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    _create_env(envs_root, "alpha", ["hr", "coding"])
    _create_env(envs_root, "beta", ["email"])

    runner = CliRunner()
    with (
        patch("simlab.runtime.env_lifecycle.has_any_running_containers", return_value=False),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["list"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    assert "NAME" in result.output
    assert "STATUS" in result.output
    assert "TOOLS" in result.output
    assert "CREATED" in result.output
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "hr, coding" in result.output
    assert "email" in result.output


def test_env_list_json_output(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    _create_env(envs_root, "my-env", ["hr"])

    runner = CliRunner()
    with (
        patch("simlab.runtime.env_lifecycle.has_any_running_containers", return_value=False),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["list", "--json"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["name"] == "my-env"
    assert data[0]["status"] == "stopped"
    assert data[0]["tools"] == ["hr"]
    assert "created" in data[0]
    assert "path" in data[0]


def test_env_list_quiet_output(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    _create_env(envs_root, "env-a", ["hr"])
    _create_env(envs_root, "env-b", ["coding"])

    runner = CliRunner()
    with (
        patch("simlab.runtime.env_lifecycle.has_any_running_containers", return_value=False),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["list", "--quiet"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines == ["env-a", "env-b"]


def test_env_list_empty_dir(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    envs_root.mkdir()

    runner = CliRunner()
    with patch("simlab.cli.env.emit_cli_event"):
        result = runner.invoke(
            env,
            ["list"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    assert "No environments found" in result.output


def test_env_list_no_dir(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"  # does not exist

    runner = CliRunner()
    with patch("simlab.cli.env.emit_cli_event"):
        result = runner.invoke(
            env,
            ["list"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    assert "No environments directory found" in result.output


def test_env_list_skips_non_env_dirs(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    _create_env(envs_root, "real-env", ["hr"])
    # Create a directory without env.yaml
    (envs_root / "not-an-env").mkdir()

    runner = CliRunner()
    with (
        patch("simlab.runtime.env_lifecycle.has_any_running_containers", return_value=False),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["list", "--quiet"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "real-env"


def test_env_list_status_running(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    _create_env(envs_root, "running-env", ["hr"])

    runner = CliRunner()
    with (
        patch("simlab.runtime.env_lifecycle.has_any_running_containers", return_value=True),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["list"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    assert "running" in result.output


def test_env_list_error_status(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    bad_dir = envs_root / "bad-env"
    bad_dir.mkdir(parents=True)
    (bad_dir / "env.yaml").write_text("{{invalid yaml", encoding="utf-8")

    runner = CliRunner()
    with patch("simlab.cli.env.emit_cli_event"):
        result = runner.invoke(
            env,
            ["list"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    assert "error" in result.output


# ---------------------------------------------------------------------------
# env delete
# ---------------------------------------------------------------------------


def test_env_delete_removes_directory(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    env_dir = _create_env(envs_root, "doomed", ["hr"])

    runner = CliRunner()
    completed = SimpleNamespace(returncode=0, stderr="")
    with (
        patch("simlab.cli.env.has_any_running_containers", return_value=False),
        patch("simlab.runtime.env_lifecycle.subprocess.run", return_value=completed),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["delete", "doomed", "--force"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    assert "deleted" in result.output.lower()
    assert not env_dir.exists()


def test_env_delete_refuses_running_env(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    _create_env(envs_root, "active", ["hr"])

    runner = CliRunner()
    with (
        patch("simlab.cli.env.has_any_running_containers", return_value=True),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["delete", "active", "--force"],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code != 0
    assert "currently running" in result.output
    assert "simlab env down" in result.output


def test_env_delete_calls_purge_docker(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    env_dir = _create_env(envs_root, "purge-me", ["hr"])
    (env_dir / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")

    runner = CliRunner()
    completed = SimpleNamespace(returncode=0, stderr="")
    with (
        patch("simlab.cli.env.has_any_running_containers", return_value=False),
        patch("simlab.runtime.env_lifecycle.subprocess.run", return_value=completed) as mocked_run,
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["delete", "purge-me", "--force"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    mocked_run.assert_called_once()
    call_args = mocked_run.call_args[0][0]
    assert "-v" in call_args
    assert "--remove-orphans" in call_args


def test_env_delete_prompts_without_force(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    env_dir = _create_env(envs_root, "ask-me", ["hr"])

    runner = CliRunner()
    completed = SimpleNamespace(returncode=0, stderr="")
    with (
        patch("simlab.cli.env.has_any_running_containers", return_value=False),
        patch("simlab.runtime.env_lifecycle.subprocess.run", return_value=completed),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["delete", "ask-me"],
            input="y\n",
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    assert not env_dir.exists()


def test_env_delete_aborts_on_no(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    env_dir = _create_env(envs_root, "keep-me", ["hr"])

    runner = CliRunner()
    with (
        patch("simlab.cli.env.has_any_running_containers", return_value=False),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["delete", "keep-me"],
            input="n\n",
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    assert "Aborted" in result.output
    assert env_dir.exists()


def test_env_delete_not_found(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    envs_root.mkdir()

    runner = CliRunner()
    with patch("simlab.cli.env.emit_cli_event"):
        result = runner.invoke(
            env,
            ["delete", "ghost"],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code != 0


def test_env_delete_warns_daytona_state(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    env_dir = _create_env(envs_root, "with-daytona", ["hr"])
    (env_dir / "daytona-state.json").write_text('{"sandbox_id": "abc"}', encoding="utf-8")

    runner = CliRunner()
    completed = SimpleNamespace(returncode=0, stderr="")
    with (
        patch("simlab.cli.env.has_any_running_containers", return_value=False),
        patch("simlab.runtime.env_lifecycle.subprocess.run", return_value=completed),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["delete", "with-daytona", "--force"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    assert "Daytona state" in result.output
    assert "--daytona" in result.output


def test_env_delete_no_compose_skips_docker_cleanup(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    env_dir = _create_env(envs_root, "no-compose", ["hr"])
    # No docker-compose.yml in the env dir

    runner = CliRunner()
    with (
        patch("simlab.cli.env.has_any_running_containers", return_value=False),
        patch("simlab.runtime.env_lifecycle.subprocess.run") as mocked_run,
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["delete", "no-compose", "--force"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    mocked_run.assert_not_called()
    assert not env_dir.exists()


def test_env_delete_refuses_non_simlab_directory(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    stray_dir = envs_root / "not-an-env"
    stray_dir.mkdir(parents=True)
    (stray_dir / "random-file.txt").write_text("not simlab", encoding="utf-8")

    runner = CliRunner()
    with patch("simlab.cli.env.emit_cli_event"):
        result = runner.invoke(
            env,
            ["delete", "not-an-env", "--force"],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code != 0
    assert "not a SimLab environment" in result.output
    assert stray_dir.exists()


def test_env_delete_refuses_path_traversal(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    envs_root.mkdir(parents=True)
    # Create a victim directory outside the environments root that has env.yaml
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "env.yaml").write_text("name: victim\ntools: []\n", encoding="utf-8")

    runner = CliRunner()
    with patch("simlab.cli.env.emit_cli_event"):
        result = runner.invoke(
            env,
            ["delete", "../victim", "--force"],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code != 0
    assert "outside the environments directory" in result.output
    assert victim.exists()


def test_env_delete_aborts_on_docker_failure(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    env_dir = _create_env(envs_root, "docker-fail", ["hr"])
    (env_dir / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")

    runner = CliRunner()
    failed = SimpleNamespace(returncode=1, stderr="Cannot connect to Docker daemon")
    with (
        patch("simlab.cli.env.has_any_running_containers", return_value=False),
        patch("simlab.runtime.env_lifecycle.subprocess.run", return_value=failed),
        patch("simlab.cli.env.emit_cli_event"),
    ):
        result = runner.invoke(
            env,
            ["delete", "docker-fail", "--force"],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code != 0
    assert "Docker resources could not be cleaned up" in result.output
    assert env_dir.exists()


def test_env_delete_emits_telemetry(tmp_path: Path) -> None:
    envs_root = tmp_path / "environments"
    _create_env(envs_root, "telemetry-env", ["hr"])

    runner = CliRunner()
    completed = SimpleNamespace(returncode=0, stderr="")
    with (
        patch("simlab.cli.env.has_any_running_containers", return_value=False),
        patch("simlab.runtime.env_lifecycle.subprocess.run", return_value=completed),
        patch("simlab.cli.env.emit_cli_event") as mocked_emit,
    ):
        result = runner.invoke(
            env,
            ["delete", "telemetry-env", "--force"],
            catch_exceptions=False,
            env={"SIMLAB_ENVIRONMENTS_DIR": str(envs_root)},
        )

    assert result.exit_code == 0, result.output
    mocked_emit.assert_any_call(
        "env_delete_completed",
        {"docker_purged": True},
    )
