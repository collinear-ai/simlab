from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from simlab.api.schemas import ScenarioSummary
from simlab.api.schemas import ScenarioToolServer
from simlab.cli.main import cli
from simlab.cli.tools import tools
from simlab.config import DEFAULT_SCENARIO_MANAGER_API_URL


def test_tools_list_accepts_env_name(tmp_path: Path) -> None:
    env_dir = tmp_path / "environments" / "my-env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: my-env\nscenario_manager_api_url: https://example.invalid\n"
    )
    runner = CliRunner()
    with (
        patch("simlab.cli.tools.resolve_scenario_manager_api_url", return_value="https://api"),
        patch("simlab.cli.tools.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = [
            ScenarioSummary(
                scenario_id="x",
                name="X",
                tool_servers=[ScenarioToolServer(name="email-env", server_type="email")],
            )
        ]
        result = runner.invoke(
            tools,
            ["list", "--env", "my-env"],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )
    assert result.exit_code == 0, result.output
    assert "1 tools available from Scenario Manager API" in result.output


def test_tools_list_with_selected_config_does_not_inherit_default_config_url(
    tmp_path: Path,
) -> None:
    leaked_config = tmp_path / "default.toml"
    leaked_config.write_text(
        """
collinear_api_key = "leaked-key"
scenario_manager_api_url = "https://leaked.example.com"
"""
    )
    alt_config = tmp_path / "alt.toml"
    alt_config.write_text("")

    runner = CliRunner()
    with patch("simlab.cli.tools.ScenarioManagerClient") as mocked_client_cls:
        mocked_client_cls.return_value.list_scenarios.return_value = [
            ScenarioSummary(
                scenario_id="x",
                name="X",
                tool_servers=[ScenarioToolServer(name="email-env", server_type="email")],
            )
        ]
        result = runner.invoke(
            cli,
            ["--config-file", str(alt_config), "tools", "list"],
            env={
                "SIMLAB_CONFIG": str(leaked_config),
                "SIMLAB_COLLINEAR_API_KEY": "env-key",
            },
        )

    assert result.exit_code == 0, result.output
    mocked_client_cls.assert_called_once_with(
        base_url=DEFAULT_SCENARIO_MANAGER_API_URL,
        api_key="env-key",
    )
