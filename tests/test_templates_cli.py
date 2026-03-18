from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from simlab.api.schemas import ScenarioSummary
from simlab.api.schemas import ScenarioToolServer
from simlab.cli.templates import templates


def test_templates_list_shows_slug() -> None:
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="customer_support",
            name="Customer Support",
            description="visible",
            tool_servers=[ScenarioToolServer(name="rocketchat")],
        )
    ]
    runner = CliRunner()
    with (
        patch(
            "simlab.cli.templates.resolve_scenario_manager_api_url",
            return_value="https://api",
        ),
        patch("simlab.cli.templates.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = fake_scenarios
        result = runner.invoke(templates, ["list"])
    assert result.exit_code == 0, result.output
    assert "Customer Support" in result.output
    assert "customer_support" in result.output
    assert "1 templates available" in result.output


def test_templates_list_accepts_env_name(tmp_path: Path) -> None:
    env_dir = tmp_path / "environments" / "my-env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: my-env\nscenario_manager_api_url: https://example.invalid\n"
    )
    runner = CliRunner()
    with (
        patch(
            "simlab.cli.templates.resolve_scenario_manager_api_url",
            return_value="https://api",
        ),
        patch("simlab.cli.templates.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = []
        result = runner.invoke(
            templates,
            ["list", "--env", "my-env"],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )
    assert result.exit_code == 0, result.output
    assert "0 templates available" in result.output
