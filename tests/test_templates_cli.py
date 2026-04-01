from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock
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
    client = Mock()
    client.list_scenarios.return_value = fake_scenarios
    with patch("simlab.runtime.templates.build_scenario_manager_client", return_value=client):
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
    client = Mock()
    client.list_scenarios.return_value = []
    with patch(
        "simlab.runtime.templates.build_scenario_manager_client",
        return_value=client,
    ) as mocked_build_client:
        result = runner.invoke(
            templates,
            ["list", "--env", "my-env"],
            env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
        )
    assert mocked_build_client.call_args.kwargs["env_name"] == "my-env"
    assert result.exit_code == 0, result.output
    assert "0 templates available" in result.output


def test_templates_list_wraps_long_names_and_descriptions() -> None:
    fake_scenarios = [
        ScenarioSummary(
            scenario_id="enterprise_customer_operations",
            name=("Enterprise Customer Operations and Multi Region Escalation Management"),
            description=(
                "Coordinate a multi-team escalation that spans support, billing, "
                "success, and product operations while preserving the complete "
                "handoff narrative for the next reviewer."
            ),
            tool_servers=[
                ScenarioToolServer(name="crm"),
                ScenarioToolServer(name="email"),
                ScenarioToolServer(name="calendar"),
            ],
        )
    ]
    runner = CliRunner()
    client = Mock()
    client.list_scenarios.return_value = fake_scenarios
    with (
        patch("simlab.runtime.templates.build_scenario_manager_client", return_value=client),
        patch(
            "simlab.cli.templates.shutil.get_terminal_size",
            return_value=os.terminal_size((80, 20)),
        ),
    ):
        result = runner.invoke(templates, ["list"])

    assert result.exit_code == 0, result.output
    assert "Name: Enterprise Customer Operations and Multi Region Escalation" in result.output
    assert "Management" in result.output
    assert "Description: Coordinate a multi-team escalation" in result.output
    assert "complete" in result.output
    assert "handoff narrative for the next reviewer." in result.output
    assert "Tools: crm, email, calendar" in result.output
    assert "…" not in result.output
