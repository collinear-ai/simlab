from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from simlab.api.schemas import ScenarioSummary
from simlab.api.schemas import ScenarioToolServer
from simlab.cli.main import cli


def test_cli_blocks_non_config_commands_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    monkeypatch.setenv("SIMLAB_CONFIG", str(tmp_path / "missing-config.toml"))

    runner = CliRunner()
    with patch("simlab.runtime.templates.ScenarioManagerClient") as mocked_client_cls:
        result = runner.invoke(cli, ["templates", "list"])

    assert result.exit_code == 1
    assert "Error: API key required. Run: simlab auth login" in result.output
    assert "Get your key at https://platform.collinear.ai" in result.output
    mocked_client_cls.assert_not_called()


def test_cli_root_api_key_unblocks_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    monkeypatch.setenv("SIMLAB_CONFIG", str(tmp_path / "missing-config.toml"))

    runner = CliRunner()
    with (
        patch("simlab.cli.main._verify_key_with_server", return_value=True),
        patch("simlab.runtime.templates.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = [
            ScenarioSummary(
                scenario_id="customer_support",
                name="Customer Support",
                tool_servers=[ScenarioToolServer(name="rocketchat")],
            )
        ]
        result = runner.invoke(
            cli,
            [
                "--collinear-api-key",
                "ck_test",
                "--scenario-manager-api-url",
                "https://api.example.com",
                "templates",
                "list",
            ],
        )

    assert result.exit_code == 0, result.output
    mocked_client_cls.assert_called_once_with(
        base_url="https://api.example.com",
        api_key="ck_test",
    )


def test_cli_config_file_api_key_unblocks_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    config_file = tmp_path / "config.toml"
    config_file.write_text('collinear_api_key = "file-key"\n', encoding="utf-8")

    runner = CliRunner()
    with (
        patch("simlab.cli.main._verify_key_with_server", return_value=True),
        patch("simlab.runtime.templates.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = [
            ScenarioSummary(
                scenario_id="customer_support",
                name="Customer Support",
                tool_servers=[ScenarioToolServer(name="rocketchat")],
            )
        ]
        result = runner.invoke(
            cli,
            [
                "--config-file",
                str(config_file),
                "--scenario-manager-api-url",
                "https://api.example.com",
                "templates",
                "list",
            ],
        )

    assert result.exit_code == 0, result.output
    mocked_client_cls.assert_called_once_with(
        base_url="https://api.example.com",
        api_key="file-key",
    )


def test_cli_help_and_version_work_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    monkeypatch.setenv("SIMLAB_CONFIG", str(tmp_path / "missing-config.toml"))

    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    version_result = runner.invoke(cli, ["--version"])

    assert help_result.exit_code == 0, help_result.output
    # The branded help screen renders via Rich rather than Click's default
    # formatter, so we check invariants of the new output: the tagline and
    # at least one command name from the journey-ordered table.
    assert "hill-climbing" in help_result.output
    assert "auth" in help_result.output
    assert version_result.exit_code == 0, version_result.output
    assert "version" in version_result.output.lower()


@pytest.mark.parametrize(
    "args",
    [
        ["eval", "--help"],
        ["env", "--help"],
        ["tasks", "--help"],
        ["tasks", "run", "--help"],
        ["tasks-gen", "--help"],
        ["tasks-gen", "status", "--help"],
        ["templates", "--help"],
        ["templates", "list", "--help"],
        ["tools", "--help"],
        ["tools", "list", "--help"],
    ],
)
def test_cli_subcommand_help_works_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    args: list[str],
) -> None:
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.delenv("COLLINEAR_API_KEY", raising=False)
    monkeypatch.setenv("SIMLAB_CONFIG", str(tmp_path / "missing-config.toml"))

    runner = CliRunner()
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
