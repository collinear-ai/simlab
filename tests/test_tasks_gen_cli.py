from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from simlab.api.schemas import TaskBundleFile
from simlab.api.schemas import TaskGenJob
from simlab.api.schemas import TaskGenResult
from simlab.cli.main import cli
from simlab.cli.tasks_gen import _write_bundle
from simlab.cli.tasks_gen import tasks_gen


def test_tasks_gen_status_uses_global_config_for_client() -> None:
    runner = CliRunner()
    with (
        patch(
            "simlab.cli.tasks_gen.resolve_scenario_manager_api_url",
            return_value="https://cfg-api.example.com",
        ),
        patch("simlab.cli.tasks_gen.resolve_collinear_api_key", return_value="cfg-token"),
        patch("simlab.cli.tasks_gen.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.get_task_gen_status.return_value = TaskGenJob(
            job_id="job-123",
            status="completed",
        )
        result = runner.invoke(tasks_gen, ["status", "job-123"])

    assert result.exit_code == 0, result.output
    mocked_client_cls.assert_called_once_with(
        base_url="https://cfg-api.example.com",
        api_key="cfg-token",
    )


def test_tasks_gen_status_root_api_key_beats_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "bad-env")

    with (
        patch("simlab.cli.main._verify_key_with_server", return_value=True),
        patch(
            "simlab.cli.tasks_gen.resolve_scenario_manager_api_url",
            return_value="https://cfg-api.example.com",
        ),
        patch("simlab.cli.tasks_gen.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.get_task_gen_status.return_value = TaskGenJob(
            job_id="job-123",
            status="completed",
        )
        result = runner.invoke(
            cli,
            ["--collinear-api-key", "good-root", "tasks-gen", "status", "job-123"],
        )

    assert result.exit_code == 0, result.output
    mocked_client_cls.assert_called_once_with(
        base_url="https://cfg-api.example.com",
        api_key="good-root",
    )


def test_tasks_gen_status_env_api_key_is_used_without_root_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "env-key")

    with (
        patch("simlab.cli.main._verify_key_with_server", return_value=True),
        patch(
            "simlab.cli.tasks_gen.resolve_scenario_manager_api_url",
            return_value="https://cfg-api.example.com",
        ),
        patch("simlab.cli.tasks_gen.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.get_task_gen_status.return_value = TaskGenJob(
            job_id="job-123",
            status="completed",
        )
        result = runner.invoke(cli, ["tasks-gen", "status", "job-123"])

    assert result.exit_code == 0, result.output
    mocked_client_cls.assert_called_once_with(
        base_url="https://cfg-api.example.com",
        api_key="env-key",
    )


def test_tasks_gen_status_root_scenario_manager_url_beats_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://bad-env.example.com")

    with (
        patch("simlab.cli.main._verify_key_with_server", return_value=True),
        patch("simlab.cli.tasks_gen.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.get_task_gen_status.return_value = TaskGenJob(
            job_id="job-123",
            status="completed",
        )
        result = runner.invoke(
            cli,
            [
                "--collinear-api-key",
                "root-key",
                "--scenario-manager-api-url",
                "https://root-override.example.com",
                "tasks-gen",
                "status",
                "job-123",
            ],
        )

    assert result.exit_code == 0, result.output
    mocked_client_cls.assert_called_once_with(
        base_url="https://root-override.example.com",
        api_key="root-key",
    )


def test_write_bundle_includes_npc_profiles(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        TaskGenResult(
            job_id="job-123",
            npcs=[
                TaskBundleFile(
                    filename="profiles.json",
                    content='{"npc": {"email": "npc@example.com"}}',
                )
            ],
        ),
    )

    assert (tmp_path / "npcs" / "profiles.json").read_text(encoding="utf-8") == (
        '{"npc": {"email": "npc@example.com"}}'
    )
