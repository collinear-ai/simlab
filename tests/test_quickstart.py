"""Tests for the simlab quickstart command."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from click.testing import CliRunner
from simlab.api.schemas import ScenarioSummary
from simlab.api.schemas import ScenarioTask
from simlab.api.schemas import ScenarioTasksResponse
from simlab.cli.main import cli
from simlab.cli.quickstart import _find_task_by_id
from simlab.cli.quickstart import _select_default_task


def _fake_scenario(scenario_id: str = "hr-backend-id") -> ScenarioSummary:
    return ScenarioSummary(
        scenario_id=scenario_id,
        name="HR Tasks",
        description="HR scenario for tests.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_global_cfg(**overrides: str | bool | None) -> SimpleNamespace:
    defaults: dict[str, str | bool | None] = {
        "scenario_manager_api_url": "https://api.example.com",
        "api_key": "col_test",
        "collinear_api_key": "col_test",
        "daytona_api_key": None,
        "agent_model": None,
        "agent_provider": None,
        "agent_api_key": None,
        "agent_base_url": None,
        "verifier_model": None,
        "verifier_provider": None,
        "verifier_base_url": None,
        "verifier_api_key": None,
        "npc_chat_model": None,
        "npc_chat_provider": None,
        "npc_chat_base_url": None,
        "npc_chat_api_key": None,
        "tasks_rollout_format": None,
        "environments_dir": None,
        "telemetry_disabled": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_tasks() -> list[ScenarioTask]:
    return [
        ScenarioTask(
            task_id="200_medium_task",
            name="Medium Task",
            difficulty="medium",
        ),
        ScenarioTask(
            task_id="100_easy_task",
            name="Easy Task",
            difficulty="easy",
        ),
        ScenarioTask(
            task_id="300_hard_task",
            name="Hard Task",
            difficulty="hard",
        ),
    ]


def _make_task_response() -> ScenarioTasksResponse:
    return ScenarioTasksResponse(scenario_id="hr", tasks=_make_tasks())


def _mock_sm_client() -> MagicMock:
    mock_sm = MagicMock()
    mock_sm.resolve_template_to_backend_id.return_value = "hr-backend-id"
    mock_sm.list_scenario_tasks.return_value = _make_task_response()
    mock_sm.list_scenarios.return_value = []
    return mock_sm


@contextlib.contextmanager
def _quickstart_patched(env_dir: Path):
    """Apply all common patches for quickstart integration tests."""
    cfg = _fake_global_cfg()
    mock_sm = _mock_sm_client()
    with (
        # Auth gate patches (main.py)
        patch("simlab.cli.main.get_global_config_from_ctx", return_value=cfg),
        patch("simlab.cli.main._verify_key_with_server", return_value=True),
        patch(
            "simlab.cli.main.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        # Quickstart patches
        patch(
            "simlab.cli.quickstart.get_global_config_from_ctx",
            return_value=cfg,
        ),
        patch(
            "simlab.cli.quickstart.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        patch(
            "simlab.cli.quickstart.resolve_collinear_api_key",
            return_value="col_test",
        ),
        patch(
            "simlab.cli.quickstart.ScenarioManagerClient",
            return_value=mock_sm,
        ),
        patch("simlab.cli.quickstart.get_env_dir", return_value=env_dir),
        patch(
            "simlab.cli.quickstart.resolve_template_tools",
            return_value=("hr-backend-id", ["rocketchat", "email"], _fake_scenario()),
        ),
        patch("simlab.cli.quickstart.init_environment"),
        patch("simlab.cli.quickstart.subprocess.run", return_value=MagicMock(returncode=0)),
    ):
        yield


# ---------------------------------------------------------------------------
# _select_default_task
# ---------------------------------------------------------------------------


def test_select_default_task_picks_first_easy() -> None:
    tasks = _make_tasks()
    selected = _select_default_task(tasks)
    assert selected.task_id == "100_easy_task"


def test_select_default_task_picks_first_when_no_easy() -> None:
    tasks = [
        ScenarioTask(
            task_id="200_medium_task",
            name="Medium",
            difficulty="medium",
        ),
        ScenarioTask(
            task_id="300_hard_task",
            name="Hard",
            difficulty="hard",
        ),
    ]
    selected = _select_default_task(tasks)
    assert selected.task_id == "200_medium_task"


# ---------------------------------------------------------------------------
# _find_task_by_id
# ---------------------------------------------------------------------------


def test_find_task_exact_match() -> None:
    tasks = _make_tasks()
    result = _find_task_by_id(tasks, "100_easy_task")
    assert result is not None
    assert result.task_id == "100_easy_task"


def test_find_task_suffix_match() -> None:
    tasks = _make_tasks()
    result = _find_task_by_id(tasks, "easy_task")
    assert result is not None
    assert result.task_id == "100_easy_task"


def test_find_task_substring_match() -> None:
    tasks = _make_tasks()
    result = _find_task_by_id(tasks, "medium")
    assert result is not None
    assert result.task_id == "200_medium_task"


def test_find_task_not_found() -> None:
    tasks = _make_tasks()
    result = _find_task_by_id(tasks, "nonexistent")
    assert result is None


# ---------------------------------------------------------------------------
# quickstart command integration
# ---------------------------------------------------------------------------


def test_quickstart_defaults_to_hr_template(tmp_path: Path) -> None:
    env_dir = tmp_path / "environments" / "quickstart-hr"
    with _quickstart_patched(env_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["quickstart", "--agent-model", "gpt-4o", "--task", "100_easy_task"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert "quickstart-hr" in result.output
    assert "hr" in result.output


def test_quickstart_uses_custom_template(tmp_path: Path) -> None:
    env_dir = tmp_path / "environments" / "quickstart-coding"
    with _quickstart_patched(env_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quickstart",
                "--template",
                "coding",
                "--agent-model",
                "gpt-4o",
                "--task",
                "100_easy_task",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert "coding" in result.output


def test_quickstart_skips_init_if_env_exists(tmp_path: Path) -> None:
    env_dir = tmp_path / "environments" / "quickstart-hr"
    env_dir.mkdir(parents=True)
    (env_dir / "env.yaml").write_text("name: quickstart-hr\ntools: [email]\n")
    with _quickstart_patched(env_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["quickstart", "--agent-model", "gpt-4o", "--task", "100_easy_task"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert "Using existing environment" in result.output


def test_quickstart_validates_task_flag(tmp_path: Path) -> None:
    env_dir = tmp_path / "environments" / "quickstart-hr"
    with _quickstart_patched(env_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quickstart",
                "--agent-model",
                "gpt-4o",
                "--task",
                "nonexistent_task",
            ],
        )

    assert result.exit_code != 0
    assert "not found" in result.output


def test_quickstart_selects_easy_task_by_default(tmp_path: Path) -> None:
    """When no --task given and questionary unavailable, default to easy task."""
    env_dir = tmp_path / "environments" / "quickstart-hr"
    with (
        _quickstart_patched(env_dir),
        patch("simlab.cli.quickstart.questionary", None),
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["quickstart", "--agent-model", "gpt-4o"],
            input="\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert "100_easy_task" in result.output


def test_quickstart_forwards_environments_dir(tmp_path: Path) -> None:
    """Global --environments-dir is forwarded to the tasks run subprocess."""
    env_dir = tmp_path / "environments" / "quickstart-hr"
    cfg = _fake_global_cfg()
    mock_sm = _mock_sm_client()
    mock_subprocess = MagicMock(returncode=0)

    with (
        patch("simlab.cli.main.get_global_config_from_ctx", return_value=cfg),
        patch("simlab.cli.main._verify_key_with_server", return_value=True),
        patch(
            "simlab.cli.main.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        patch(
            "simlab.cli.quickstart.get_global_config_from_ctx",
            return_value=cfg,
        ),
        patch(
            "simlab.cli.quickstart.resolve_scenario_manager_api_url",
            return_value="https://api.example.com",
        ),
        patch(
            "simlab.cli.quickstart.resolve_collinear_api_key",
            return_value="col_test",
        ),
        patch(
            "simlab.cli.quickstart.ScenarioManagerClient",
            return_value=mock_sm,
        ),
        patch("simlab.cli.quickstart.get_env_dir", return_value=env_dir),
        patch(
            "simlab.cli.quickstart.resolve_template_tools",
            return_value=("hr-backend-id", ["email"], _fake_scenario()),
        ),
        patch("simlab.cli.quickstart.init_environment"),
        patch(
            "simlab.cli.quickstart.subprocess.run",
            return_value=mock_subprocess,
        ) as mock_run,
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--environments-dir",
                "/custom/envs",
                "quickstart",
                "--agent-model",
                "gpt-4o",
                "--task",
                "100_easy_task",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    call_args = mock_run.call_args[0][0]
    assert "--environments-dir" in call_args
    idx = call_args.index("--environments-dir")
    assert call_args[idx + 1] == "/custom/envs"
