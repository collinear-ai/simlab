from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import simlab.cli.tasks as tasks_cli
from click.testing import CliRunner
from simlab.cli.rollout_summary_card import parallel_rollout_status
from simlab.cli.rollout_summary_card import print_single_rollout_summary_card
from simlab.cli.tasks import tasks
from simlab.composer.engine import EnvConfig


def write_env_dir(tmp_path: Path, env_name: str = "my-env") -> Path:
    env_dir = tmp_path / "environments" / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text("name: my-env\n", encoding="utf-8")
    return env_dir


def write_task_bundle(tmp_path: Path, task_id: str = "task-a") -> Path:
    bundle_dir = tmp_path / "tasks-bundle"
    (bundle_dir / "tasks").mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "task_id": task_id,
            "display_name": "Task A",
            "difficulty": "easy",
        },
        "task": "Do the task.",
        "tool_servers": [],
        "verifiers": [],
    }
    (bundle_dir / "tasks" / f"{task_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return bundle_dir


def resolve_agent_runtime_settings_stub(
    *_args: object, **_kwargs: object
) -> tuple[str, str, str, None]:
    return ("gpt-5.2", "openai", "test-key", None)


def test_tasks_run_prints_summary_card_to_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_env_dir(tmp_path)
    bundle_dir = write_task_bundle(tmp_path)

    monkeypatch.setattr(tasks_cli, "ensure_env_artifacts_current", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "load_env_config", lambda *args, **kwargs: EnvConfig(tools=[]))
    monkeypatch.setattr(
        tasks_cli,
        "get_global_config_from_ctx",
        lambda *args, **kwargs: SimpleNamespace(daytona_api_key=None),
    )
    monkeypatch.setattr(
        tasks_cli,
        "_resolve_agent_runtime_settings",
        resolve_agent_runtime_settings_stub,
    )
    monkeypatch.setattr(
        tasks_cli, "resolve_scenario_manager_api_url", lambda *args, **kwargs: "https://api"
    )
    monkeypatch.setattr(tasks_cli, "resolve_collinear_api_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "env_has_local_services", lambda *args, **kwargs: False)

    output_dir = tmp_path / "output" / "agent_run_task-a_20260101_000000"
    verifier_results = [
        {
            "module": "collinear.scenarios.hr.verifiers.example",
            "success": True,
            "message": "",
            "output": json.dumps(
                {
                    "checks": [
                        {"name": "seed_email_read", "passed": True},
                        {"name": "hris_lookup", "passed": True},
                    ]
                }
            ),
        },
        {
            "module": "collinear.scenarios.hr.verifiers.failing_text_verifier",
            "success": False,
            "message": "verifier failed",
            "output": "not json",
        },
    ]
    outcome = tasks_cli.SingleRolloutOutcome(
        task_id="task-a",
        model="gpt-5.2",
        provider="openai",
        steps_taken=12,
        max_steps=30,
        reward=1.0,
        verification_passed=True,
        run_error=None,
        verifier_results=verifier_results,
        output_dir=output_dir,
        exit_code=0,
    )
    monkeypatch.setattr(tasks_cli, "_run_single_rollout", lambda **kwargs: outcome)

    runner = CliRunner()
    result = runner.invoke(
        tasks,
        [
            "run",
            "--env",
            "my-env",
            "--tasks-dir",
            str(bundle_dir),
            "--task",
            "task-a",
        ],
        env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
    )

    assert result.exit_code == 0, result.output
    assert "Rollout Summary" not in result.stdout
    assert "Rollout Summary" in result.stderr
    assert "PASS" in result.stderr
    assert "failing_text_verifier" in result.stderr
    assert "seed_email_read" in result.stderr
    assert "hris_lookup" in result.stderr
    assert "Next: simlab eval" in result.stderr


def test_tasks_run_quiet_suppresses_summary_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_env_dir(tmp_path)
    bundle_dir = write_task_bundle(tmp_path)

    monkeypatch.setattr(tasks_cli, "ensure_env_artifacts_current", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "load_env_config", lambda *args, **kwargs: EnvConfig(tools=[]))
    monkeypatch.setattr(
        tasks_cli,
        "get_global_config_from_ctx",
        lambda *args, **kwargs: SimpleNamespace(daytona_api_key=None),
    )
    monkeypatch.setattr(
        tasks_cli,
        "_resolve_agent_runtime_settings",
        resolve_agent_runtime_settings_stub,
    )
    monkeypatch.setattr(
        tasks_cli, "resolve_scenario_manager_api_url", lambda *args, **kwargs: "https://api"
    )
    monkeypatch.setattr(tasks_cli, "resolve_collinear_api_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "env_has_local_services", lambda *args, **kwargs: False)

    outcome = tasks_cli.SingleRolloutOutcome(
        task_id="task-a",
        model="gpt-5.2",
        provider="openai",
        steps_taken=12,
        max_steps=30,
        reward=1.0,
        verification_passed=True,
        run_error=None,
        verifier_results=[],
        output_dir=tmp_path / "output" / "agent_run_task-a_20260101_000000",
        exit_code=0,
    )
    monkeypatch.setattr(tasks_cli, "_run_single_rollout", lambda **kwargs: outcome)

    runner = CliRunner()
    result = runner.invoke(
        tasks,
        [
            "run",
            "--env",
            "my-env",
            "--tasks-dir",
            str(bundle_dir),
            "--task",
            "task-a",
            "--quiet",
        ],
        env={"SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments")},
    )

    assert result.exit_code == 0, result.output
    assert "Rollout Summary" not in result.stderr
    assert "Rollout Summary" not in result.stdout


def test_parallel_rollout_status_shows_na_when_unverified() -> None:
    status = parallel_rollout_status(SimpleNamespace(error=None, verification_passed=None))
    assert status.plain == "N/A"


def test_summary_card_default_console_writes_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "output" / "agent_run_task-a_20260101_000000"
    verifier_results = [
        {
            "module": "collinear.scenarios.hr.verifiers.example",
            "success": True,
            "message": "",
            "output": json.dumps(
                {
                    "checks": [
                        {"name": "seed_email_read", "passed": True},
                        {"name": "hris_lookup", "passed": True},
                    ]
                }
            ),
        }
    ]

    print_single_rollout_summary_card(
        task_id="task-a",
        model="gpt-5.2",
        provider="openai",
        steps_taken=12,
        max_steps=30,
        duration_seconds=42.0,
        reward=1.0,
        verification_passed=True,
        run_error=None,
        verifier_results=verifier_results,
        output_dir=output_dir,
        quiet=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Rollout Summary" in captured.err
