from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from simlab.cli.runs import runs_history


def write_single_run(
    run_dir: Path,
    *,
    task_id: str,
    model: str,
    provider: str,
    created_at: str,
    reward: float | None,
    duration_seconds: float | None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {}
    if duration_seconds is not None:
        metadata["rollout_metrics"] = {
            "timing": {"duration_seconds": duration_seconds},
        }

    artifacts = {
        "task_id": task_id,
        "model": model,
        "provider": provider,
        "created_at": created_at,
        "metadata": metadata,
        "error": None,
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")

    if reward is None:
        return

    verifier_dir = run_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "reward.json").write_text(
        json.dumps({"reward": reward}),
        encoding="utf-8",
    )


def write_single_atif_run(
    run_dir: Path,
    *,
    task_id: str,
    model: str,
    provider: str,
    created_at: str,
    reward: float | None,
    duration_seconds: float | None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {}
    if duration_seconds is not None:
        metadata["rollout_metrics"] = {
            "timing": {"duration_seconds": duration_seconds},
        }

    trajectory = {
        "schema_version": "ATIF-v1.4",
        "session_id": run_dir.name,
        "agent": {
            "name": "simlab-reference-agent",
            "version": "0.1",
            "model_name": model,
            "extra": {"provider": provider},
        },
        "steps": [],
        "extra": {
            "simlab": {
                "task_id": task_id,
                "created_at": created_at,
                "metadata": metadata,
            }
        },
    }
    agent_dir = run_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    if reward is None:
        return

    verifier_dir = run_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "reward.json").write_text(
        json.dumps({"reward": reward}),
        encoding="utf-8",
    )


def write_parallel_run(
    run_set_dir: Path,
    *,
    task_id: str,
    model: str,
    provider: str,
    rollout_count: int,
    passed: int,
    failed: int,
    total_duration_seconds: float,
) -> None:
    run_set_dir.mkdir(parents=True, exist_ok=True)
    (run_set_dir / "summary.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "rollout_count": rollout_count,
                "passed": passed,
                "failed": failed,
                "total_duration_seconds": total_duration_seconds,
                "results": [],
            }
        ),
        encoding="utf-8",
    )

    rollout_dir = run_set_dir / "rollout_0"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "model": model,
                "provider": provider,
                "created_at": "2026-03-27T13:15:00Z",
                "metadata": {},
                "error": None,
            }
        ),
        encoding="utf-8",
    )


def write_parallel_atif_run(
    run_set_dir: Path,
    *,
    task_id: str,
    model: str,
    provider: str,
    rollout_count: int,
    passed: int,
    failed: int,
    total_duration_seconds: float,
) -> None:
    run_set_dir.mkdir(parents=True, exist_ok=True)
    (run_set_dir / "summary.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "rollout_count": rollout_count,
                "passed": passed,
                "failed": failed,
                "total_duration_seconds": total_duration_seconds,
                "results": [],
            }
        ),
        encoding="utf-8",
    )

    rollout_dir = run_set_dir / "rollout_0"
    agent_dir = rollout_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.4",
                "session_id": rollout_dir.name,
                "agent": {
                    "name": "simlab-reference-agent",
                    "version": "0.1",
                    "model_name": model,
                    "extra": {"provider": provider},
                },
                "steps": [],
                "extra": {
                    "simlab": {
                        "task_id": task_id,
                        "created_at": "2026-03-27T13:15:00Z",
                        "metadata": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_runs_history_json_output_supports_filters_and_sorting(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    write_single_run(
        output_dir / "agent_run_task_a_20260327_143200",
        task_id="task-a",
        model="gpt-5.2",
        provider="openai",
        created_at="2026-03-27T14:32:00Z",
        reward=0.0,
        duration_seconds=151.0,
    )
    write_single_run(
        output_dir / "agent_run_task_b_20260327_142800",
        task_id="task-b",
        model="gpt-5.2",
        provider="openai",
        created_at="2026-03-27T14:28:00Z",
        reward=1.0,
        duration_seconds=72.0,
    )
    write_single_run(
        output_dir / "agent_run_task_c_20260326_213000",
        task_id="task-c",
        model="gpt-4o",
        provider="openai",
        created_at="2026-03-26T21:30:00Z",
        reward=1.0,
        duration_seconds=175.0,
    )
    write_parallel_run(
        output_dir / "parallel_run_task_d_20260327_131500",
        task_id="task-d",
        model="claude-sonnet",
        provider="anthropic",
        rollout_count=3,
        passed=3,
        failed=0,
        total_duration_seconds=250.0,
    )

    runner = CliRunner()
    result = runner.invoke(runs_history, ["--output-dir", str(output_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 4
    assert [row["task_id"] for row in payload["runs"]] == ["task-a", "task-b", "task-d", "task-c"]

    result = runner.invoke(runs_history, ["--output-dir", str(output_dir), "--json", "--last", "2"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 2
    assert [row["task_id"] for row in payload["runs"]] == ["task-a", "task-b"]

    result = runner.invoke(
        runs_history,
        ["--output-dir", str(output_dir), "--json", "--model", "gpt-5.2"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 2
    assert {row["task_id"] for row in payload["runs"]} == {"task-a", "task-b"}

    result = runner.invoke(
        runs_history, ["--output-dir", str(output_dir), "--json", "--task", "task-b"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["runs"][0]["task_id"] == "task-b"

    result = runner.invoke(
        runs_history, ["--output-dir", str(output_dir), "--json", "--task", "TASK-B"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["runs"][0]["task_id"] == "task-b"

    result = runner.invoke(
        runs_history, ["--output-dir", str(output_dir), "--json", "--result", "pass"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 3
    assert {row["task_id"] for row in payload["runs"]} == {"task-b", "task-c", "task-d"}

    result = runner.invoke(
        runs_history, ["--output-dir", str(output_dir), "--json", "--result", "fail"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["runs"][0]["task_id"] == "task-a"


def test_runs_history_prefers_single_run_over_rollout_subdir(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    run_dir = output_dir / "agent_run_shadow_20260327_140000"
    write_single_run(
        run_dir,
        task_id="shadow-task",
        model="gpt-5.2",
        provider="openai",
        created_at="2026-03-27T14:00:00Z",
        reward=1.0,
        duration_seconds=12.0,
    )
    rollout_dir = run_dir / "rollout_0"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "artifacts.json").write_text(
        json.dumps({"task_id": "ignored", "model": "gpt-4o", "provider": "openai"}),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(runs_history, ["--output-dir", str(output_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["runs"][0]["run_type"] == "single"
    assert payload["runs"][0]["task_id"] == "shadow-task"


def test_runs_history_ignores_bool_reward_payloads(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    run_dir = output_dir / "agent_run_bool_reward_20260327_150000"
    write_single_run(
        run_dir,
        task_id="bool-reward",
        model="gpt-5.2",
        provider="openai",
        created_at="2026-03-27T15:00:00Z",
        reward=None,
        duration_seconds=5.0,
    )
    verifier_dir = run_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "reward.json").write_text(json.dumps({"reward": True}), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(runs_history, ["--output-dir", str(output_dir), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["runs"][0]["result"] == "unknown"


def test_runs_history_supports_atif_only_single_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    write_single_atif_run(
        output_dir / "agent_run_atif_task_20260327_160000",
        task_id="atif-task",
        model="gpt-5.2",
        provider="openai",
        created_at="2026-03-27T16:00:00Z",
        reward=1.0,
        duration_seconds=8.0,
    )

    runner = CliRunner()
    result = runner.invoke(runs_history, ["--output-dir", str(output_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["runs"][0]["run_type"] == "single"
    assert payload["runs"][0]["task_id"] == "atif-task"
    assert payload["runs"][0]["model"] == "gpt-5.2"
    assert payload["runs"][0]["provider"] == "openai"
    assert payload["runs"][0]["result"] == "pass"


def test_runs_history_prefers_atif_single_run_over_rollout_subdir(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    run_dir = output_dir / "agent_run_shadow_atif_20260327_141000"
    write_single_atif_run(
        run_dir,
        task_id="shadow-atif-task",
        model="gpt-5.2",
        provider="openai",
        created_at="2026-03-27T14:10:00Z",
        reward=1.0,
        duration_seconds=12.0,
    )
    rollout_dir = run_dir / "rollout_0"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "artifacts.json").write_text(
        json.dumps({"task_id": "ignored", "model": "gpt-4o", "provider": "openai"}),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(runs_history, ["--output-dir", str(output_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["runs"][0]["run_type"] == "single"
    assert payload["runs"][0]["task_id"] == "shadow-atif-task"


def test_runs_history_parallel_run_model_provider_supports_atif_rollouts(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    write_parallel_atif_run(
        output_dir / "parallel_run_task_atif_20260327_131500",
        task_id="task-atif",
        model="claude-sonnet",
        provider="anthropic",
        rollout_count=3,
        passed=3,
        failed=0,
        total_duration_seconds=250.0,
    )

    runner = CliRunner()
    result = runner.invoke(runs_history, ["--output-dir", str(output_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["runs"][0]["run_type"] == "parallel"
    assert payload["runs"][0]["task_id"] == "task-atif"
    assert payload["runs"][0]["model"] == "claude-sonnet"
    assert payload["runs"][0]["provider"] == "anthropic"
