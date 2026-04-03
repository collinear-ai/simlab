from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from simlab.cli.eval import eval_command
from simlab.cli.main import cli
from simlab.evaluation import build_tool_calls
from simlab.evaluation import detect_cofailure_patterns
from simlab.evaluation import summarize_score_summary


def write_atif_rollout(
    rollout_dir: Path,
    *,
    task_id: str,
    model: str,
    reward: float | None,
    criteria: list[dict[str, object]] | None = None,
    reward_model: dict[str, object] | None = None,
    duration_seconds: float | None = None,
    estimated_cost_usd: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    tool_calls: list[dict[str, object]] | None = None,
    tool_results: list[dict[str, object]] | None = None,
    error: str | None = None,
    include_rollout_metrics: bool = True,
    agent_step_timestamp: str = "2026-03-24T20:00:01Z",
) -> None:
    """Write an ATIF rollout fixture directory for eval CLI tests."""
    rollout_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {}
    include_metrics = any(
        value is not None
        for value in (
            duration_seconds,
            estimated_cost_usd,
            prompt_tokens,
            completion_tokens,
        )
    )
    if include_metrics and include_rollout_metrics:
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        token_usage: dict[str, int | None] = {
            "prompt_tokens_total": prompt_tokens,
            "completion_tokens_total": completion_tokens,
        }
        rollout_metrics = {
            "token_usage": token_usage,
            "timing": {
                "duration_seconds": duration_seconds,
            },
            "cost": {
                "estimated_cost_usd": estimated_cost_usd,
            },
        }
        if total_tokens is not None:
            token_usage["total_tokens_total"] = total_tokens
        metadata["rollout_metrics"] = rollout_metrics

    steps: list[dict[str, object]] = [
        {
            "step_id": 1,
            "timestamp": "2026-03-24T20:00:00Z",
            "source": "user",
            "message": f"Run {task_id}",
        }
    ]
    for index, call in enumerate(tool_calls or [], start=1):
        result = (tool_results or [])[index - 1] if index - 1 < len(tool_results or []) else {}
        observation = result.get("observation") if isinstance(result, dict) else None
        content = None
        if isinstance(observation, dict):
            content = observation.get("text")
        elif isinstance(observation, str):
            content = observation
        step: dict[str, object] = {
            "step_id": len(steps) + 1,
            "timestamp": agent_step_timestamp,
            "source": "agent",
            "message": "Tool call",
            "model_name": model,
            "tool_calls": [
                {
                    "tool_call_id": f"call_{index}",
                    "function_name": call.get("tool_name") if isinstance(call, dict) else "tool",
                    "arguments": (call.get("parameters") if isinstance(call, dict) else {}),
                    "extra": {
                        "tool_server": (
                            call.get("tool_server") if isinstance(call, dict) else "unknown"
                        )
                    },
                }
            ],
        }
        if content is not None:
            step["observation"] = {
                "results": [
                    {
                        "source_call_id": f"call_{index}",
                        "content": str(content),
                        "extra": {
                            "tool_server": (
                                call.get("tool_server") if isinstance(call, dict) else "unknown"
                            ),
                            "tool_name": (
                                call.get("tool_name") if isinstance(call, dict) else "tool"
                            ),
                            "is_error": bool(result.get("is_error"))
                            if isinstance(result, dict) and "is_error" in result
                            else False,
                            "raw_observation": observation,
                        },
                    }
                ]
            }
        steps.append(step)

    trajectory = {
        "schema_version": "ATIF-v1.4",
        "session_id": rollout_dir.name,
        "agent": {
            "name": "simlab-reference-agent",
            "version": "0.1",
            "model_name": model,
            "extra": {"provider": "openai"},
        },
        "steps": steps,
        "extra": {
            "simlab": {
                "task_id": task_id,
                "task": f"Complete {task_id}",
                "created_at": "2026-03-24T20:00:00Z",
                "steps_taken": len(tool_calls or []),
                "max_steps": 30,
                "final_observation": "Done",
                "run_error": error,
                "metadata": metadata,
            }
        },
    }
    if include_metrics:
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        final_metrics: dict[str, object] = {"total_steps": len(steps)}
        if prompt_tokens is not None:
            final_metrics["total_prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            final_metrics["total_completion_tokens"] = completion_tokens
        if total_tokens is not None:
            final_metrics["total_tokens"] = total_tokens
        if estimated_cost_usd is not None:
            final_metrics["total_cost_usd"] = estimated_cost_usd
        trajectory["final_metrics"] = final_metrics
    (rollout_dir / "agent" / "trajectory.json").parent.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    if reward is None and criteria is None and reward_model is None:
        return

    verifier_dir = rollout_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    verifier_results = []
    if criteria is not None:
        verifier_results.append(
            {
                "module": f"collinear.scenarios.demo.verifiers.{task_id}",
                "success": all(bool(item.get("pass")) for item in criteria),
                "message": "",
                "output": json.dumps(criteria),
            }
        )
    if reward_model is not None:
        reward_model_score = reward_model.get("score")
        if not isinstance(reward_model_score, (int, float)):
            reward_model_score = 0.0
        verifier_results.append(
            {
                "module": "collinear.scenarios.demo.verifiers.universal_verifier",
                "success": bool(reward_model_score >= 0.6),
                "message": "",
                "output": json.dumps(reward_model),
            }
        )
    reward_payload = {
        "reward": reward,
        "verifier_results": verifier_results,
    }
    (verifier_dir / "reward.json").write_text(json.dumps(reward_payload), encoding="utf-8")


def write_rollout(
    rollout_dir: Path,
    *,
    task_id: str,
    model: str,
    reward: float | None,
    criteria: list[dict[str, object]] | None = None,
    reward_model: dict[str, object] | None = None,
    duration_seconds: float | None = None,
    estimated_cost_usd: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    tool_calls: list[dict[str, object]] | None = None,
    tool_results: list[dict[str, object]] | None = None,
    error: str | None = None,
) -> None:
    """Write a rollout fixture directory for eval CLI tests."""
    rollout_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {}
    if any(
        value is not None
        for value in (
            duration_seconds,
            estimated_cost_usd,
            prompt_tokens,
            completion_tokens,
        )
    ):
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        token_usage: dict[str, int | None] = {
            "prompt_tokens_total": prompt_tokens,
            "completion_tokens_total": completion_tokens,
        }
        rollout_metrics = {
            "token_usage": token_usage,
            "timing": {
                "duration_seconds": duration_seconds,
            },
            "cost": {
                "estimated_cost_usd": estimated_cost_usd,
            },
        }
        if total_tokens is not None:
            token_usage["total_tokens_total"] = total_tokens
        metadata["rollout_metrics"] = rollout_metrics

    artifacts = {
        "task_id": task_id,
        "task": f"Complete {task_id}",
        "model": model,
        "provider": "openai",
        "messages": [
            {"role": "user", "content": f"Run {task_id}"},
            {"role": "assistant", "content": "Done"},
        ],
        "tool_calls": tool_calls or [],
        "tool_results": tool_results or [],
        "metadata": metadata,
        "final_observation": "Done",
        "error": error,
        "steps_taken": len(tool_calls or []),
        "max_steps": 30,
        "created_at": "2026-03-24T20:00:00Z",
    }
    (rollout_dir / "artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")

    if reward is None and criteria is None and reward_model is None:
        return

    verifier_dir = rollout_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    verifier_results = []
    if criteria is not None:
        verifier_results.append(
            {
                "module": f"collinear.scenarios.demo.verifiers.{task_id}",
                "success": all(bool(item.get("pass")) for item in criteria),
                "message": "",
                "output": json.dumps(criteria),
            }
        )
    if reward_model is not None:
        reward_model_score = reward_model.get("score")
        if not isinstance(reward_model_score, (int, float)):
            reward_model_score = 0.0
        verifier_results.append(
            {
                "module": "collinear.scenarios.demo.verifiers.universal_verifier",
                "success": bool(reward_model_score >= 0.6),
                "message": "",
                "output": json.dumps(reward_model),
            }
        )
    reward_payload = {
        "reward": reward,
        "verifier_results": verifier_results,
    }
    (verifier_dir / "reward.json").write_text(json.dumps(reward_payload), encoding="utf-8")


def test_main_cli_eval_requires_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    write_rollout(output_dir / "run_a", task_id="task-a", model="gpt-4o-mini", reward=1.0)

    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.setenv("SIMLAB_CONFIG", str(tmp_path / "missing-config.toml"))

    runner = CliRunner()
    result = runner.invoke(cli, ["eval", str(output_dir), "--json"])

    assert result.exit_code == 1
    assert "Error: API key required. Run: simlab auth login" in result.output


def test_eval_defaults_to_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_rollout(tmp_path / "output" / "run_a", task_id="task-a", model="gpt-4o-mini", reward=1.0)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(eval_command, ["--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["path"] == str((tmp_path / "output").resolve())
    assert payload["summary"]["rollout_count"] == 1


def test_eval_run_set_treats_null_verification_passed_without_error_as_passed(
    tmp_path: Path,
) -> None:
    run_set_dir = tmp_path / "parallel_run"
    write_rollout(
        run_set_dir / "rollout_0",
        task_id="task-a",
        model="gpt-4o-mini",
        reward=None,
    )
    (run_set_dir / "summary.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "rollout_idx": 0,
                        "verification_passed": None,
                        "duration": 12.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(eval_command, [str(run_set_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["passed_count"] == 1
    assert payload["summary"]["pass_rate"] == pytest.approx(1.0)


def test_eval_single_rollout_json_includes_criteria_tool_calls_and_reward_model(
    tmp_path: Path,
) -> None:
    rollout_dir = tmp_path / "agent_run_task_a"
    write_rollout(
        rollout_dir,
        task_id="task-a",
        model="gpt-4o-mini",
        reward=0.0,
        criteria=[
            {"criteria": "meeting_created", "pass": False},
            {"criteria": "scope_discipline", "pass": True},
        ],
        reward_model={
            "score": 0.45,
            "confidence": 0.81,
            "verdict": "partial",
            "failed_criteria": ["meeting_created"],
            "dimension_scores": [
                {"dimension": "Calendar Event Update", "score": 0.2, "reason": "No event created"},
                {"dimension": "Scope Discipline", "score": 0.7, "reason": "No extra actions"},
            ],
        },
        duration_seconds=42.5,
        estimated_cost_usd=0.12,
        prompt_tokens=1200,
        completion_tokens=300,
        tool_calls=[
            {
                "tool_server": "calendar",
                "tool_name": "create_event",
                "parameters": {"title": "Interview"},
            }
        ],
        tool_results=[
            {
                "observation": {"text": "No event created"},
                "is_error": True,
            }
        ],
    )

    runner = CliRunner()
    result = runner.invoke(eval_command, [str(rollout_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "single_rollout"
    assert payload["rollout"]["task_id"] == "task-a"
    assert payload["rollout"]["metrics"]["estimated_cost_usd"] == pytest.approx(0.12)
    assert payload["rollout"]["criteria"][0]["name"] == "meeting_created"
    assert payload["rollout"]["tool_calls"][0]["tool_name"] == "create_event"
    assert payload["rollout"]["reward_model"]["score"] == pytest.approx(0.45)
    assert (
        payload["rollout"]["reward_model"]["dimension_scores"][0]["dimension"]
        == "Calendar Event Update"
    )


def test_eval_single_atif_rollout_json_includes_metrics_and_tool_calls(
    tmp_path: Path,
) -> None:
    rollout_dir = tmp_path / "agent_run_task_a"
    write_atif_rollout(
        rollout_dir,
        task_id="task-a",
        model="gpt-4o-mini",
        reward=0.0,
        criteria=[
            {"criteria": "meeting_created", "pass": False},
            {"criteria": "scope_discipline", "pass": True},
        ],
        reward_model={
            "score": 0.45,
            "confidence": 0.81,
            "verdict": "partial",
            "failed_criteria": ["meeting_created"],
            "dimension_scores": [
                {"dimension": "Calendar Event Update", "score": 0.2, "reason": "No event created"},
                {"dimension": "Scope Discipline", "score": 0.7, "reason": "No extra actions"},
            ],
        },
        duration_seconds=42.5,
        estimated_cost_usd=0.12,
        prompt_tokens=1200,
        completion_tokens=300,
        tool_calls=[
            {
                "tool_server": "calendar",
                "tool_name": "create_event",
                "parameters": {"title": "Interview"},
            }
        ],
        tool_results=[
            {
                "observation": {"text": "No event created"},
                "is_error": True,
            }
        ],
    )

    runner = CliRunner()
    result = runner.invoke(eval_command, [str(rollout_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "single_rollout"
    assert payload["rollout"]["task_id"] == "task-a"
    assert payload["rollout"]["metrics"]["estimated_cost_usd"] == pytest.approx(0.12)
    assert payload["rollout"]["criteria"][0]["name"] == "meeting_created"
    assert payload["rollout"]["tool_calls"][0]["tool_name"] == "create_event"
    assert payload["rollout"]["tool_calls"][0]["summary"] == "No event created"
    assert payload["rollout"]["tool_calls"][0]["is_error"] is True
    assert payload["rollout"]["reward_model"]["score"] == pytest.approx(0.45)


def test_eval_single_atif_rollout_falls_back_to_final_metrics_and_timestamps(
    tmp_path: Path,
) -> None:
    rollout_dir = tmp_path / "agent_run_task_a"
    write_atif_rollout(
        rollout_dir,
        task_id="task-a",
        model="gpt-4o-mini",
        reward=1.0,
        duration_seconds=42.5,
        estimated_cost_usd=0.12,
        prompt_tokens=1200,
        completion_tokens=300,
        tool_calls=[
            {
                "tool_server": "calendar",
                "tool_name": "create_event",
                "parameters": {"title": "Interview"},
            }
        ],
        tool_results=[
            {
                "observation": {"text": "Created event"},
                "is_error": False,
            }
        ],
        include_rollout_metrics=False,
        agent_step_timestamp="2026-03-24T20:00:05Z",
    )

    runner = CliRunner()
    result = runner.invoke(eval_command, [str(rollout_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rollout"]["metrics"]["estimated_cost_usd"] == pytest.approx(0.12)
    assert payload["rollout"]["metrics"]["prompt_tokens"] == 1200
    assert payload["rollout"]["metrics"]["completion_tokens"] == 300
    assert payload["rollout"]["metrics"]["total_tokens"] == 1500
    assert payload["rollout"]["metrics"]["duration_seconds"] == pytest.approx(5.0)


def test_eval_single_atif_rollout_extracts_harbor_checks_without_verifier_results(
    tmp_path: Path,
) -> None:
    rollout_dir = tmp_path / "agent_run_task_a"
    write_atif_rollout(
        rollout_dir,
        task_id="task-a",
        model="gpt-4o-mini",
        reward=0.0,
        tool_calls=[
            {
                "tool_server": "calendar",
                "tool_name": "create_event",
                "parameters": {"title": "Interview"},
            }
        ],
        tool_results=[
            {
                "observation": {"text": "No event created"},
                "is_error": True,
            }
        ],
    )
    (rollout_dir / "verifier" / "reward.json").write_text(
        json.dumps(
            {
                "reward": 0.0,
                "checks": [
                    {
                        "check": "meeting_created",
                        "passed": False,
                        "description": "Meeting was not created",
                        "weight": 1,
                        "points": 0,
                    }
                ],
                "score": 0.0,
                "harbor_test_sh": {
                    "exit_code": 1,
                    "output": "verification failed",
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(eval_command, [str(rollout_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rollout"]["criteria"][0]["name"] == "meeting_created"
    assert payload["rollout"]["criteria"][0]["passed"] is False
    assert payload["rollout"]["verifier_results"][0]["module"] == "harbor_test_sh"


def test_eval_multi_rollout_summarizes_reward_model_scores_and_error_tables(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    summary_dir = tmp_path / "summary_output"
    write_rollout(
        output_dir / "run_a",
        task_id="schedule_interview",
        model="gpt-4o-mini",
        reward=0.0,
        criteria=[
            {"criteria": "meeting_created", "pass": False},
            {"criteria": "meeting_attendees", "pass": False},
        ],
        reward_model={
            "score": 0.2,
            "failed_criteria": ["meeting_created", "meeting_attendees"],
            "dimension_scores": [
                {"dimension": "Calendar Event Update", "score": 0.1, "reason": "No event"},
                {"dimension": "Chat Communications Quality", "score": 0.3, "reason": "No chat"},
            ],
        },
        duration_seconds=30.0,
        estimated_cost_usd=0.11,
        tool_calls=[
            {"tool_server": "calendar", "tool_name": "search_slots", "parameters": {}},
            {"tool_server": "calendar", "tool_name": "create_event", "parameters": {}},
        ],
        tool_results=[
            {"observation": {"text": "Found 3 slots"}, "is_error": False},
            {
                "observation": {"text": "TimeoutError: locator.click: Timeout 5000ms exceeded"},
                "is_error": True,
            },
        ],
    )
    write_rollout(
        summary_dir / "run_a",
        task_id="schedule_interview",
        model="gpt-4o-mini",
        reward=0.0,
        criteria=[
            {"criteria": "meeting_created", "pass": False},
            {"criteria": "meeting_attendees", "pass": False},
        ],
        reward_model={
            "score": 0.2,
            "failed_criteria": ["meeting_created", "meeting_attendees"],
            "dimension_scores": [
                {"dimension": "Calendar Event Update", "score": 0.1, "reason": "No event"},
                {"dimension": "Chat Communications Quality", "score": 0.3, "reason": "No chat"},
            ],
        },
        duration_seconds=30.0,
        estimated_cost_usd=0.11,
        tool_calls=[
            {"tool_server": "calendar", "tool_name": "search_slots", "parameters": {}},
            {"tool_server": "calendar", "tool_name": "create_event", "parameters": {}},
        ],
        tool_results=[
            {"observation": {"text": "Found 3 slots"}, "is_error": False},
            {
                "observation": {"text": "TimeoutError: locator.click: Timeout 5000ms exceeded"},
                "is_error": True,
            },
        ],
    )
    write_rollout(
        output_dir / "run_b",
        task_id="schedule_interview",
        model="gpt-4o-mini",
        reward=0.0,
        criteria=[
            {"criteria": "meeting_created", "pass": False},
            {"criteria": "meeting_attendees", "pass": False},
        ],
        reward_model={
            "score": 0.8,
            "failed_criteria": ["meeting_attendees"],
            "dimension_scores": [
                {"dimension": "Calendar Event Update", "score": 0.9, "reason": "Event exists"},
                {"dimension": "Chat Communications Quality", "score": 0.7, "reason": "Reasonable"},
            ],
        },
        duration_seconds=34.0,
        estimated_cost_usd=0.13,
        tool_calls=[
            {"tool_server": "calendar", "tool_name": "search_slots", "parameters": {}},
            {"tool_server": "calendar", "tool_name": "create_event", "parameters": {}},
        ],
        tool_results=[
            {"observation": {"text": "Found 2 slots"}, "is_error": False},
            {
                "observation": {"text": "Missing payload for create_event"},
                "is_error": True,
            },
        ],
    )
    write_rollout(
        summary_dir / "run_b",
        task_id="schedule_interview",
        model="gpt-4o-mini",
        reward=0.0,
        criteria=[
            {"criteria": "meeting_created", "pass": False},
            {"criteria": "meeting_attendees", "pass": False},
        ],
        reward_model={
            "score": 0.8,
            "failed_criteria": ["meeting_attendees"],
            "dimension_scores": [
                {"dimension": "Calendar Event Update", "score": 0.9, "reason": "Event exists"},
                {"dimension": "Chat Communications Quality", "score": 0.7, "reason": "Reasonable"},
            ],
        },
        duration_seconds=34.0,
        estimated_cost_usd=0.13,
        tool_calls=[
            {"tool_server": "calendar", "tool_name": "search_slots", "parameters": {}},
            {"tool_server": "calendar", "tool_name": "create_event", "parameters": {}},
        ],
        tool_results=[
            {"observation": {"text": "Found 2 slots"}, "is_error": False},
            {
                "observation": {"text": "Missing payload for create_event"},
                "is_error": True,
            },
        ],
    )
    write_rollout(
        output_dir / "run_c",
        task_id="send_rejection",
        model="gpt-4o",
        reward=1.0,
        criteria=[
            {"criteria": "email_sent", "pass": True},
            {"criteria": "scope_discipline", "pass": True},
        ],
        reward_model={
            "score": 0.95,
            "failed_criteria": [],
            "dimension_scores": [
                {"dimension": "Email Notification Quality", "score": 0.95, "reason": "Good"},
            ],
        },
        duration_seconds=20.0,
        estimated_cost_usd=0.05,
    )
    broken_dir = output_dir / "run_broken"
    write_rollout(broken_dir, task_id="send_rejection", model="gpt-4o", reward=1.0)
    (broken_dir / "verifier" / "reward.json").write_text("{bad json", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(eval_command, [str(summary_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "multi_rollout"
    assert payload["summary"]["rollout_count"] == 2
    assert payload["summary"]["results_by_task"][0]["task_id"] == "schedule_interview"
    assert payload["summary"]["score_summary"]["mean_reward_model_score"] == pytest.approx(0.5)
    assert payload["summary"]["score_summary"]["reward_model_score_variance"] == pytest.approx(0.09)
    assert payload["summary"]["overview"]["total_steps"] == 4
    assert payload["summary"]["overview"]["rollouts_with_steps"] == 2
    assert "avg_steps_per_active_rollout" not in payload["summary"]["overview"]
    assert payload["summary"]["never_passed_checks"][0]["name"] == "meeting_attendees"
    assert (
        payload["summary"]["call_sequence_patterns"]["first_calls"][0]["tool"]
        == "calendar__search_slots"
    )

    error_categories = {
        row["category"] for row in payload["summary"]["tool_error_taxonomy"]["categories"]
    }
    assert "timeout" in error_categories
    assert "validation_error" in error_categories

    capability_row = payload["summary"]["capability_analysis"][0]
    assert capability_row["capability"] == "Calendar Operations"
    assert capability_row["cofailure_patterns"][0]["criteria_names"] == [
        "meeting_attendees",
        "meeting_created",
    ]

    unfiltered_result = runner.invoke(eval_command, [str(output_dir), "--json"])
    unfiltered_payload = json.loads(unfiltered_result.output)
    assert unfiltered_payload["warnings"]
    assert "reward.json" in unfiltered_payload["warnings"][0]


def test_build_tool_calls_keeps_recorded_non_errors() -> None:
    tool_calls = build_tool_calls(
        [
            {
                "tool_server": "calendar",
                "tool_name": "find_event",
                "parameters": {},
            }
        ],
        [
            {
                "observation": {"text": "Event not found"},
                "is_error": False,
            }
        ],
        None,
    )

    assert tool_calls[0]["is_error"] is False


def test_score_summary_aggregates_check_pass_rate_across_checks() -> None:
    summary = summarize_score_summary(
        [
            {
                "criteria": [{"passed": True}],
                "reward_model": None,
                "reward": None,
            },
            {
                "criteria": [{"passed": False}] * 9,
                "reward_model": None,
                "reward": None,
            },
        ]
    )

    assert summary["individual_module_check_pass_rate"] == pytest.approx(0.1)
    assert summary["mean_composite_score"] == pytest.approx(0.5)


def test_detect_cofailure_patterns_uses_matched_rollout_count() -> None:
    patterns = detect_cofailure_patterns(
        {
            "criterion_a": {"rollout_1", "rollout_2"},
            "criterion_b": {"rollout_1", "rollout_2"},
        },
        {
            "criterion_a": "Calendar Operations",
            "criterion_b": "Calendar Operations",
        },
    )

    assert patterns[0]["fail_count"] == 2
    assert patterns[0]["rollout_count"] == 2


def test_eval_compare_json_includes_overview_and_task_deltas(tmp_path: Path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"

    write_rollout(
        left_dir / "run_a",
        task_id="send_rejection",
        model="gpt-5",
        reward=0.0,
        criteria=[{"criteria": "email_sent", "pass": False}],
        reward_model={
            "score": 0.2,
            "dimension_scores": [{"dimension": "Email Notification Quality", "score": 0.2}],
        },
        estimated_cost_usd=0.20,
        duration_seconds=40.0,
    )
    write_rollout(
        right_dir / "run_a",
        task_id="send_rejection",
        model="gpt-5",
        reward=1.0,
        criteria=[{"criteria": "email_sent", "pass": True}],
        reward_model={
            "score": 0.9,
            "dimension_scores": [{"dimension": "Email Notification Quality", "score": 0.9}],
        },
        estimated_cost_usd=0.25,
        duration_seconds=28.0,
    )

    runner = CliRunner()
    result = runner.invoke(eval_command, [str(left_dir), "--compare", str(right_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "compare"
    assert payload["compare_kind"] == "rollout"
    overview = {row["metric"]: row for row in payload["comparison"]["overview"]}
    assert overview["mean_reward_model_score"]["left_value"] == pytest.approx(0.2)
    assert overview["mean_reward_model_score"]["right_value"] == pytest.approx(0.9)
    assert overview["mean_reward_model_score"]["delta"] == pytest.approx(-0.7)
    assert payload["rollout_comparison"]["criteria"][0]["criterion"] == "email_sent"

    task_row = payload["comparison"]["results_by_task"][0]
    assert task_row["task_id"] == "send_rejection"
    assert task_row["delta_pass_rate"] == pytest.approx(-1.0)
    assert task_row["delta_mean_reward_model_score"] == pytest.approx(-0.7)


def test_eval_single_rollout_text_uses_verifier_terms_and_full_tool_names(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "agent_run_task_a"
    write_rollout(
        rollout_dir,
        task_id="task-a",
        model="gpt-5",
        reward=0.0,
        criteria=[{"criteria": "calendar_done", "pass": False}],
        reward_model={
            "score": 0.3,
            "dimension_scores": [{"dimension": "Calendar Event Update", "score": 0.3}],
        },
        tool_calls=[
            {
                "tool_server": "calendar",
                "tool_name": "create_event",
                "parameters": {"title": "Interview"},
            }
        ],
        tool_results=[{"observation": {"text": "No event created"}, "is_error": True}],
    )

    runner = CliRunner()
    result = runner.invoke(eval_command, [str(rollout_dir)])

    assert result.exit_code == 0, result.output
    assert "Programmatic Verifier Results" in result.output
    assert "Reward Model" in result.output
    assert "calendar__create_event" in result.output
    assert "Final Observation" not in result.output
    assert "First tool pattern" not in result.output


def test_eval_compare_text_renders_tables(tmp_path: Path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"

    write_rollout(
        left_dir / "run_a",
        task_id="task-a",
        model="gpt-5",
        reward=0.0,
        criteria=[{"criteria": "calendar_done", "pass": False}],
        reward_model={
            "score": 0.3,
            "dimension_scores": [{"dimension": "Calendar Event Update", "score": 0.3}],
        },
        tool_calls=[
            {"tool_server": "calendar", "tool_name": "create_event", "parameters": {}},
        ],
        tool_results=[{"observation": {"text": "TimeoutError: click"}, "is_error": True}],
    )
    write_rollout(
        right_dir / "run_a",
        task_id="task-a",
        model="gpt-5",
        reward=1.0,
        criteria=[{"criteria": "calendar_done", "pass": True}],
        reward_model={
            "score": 0.8,
            "dimension_scores": [{"dimension": "Calendar Event Update", "score": 0.8}],
        },
        tool_calls=[
            {"tool_server": "calendar", "tool_name": "create_event", "parameters": {}},
        ],
        tool_results=[{"observation": {"text": "Created event"}, "is_error": False}],
    )

    runner = CliRunner()
    result = runner.invoke(eval_command, [str(left_dir), "--compare", str(right_dir)])

    assert result.exit_code == 0, result.output
    assert "Rollout Comparison" in result.output
    assert "Programmatic Verifier Comparison" in result.output
    assert "Reward Model Dimension Comparison" in result.output
    assert "Tool Sequence" in result.output
    assert "Shared starting tool prefix" not in result.output
    assert "calendar__create_event" in result.output
    assert f"A: {left_dir.name}" not in result.output
    assert f"B: {right_dir.name}" not in result.output


def test_eval_multi_rollout_text_uses_verifier_terms(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    write_rollout(
        output_dir / "run_a",
        task_id="schedule_interview",
        model="gpt-5",
        reward=0.0,
        criteria=[
            {"criteria": "meeting_created", "pass": False},
            {"criteria": "meeting_attendees", "pass": False},
        ],
        reward_model={
            "score": 0.2,
            "failed_criteria": ["meeting_created"],
            "dimension_scores": [
                {"dimension": "Calendar Event Update", "score": 0.2},
            ],
        },
        tool_calls=[
            {"tool_server": "calendar", "tool_name": "create_event", "parameters": {}},
        ],
        tool_results=[{"observation": {"text": "Missing payload"}, "is_error": True}],
    )

    runner = CliRunner()
    result = runner.invoke(eval_command, [str(output_dir)])

    assert result.exit_code == 0, result.output
    assert "Programmatic verifier all-checks pass rate" in result.output
    assert "Mean reward model score" in result.output
    assert "Programmatic Verifier Checks That Never Pass" in result.output
    assert "Reward Model Failed Criteria" in result.output
    assert "Average steps among active rollouts" not in result.output


def test_eval_report_flag_writes_default_markdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    write_rollout(output_dir / "run_a", task_id="task-a", model="gpt-4o-mini", reward=1.0)

    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SIMLAB_CONFIG", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SIMLAB_AGENT_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(eval_command, ["--json", "--report"])

    assert result.exit_code == 0, result.output

    report_file = tmp_path / "eval-report.md"
    assert report_file.is_file()
    report_text = report_file.read_text(encoding="utf-8")
    assert report_text.startswith("# Evaluation Report\n")
    assert "## Results by Task" in report_text
    assert "## Capability Analysis" in report_text
    assert "*Generated by SimLab on" in report_text


def test_eval_report_option_accepts_explicit_path_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    write_rollout(output_dir / "run_a", task_id="task-a", model="gpt-4o-mini", reward=1.0)

    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SIMLAB_CONFIG", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SIMLAB_AGENT_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(eval_command, ["--json", "--report", "custom-report.md"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "custom-report.md").is_file()


def test_eval_report_warns_before_overwriting_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    write_rollout(output_dir / "run_a", task_id="task-a", model="gpt-4o-mini", reward=1.0)

    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SIMLAB_CONFIG", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SIMLAB_AGENT_API_KEY", raising=False)

    report_file = tmp_path / "custom-report.md"
    report_file.write_text("old", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        eval_command,
        [str(output_dir), "--report", "custom-report.md"],
    )

    assert result.exit_code == 0, result.output
    assert result.output.startswith("WARNING: report file already exists and will be overwritten:")

    report_text = report_file.read_text(encoding="utf-8")
    assert report_text.startswith("# Evaluation Report\n")


def test_eval_report_respects_task_and_model_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    write_rollout(output_dir / "run_a", task_id="task-a", model="gpt-4o-mini", reward=1.0)
    write_rollout(output_dir / "run_b", task_id="task-b", model="gpt-4o-mini", reward=0.0)
    write_rollout(output_dir / "run_c", task_id="task-a", model="gpt-5.2", reward=1.0)

    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SIMLAB_CONFIG", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SIMLAB_AGENT_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(
        eval_command,
        [
            str(output_dir),
            "--json",
            "--task",
            "task-a",
            "--model",
            "gpt-4o-mini",
            "--report",
            "filtered.md",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["summary"]["rollout_count"] == 1
    assert payload["summary"]["task_count"] == 1
    assert payload["summary"]["models"] == ["gpt-4o-mini"]

    report_text = (tmp_path / "filtered.md").read_text(encoding="utf-8")
    assert "task-b" not in report_text
    assert "gpt-5.2" not in report_text


def test_eval_report_works_with_compare_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    write_rollout(left_dir / "run_a", task_id="task-a", model="gpt-4o-mini", reward=1.0)
    write_rollout(right_dir / "run_b", task_id="task-a", model="gpt-4o-mini", reward=0.0)

    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SIMLAB_CONFIG", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SIMLAB_AGENT_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(
        eval_command,
        [
            str(left_dir),
            "--compare",
            str(right_dir),
            "--json",
            "--report",
            "compare.md",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "compare"
    report_text = (tmp_path / "compare.md").read_text(encoding="utf-8")
    assert report_text.startswith("# Comparison Report\n")
    assert "## Programmatic Verifier Comparison" in report_text
    assert "## Tool Sequence" in report_text
