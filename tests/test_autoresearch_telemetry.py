from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from simlab.cli.autoresearch import autoresearch


class _FakePrompt:
    def __init__(self, value: object) -> None:
        self.value = value

    def ask(self) -> object:
        return self.value


def _make_fake_queue(captured: list[dict[str, Any]]):
    def fake_queue_telemetry_request(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> None:
        captured.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout_seconds": timeout_seconds,
            }
        )

    return fake_queue_telemetry_request


def test_autoresearch_run_wizard_emits_step_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")

    captured: list[dict[str, Any]] = []
    runner = CliRunner()

    confirm_answers = iter([False, True])

    def fake_confirm(*args: object, **kwargs: object) -> _FakePrompt:
        _ = args, kwargs
        return _FakePrompt(next(confirm_answers))

    def fake_text(*args: object, **kwargs: object) -> _FakePrompt:
        _ = args, kwargs
        return _FakePrompt("3")

    def fake_select(*args: object, **kwargs: object) -> _FakePrompt:
        _ = args, kwargs
        return _FakePrompt("local")

    def fake_configure_models(
        *,
        agent_section: dict[str, object],
        proposer_section: dict[str, object],
        verifier_section: dict[str, object],
    ) -> None:
        agent_section["model"] = "gpt-4o-mini"
        proposer_section["model"] = "gpt-4o-mini"
        verifier_section["model"] = "gpt-4o-mini"

    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=tmp_path / "telemetry.json"),
        patch("simlab.telemetry.queue_telemetry_request", side_effect=_make_fake_queue(captured)),
        patch("simlab.cli.autoresearch.sys_tty", return_value=True),
        patch("simlab.cli.autoresearch._resolve_env_name", return_value="setup1"),
        patch("simlab.cli.autoresearch._resolve_tasks_dir", return_value="tasks_dir"),
        patch(
            "simlab.cli.autoresearch._discover_tasks",
            return_value=[("t1", "Task 1", True), ("t2", "Task 2", True), ("t3", "Task 3", True)],
        ),
        patch("simlab.cli.autoresearch._resolve_task_ids", return_value=["t1", "t2", "t3"]),
        patch("simlab.cli.autoresearch._daytona_runtime_available", return_value=(True, "")),
        patch("simlab.cli.autoresearch.select", side_effect=fake_select),
        patch("simlab.cli.autoresearch.text", side_effect=fake_text),
        patch("simlab.cli.autoresearch.confirm", side_effect=fake_confirm),
        patch(
            "simlab.cli.autoresearch._configure_model_sections",
            side_effect=fake_configure_models,
        ),
        patch("simlab.cli.autoresearch.ensure_autoresearch_environment_resolves"),
        patch("simlab.cli.autoresearch.run_autoresearch"),
    ):
        result = runner.invoke(autoresearch, ["run"])

    assert result.exit_code == 0, result.output

    events = [request["payload"]["event"] for request in captured]
    assert "autoresearch_wizard_started" in events
    assert events.count("autoresearch_wizard_step_shown") == 8
    assert events.count("autoresearch_wizard_step_completed") == 8
    assert "autoresearch_wizard_completed" in events

    task_event = next(
        request
        for request in captured
        if request["payload"]["event"] == "autoresearch_wizard_step_completed"
        and request["payload"]["properties"].get("step") == "tasks"
    )
    assert task_event["payload"]["properties"]["task_count"] == 3

    runtime_event = next(
        request
        for request in captured
        if request["payload"]["event"] == "autoresearch_wizard_step_completed"
        and request["payload"]["properties"].get("step") == "runtime"
    )
    assert runtime_event["payload"]["properties"]["runtime"] == "local"
    assert runtime_event["payload"]["properties"]["prompted"] is True
