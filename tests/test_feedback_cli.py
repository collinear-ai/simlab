from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from simlab.cli.feedback import feedback


def _make_fake_queue(captured: list[dict[str, Any]]):
    """Return a replacement for queue_telemetry_request that records calls."""

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


def test_feedback_inline_message_sends_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inline mode: simlab feedback 'some message' sends feedback_submitted event."""
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")
    monkeypatch.delenv("SIMLAB_DISABLE_TELEMETRY", raising=False)

    captured: list[dict[str, Any]] = []
    runner = CliRunner()
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=tmp_path / "telemetry.json"),
        patch("simlab.telemetry.queue_telemetry_request", side_effect=_make_fake_queue(captured)),
    ):
        result = runner.invoke(feedback, ["The task gen is great!"])

    assert result.exit_code == 0, result.output
    assert "Thanks! Your feedback has been sent." in result.output

    events = [r["payload"]["event"] for r in captured]
    assert "cli_command_started" in events
    assert "feedback_submitted" in events
    assert "cli_command_finished" in events

    feedback_event = next(r for r in captured if r["payload"]["event"] == "feedback_submitted")
    props = feedback_event["payload"]["properties"]
    assert props["message"] == "The task gen is great!"
    assert props["command"] == "feedback"


def test_feedback_interactive_mode_prompts_and_sends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Interactive mode: simlab feedback prompts for input, then sends event."""
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")
    monkeypatch.delenv("SIMLAB_DISABLE_TELEMETRY", raising=False)

    captured: list[dict[str, Any]] = []
    runner = CliRunner()
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=tmp_path / "telemetry.json"),
        patch("simlab.telemetry.queue_telemetry_request", side_effect=_make_fake_queue(captured)),
    ):
        result = runner.invoke(feedback, input="Interactive feedback\n")

    assert result.exit_code == 0, result.output
    assert "Thanks! Your feedback has been sent." in result.output

    feedback_event = next(r for r in captured if r["payload"]["event"] == "feedback_submitted")
    assert feedback_event["payload"]["properties"]["message"] == "Interactive feedback"


def test_feedback_empty_message_shows_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Empty message shows warning and does not send event."""
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")
    monkeypatch.delenv("SIMLAB_DISABLE_TELEMETRY", raising=False)

    captured: list[dict[str, Any]] = []
    runner = CliRunner()
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=tmp_path / "telemetry.json"),
        patch("simlab.telemetry.queue_telemetry_request", side_effect=_make_fake_queue(captured)),
    ):
        result = runner.invoke(feedback, ["   "])

    assert result.exit_code == 0, result.output
    assert "No feedback provided." in result.output

    feedback_events = [r for r in captured if r["payload"]["event"] == "feedback_submitted"]
    assert len(feedback_events) == 0


def test_feedback_with_env_includes_env_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--env flag includes env_name in event properties."""
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")
    monkeypatch.delenv("SIMLAB_DISABLE_TELEMETRY", raising=False)

    captured: list[dict[str, Any]] = []
    runner = CliRunner()
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=tmp_path / "telemetry.json"),
        patch("simlab.telemetry.queue_telemetry_request", side_effect=_make_fake_queue(captured)),
    ):
        result = runner.invoke(feedback, ["--env", "my-env", "feedback with env context"])

    assert result.exit_code == 0, result.output
    assert "Thanks! Your feedback has been sent." in result.output

    feedback_event = next(r for r in captured if r["payload"]["event"] == "feedback_submitted")
    props = feedback_event["payload"]["properties"]
    assert props["message"] == "feedback with env context"
    assert props["env_name"] == "my-env"


def test_feedback_telemetry_disabled_shows_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When telemetry is disabled, feedback is not sent and user is informed."""
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")
    monkeypatch.setenv("SIMLAB_DISABLE_TELEMETRY", "1")

    captured: list[dict[str, Any]] = []
    runner = CliRunner()
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=tmp_path / "telemetry.json"),
        patch("simlab.telemetry.queue_telemetry_request", side_effect=_make_fake_queue(captured)),
    ):
        result = runner.invoke(feedback, ["some message"])

    assert result.exit_code == 0, result.output
    assert "Feedback could not be sent because telemetry is disabled" in result.output
    assert "simlab@collinear.ai" in result.output
    assert "Thanks!" not in result.output

    feedback_events = [r for r in captured if r["payload"]["event"] == "feedback_submitted"]
    assert len(feedback_events) == 0
