from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from simlab.api.client import ScenarioManagerClient
from simlab.cli.tools import tools
from simlab.telemetry import BackgroundRequestPoster
from simlab.telemetry import QueuedTelemetryRequest
from simlab.telemetry import TelemetryService
from simlab.telemetry import ensure_session
from simlab.telemetry import request_headers_var
from simlab.telemetry import utc_now


def test_command_telemetry_posts_started_and_finished_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")

    captured_requests: list[dict[str, Any]] = []

    def fake_queue_telemetry_request(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> None:
        captured_requests.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout_seconds": timeout_seconds,
            }
        )

    runner = CliRunner()
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=tmp_path / "telemetry.json"),
        patch("simlab.telemetry.queue_telemetry_request", side_effect=fake_queue_telemetry_request),
        patch("simlab.cli.tools.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = []
        result = runner.invoke(tools, ["list"])

    assert result.exit_code == 0, result.output
    assert len(captured_requests) == 3
    start_call = captured_requests[0]
    specific_call = captured_requests[1]
    finish_call = captured_requests[2]
    assert start_call["payload"]["event"] == "cli_command_started"
    assert specific_call["payload"]["event"] == "tools_list_completed"
    assert finish_call["payload"]["event"] == "cli_command_finished"
    assert start_call["payload"]["properties"]["command"] == "tools list"
    assert finish_call["payload"]["properties"]["command"] == "tools list"
    install_id = start_call["payload"]["properties"]["install_id"]
    assert install_id
    assert specific_call["payload"]["properties"]["install_id"] == install_id
    assert finish_call["payload"]["properties"]["install_id"] == install_id


def test_command_telemetry_posts_for_local_scenario_manager_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "http://127.0.0.1:9011")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")

    captured_requests: list[dict[str, Any]] = []

    def fake_queue_telemetry_request(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> None:
        captured_requests.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout_seconds": timeout_seconds,
            }
        )

    runner = CliRunner()
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=tmp_path / "telemetry.json"),
        patch("simlab.telemetry.queue_telemetry_request", side_effect=fake_queue_telemetry_request),
        patch("simlab.cli.tools.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = []
        result = runner.invoke(tools, ["list"])

    assert result.exit_code == 0, result.output
    assert len(captured_requests) == 3
    assert captured_requests[0]["url"] == "http://127.0.0.1:9011/telemetry/cli-events"
    assert captured_requests[1]["payload"]["event"] == "tools_list_completed"


def test_generic_wrapped_command_posts_for_local_catalog_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")

    fake_tool = SimpleNamespace(
        display_name="Email",
        description="Email tool",
        category="communication",
        is_external=False,
        tool_server_url=None,
        tool_server_port=8040,
        services={},
        exposed_ports=[],
        seed_services={},
        required_env_vars=[],
    )
    captured_requests: list[dict[str, Any]] = []

    def fake_queue_telemetry_request(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> None:
        captured_requests.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout_seconds": timeout_seconds,
            }
        )

    runner = CliRunner()
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=tmp_path / "telemetry.json"),
        patch("simlab.telemetry.queue_telemetry_request", side_effect=fake_queue_telemetry_request),
        patch("simlab.cli.tools.build_registry") as mocked_registry,
    ):
        mocked_registry.return_value.get_tool.return_value = fake_tool
        result = runner.invoke(tools, ["info", "email"])

    assert result.exit_code == 0, result.output
    assert len(captured_requests) == 2
    assert captured_requests[0]["payload"]["event"] == "cli_command_started"
    assert captured_requests[1]["payload"]["event"] == "cli_command_finished"
    assert captured_requests[0]["payload"]["properties"]["command"] == "tools info"


def test_command_telemetry_skips_without_collinear_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.delenv("SIMLAB_COLLINEAR_API_KEY", raising=False)
    monkeypatch.setenv("SIMLAB_CONFIG", str(tmp_path / "nonexistent.toml"))

    runner = CliRunner()
    state_path = tmp_path / "telemetry.json"
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=state_path),
        patch("simlab.telemetry.queue_telemetry_request") as mocked_queue,
        patch("simlab.cli.tools.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = []
        result = runner.invoke(tools, ["list"])

    assert result.exit_code == 0, result.output
    mocked_queue.assert_not_called()
    assert not state_path.exists()


def test_command_telemetry_skips_when_disabled_by_simlab_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")
    monkeypatch.setenv("SIMLAB_DISABLE_TELEMETRY", "1")

    runner = CliRunner()
    state_path = tmp_path / "telemetry.json"
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=state_path),
        patch("simlab.telemetry.queue_telemetry_request") as mocked_queue,
        patch("simlab.cli.tools.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = []
        result = runner.invoke(tools, ["list"])

    assert result.exit_code == 0, result.output
    mocked_queue.assert_not_called()
    assert not state_path.exists()


def test_command_telemetry_shows_first_run_notice_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")

    runner = CliRunner()
    state_path = tmp_path / "telemetry.json"
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=state_path),
        patch("simlab.telemetry.stderr_supports_notice", return_value=True),
        patch("simlab.telemetry.queue_telemetry_request"),
        patch("simlab.cli.tools.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = []
        first_result = runner.invoke(tools, ["list"])
        second_result = runner.invoke(tools, ["list"])

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    assert "SimLab sends usage telemetry" in first_result.output
    assert "Set SIMLAB_DISABLE_TELEMETRY=1 to disable." in first_result.output
    assert "SimLab sends usage telemetry" not in second_result.output

    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["notice_shown"] is True


def test_command_telemetry_does_not_crash_when_state_dir_is_unwritable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIMLAB_SCENARIO_MANAGER_API_URL", "https://rl-gym-api.collinear.ai")
    monkeypatch.setenv("SIMLAB_COLLINEAR_API_KEY", "ck_test_internal")

    state_path = tmp_path / "readonly" / "telemetry.json"
    real_mkdir = Path.mkdir

    def fail_for_state_dir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if self == state_path.parent:
            raise PermissionError("read-only state directory")
        real_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    runner = CliRunner()
    with (
        patch("simlab.telemetry.telemetry_state_path", return_value=state_path),
        patch("simlab.telemetry.Path.mkdir", new=fail_for_state_dir),
        patch("simlab.telemetry.queue_telemetry_request") as mocked_queue,
        patch("simlab.cli.tools.ScenarioManagerClient") as mocked_client_cls,
    ):
        mocked_client_cls.return_value.list_scenarios.return_value = []
        result = runner.invoke(tools, ["list"])

    assert result.exit_code == 0, result.output
    assert mocked_queue.call_count == 3
    assert not state_path.exists()


def test_api_client_includes_active_cli_headers() -> None:
    client = ScenarioManagerClient(base_url="https://api.example.com")

    class FakeResponse:
        status_code = 200
        reason = "OK"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return []

    token = request_headers_var.set(
        {
            "X-SimLab-Command": "tools list",
            "X-SimLab-Install-Id": "install-123",
            "X-SimLab-Session-Id": "session-456",
            "X-SimLab-Version": "0.1.0",
        }
    )
    try:
        with patch("simlab.api.client.requests.request", return_value=FakeResponse()) as mocked:
            client.list_scenarios()
    finally:
        request_headers_var.reset(token)

    headers = mocked.call_args.kwargs["headers"]
    assert headers["X-SimLab-Command"] == "tools list"
    assert headers["X-SimLab-Install-Id"] == "install-123"
    assert headers["X-SimLab-Session-Id"] == "session-456"
    assert headers["X-SimLab-Version"] == "0.1.0"


def test_ensure_session_preserves_install_id_when_session_rotates() -> None:
    now = utc_now()
    expired = now - timedelta(minutes=31)
    state = {
        "install_id": "install-123",
        "session_id": "session-old",
        "session_started_at": expired.isoformat(),
        "last_seen_at": expired.isoformat(),
    }

    updated = ensure_session(state, now=now)

    assert updated["install_id"] == "install-123"
    assert updated["session_id"] != "session-old"


def test_telemetry_service_ignores_local_simlab_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    local_state_path = tmp_path / "simlab" / "simlab.json"
    local_state_path.parent.mkdir(parents=True, exist_ok=True)
    local_state_path.write_text(
        json.dumps(
            {
                "install_id": "legacy-install",
                "session_id": "legacy-session",
                "session_started_at": "2026-03-20T00:00:00+00:00",
                "last_seen_at": "2026-03-20T00:05:00+00:00",
                "notice_shown": True,
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / ".config" / "simlab" / "simlab.json"

    with patch("simlab.telemetry.stderr_supports_notice", return_value=False):
        telemetry = TelemetryService(
            base_url="https://api.example.com",
            api_key="ck_test_internal",
            state_path=state_path,
        )

    assert local_state_path.exists()
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["install_id"] == telemetry.install_id
    assert saved_state["install_id"] != "legacy-install"


def test_background_request_poster_waits_for_pending_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_urls: list[str] = []

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> None:
        _ = json, headers, timeout
        captured_urls.append(url)

    monkeypatch.setattr("simlab.telemetry.requests.post", fake_post)
    poster = BackgroundRequestPoster(thread_name="test-telemetry", worker_count=3)
    request = QueuedTelemetryRequest(
        url="https://api.example.com/telemetry/cli-events",
        payload={"event": "cli_command_finished"},
        headers={"Content-Type": "application/json"},
        timeout_seconds=1.0,
    )
    poster.submit(request)
    poster.submit(request)
    poster.submit(request)

    assert poster.wait_until_idle(1.0) is True
    expected_url = "https://api.example.com/telemetry/cli-events"
    matched = [u for u in captured_urls if u == expected_url]
    assert matched == [expected_url] * 3
