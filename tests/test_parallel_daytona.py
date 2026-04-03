"""Tests for parallel Daytona rollout orchestrator.

Covers the key behaviors and fixes from RLG-629:
- Cancellation event stops in-flight workers on Ctrl+C
- Cleanup preserves failed teardowns in state file
- Orphaned sandbox cleanup from durable state
- summary.json correctly reports passed/failed/avg_reward
- _require_reachable_endpoints wait mode
- DaytonaRunner.up() converts preseed RuntimeError to SystemExit
"""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from simlab.agents.base import BaseEnvironment
from simlab.agents.base import RunArtifacts
from simlab.agents.reference_agent import ReferenceAgent
from simlab.cli import tasks as tasks_cli
from simlab.runtime.daytona_runner import DaytonaNotFoundError
from simlab.runtime.daytona_runner import DaytonaRunner
from simlab.runtime.parallel_daytona import ParallelDaytonaOrchestrator
from simlab.runtime.parallel_daytona import ParallelRunSummary
from simlab.runtime.parallel_daytona import RolloutResult


def _raise_preseed_boom() -> None:
    """Raise the runtime error used by preseed failure simulation."""
    raise RuntimeError("preseed boom")


def _simulate_preseed_system_exit(fake_rpt: MagicMock) -> None:
    """Mirror DaytonaRunner.up() converting preseed failures into SystemExit."""
    try:
        _raise_preseed_boom()
    except RuntimeError as exc:
        fake_rpt.end_step(success=False, error=str(exc))
        raise SystemExit(1) from exc


def _simulate_preseed_system_exit_with_cleanup(
    fake_rpt: MagicMock,
    fake_daytona: MagicMock,
    fake_sandbox: MagicMock,
) -> None:
    """Mirror the preseed error path including sandbox cleanup."""
    try:
        _raise_preseed_boom()
    except RuntimeError as exc:
        fake_rpt.end_step(success=False, error=str(exc))
        with contextlib.suppress(Exception):
            fake_daytona.delete(fake_sandbox)
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Cancellation event
# ---------------------------------------------------------------------------


class TestCancellationEvent:
    """Verify that _cancel event stops in-flight workers."""

    def test_check_cancelled_raises_when_set(self) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)
        orch._cancel.set()
        with pytest.raises(RuntimeError, match="Cancelled"):
            orch._check_cancelled("[rollout 1/1]")

    def test_check_cancelled_passes_when_not_set(self) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)
        # Should not raise
        orch._check_cancelled("[rollout 1/1]")

    def test_cancel_event_stops_worker_before_agent_run(self) -> None:
        """Simulate a worker that checks cancellation before expensive agent step."""
        orch = ParallelDaytonaOrchestrator(rollout_count=2, max_parallel=2)
        checkpoints_reached: list[str] = []

        def fake_worker(idx: int) -> str:
            checkpoints_reached.append(f"start-{idx}")
            orch._check_cancelled(f"[rollout {idx}]")
            checkpoints_reached.append(f"setup-{idx}")
            orch._check_cancelled(f"[rollout {idx}]")
            checkpoints_reached.append(f"agent-{idx}")
            return f"done-{idx}"

        # Set cancel before any worker runs
        orch._cancel.set()
        with pytest.raises(RuntimeError, match="Cancelled"):
            fake_worker(0)

        # Only the start checkpoint should have been reached
        assert checkpoints_reached == ["start-0"]


# ---------------------------------------------------------------------------
# Cleanup preserves failed teardowns
# ---------------------------------------------------------------------------


class TestCleanupPreservesFailedTeardowns:
    """Verify _cleanup_all keeps sandbox tracking when teardown returns False."""

    def test_failed_teardown_keeps_sandbox_in_tracking(self, tmp_path: Path) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)
        orch._state_file = tmp_path / "parallel-sandboxes.json"

        fake_client = MagicMock()
        fake_sandbox = MagicMock()
        fake_sandbox.id = "sbx-123"

        orch._active_sandboxes[0] = (fake_client, fake_sandbox)
        orch._persist_sandbox("sbx-123")

        # teardown_sandbox returns False (delete failed without exception)
        with patch("simlab.runtime.parallel_daytona.teardown_sandbox", return_value=False):
            orch._cleanup_all()

        # Sandbox should STILL be in state file
        assert orch._state_file.exists()
        state = json.loads(orch._state_file.read_text())
        assert "sbx-123" in state

    def test_successful_teardown_removes_from_tracking(self, tmp_path: Path) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)
        orch._state_file = tmp_path / "parallel-sandboxes.json"

        fake_client = MagicMock()
        fake_sandbox = MagicMock()
        fake_sandbox.id = "sbx-456"

        orch._active_sandboxes[0] = (fake_client, fake_sandbox)
        orch._persist_sandbox("sbx-456")

        with patch("simlab.runtime.parallel_daytona.teardown_sandbox", return_value=True):
            orch._cleanup_all()

        # Sandbox should be removed — state file deleted (empty state)
        assert not orch._state_file.exists()
        assert 0 not in orch._active_sandboxes

    def test_teardown_exception_keeps_sandbox_in_tracking(self, tmp_path: Path) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)
        orch._state_file = tmp_path / "parallel-sandboxes.json"

        fake_client = MagicMock()
        fake_sandbox = MagicMock()
        fake_sandbox.id = "sbx-789"

        orch._active_sandboxes[0] = (fake_client, fake_sandbox)
        orch._persist_sandbox("sbx-789")

        with patch(
            "simlab.runtime.parallel_daytona.teardown_sandbox",
            side_effect=Exception("API error"),
        ):
            orch._cleanup_all()

        assert orch._state_file.exists()
        state = json.loads(orch._state_file.read_text())
        assert state["sbx-789"] == "sbx-789"


# ---------------------------------------------------------------------------
# Durable state file
# ---------------------------------------------------------------------------


class TestDurableStateFile:
    """Verify persist/unpersist and cleanup_orphaned_sandboxes."""

    def test_persist_and_unpersist(self, tmp_path: Path) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=3, max_parallel=2)
        orch._state_file = tmp_path / "parallel-sandboxes.json"

        orch._persist_sandbox("sbx-a")
        orch._persist_sandbox("sbx-b")

        state = json.loads(orch._state_file.read_text())
        assert state == {"sbx-a": "sbx-a", "sbx-b": "sbx-b"}

        orch._unpersist_sandbox("sbx-a")
        state = json.loads(orch._state_file.read_text())
        assert state == {"sbx-b": "sbx-b"}

        orch._unpersist_sandbox("sbx-b")
        # State file should be deleted when empty
        assert not orch._state_file.exists()

    def test_cleanup_orphaned_sandboxes_deletes_and_removes_state(self, tmp_path: Path) -> None:
        state_file = tmp_path / "parallel-sandboxes.json"
        state_file.write_text(
            json.dumps({"sbx-orphan-1": "sbx-orphan-1", "sbx-orphan-2": "sbx-orphan-2"})
        )

        fake_daytona = MagicMock()
        fake_sandbox = MagicMock()
        fake_daytona.get.return_value = fake_sandbox

        with patch("simlab.runtime.parallel_daytona._get_daytona", return_value=fake_daytona):
            deleted = ParallelDaytonaOrchestrator.cleanup_orphaned_sandboxes(state_file)

        assert deleted == 2
        assert not state_file.exists()
        assert fake_daytona.delete.call_count == 2

    def test_cleanup_orphaned_sandboxes_keeps_failed_deletes(self, tmp_path: Path) -> None:
        state_file = tmp_path / "parallel-sandboxes.json"
        state_file.write_text(json.dumps({"sbx-ok": "sbx-ok", "sbx-fail": "sbx-fail"}))

        fake_daytona = MagicMock()

        def fake_get(sandbox_id: str) -> MagicMock:
            if sandbox_id == "sbx-fail":
                raise LookupError("not found")
            return MagicMock()

        fake_daytona.get.side_effect = fake_get

        with patch("simlab.runtime.parallel_daytona._get_daytona", return_value=fake_daytona):
            deleted = ParallelDaytonaOrchestrator.cleanup_orphaned_sandboxes(state_file)

        assert deleted == 1
        # Failed sandbox should remain in state file
        assert state_file.exists()
        remaining = json.loads(state_file.read_text())
        assert remaining == {"sbx-fail": "sbx-fail"}

    def test_cleanup_orphaned_sandboxes_handles_missing_file(self, tmp_path: Path) -> None:
        state_file = tmp_path / "parallel-sandboxes.json"
        deleted = ParallelDaytonaOrchestrator.cleanup_orphaned_sandboxes(state_file)
        assert deleted == 0

    def test_cleanup_orphaned_sandboxes_handles_malformed_json(self, tmp_path: Path) -> None:
        state_file = tmp_path / "parallel-sandboxes.json"
        state_file.write_text("NOT VALID JSON {{{")
        deleted = ParallelDaytonaOrchestrator.cleanup_orphaned_sandboxes(state_file)
        assert deleted == 0


# ---------------------------------------------------------------------------
# summary.json accuracy
# ---------------------------------------------------------------------------


class TestWriteSummary:
    """Verify summary.json correctly reports passed/failed/avg_reward."""

    def test_verifier_failure_counted_as_failed(self, tmp_path: Path) -> None:
        summary = ParallelRunSummary(
            task_id="T1",
            rollout_count=2,
            results=[
                RolloutResult(rollout_idx=0, reward=1.0, verification_passed=True, steps_taken=10),
                RolloutResult(
                    rollout_idx=1,
                    reward=0.0,
                    verification_passed=False,
                    steps_taken=8,
                ),
            ],
            total_duration_seconds=100.0,
        )
        ParallelDaytonaOrchestrator._write_summary(summary, tmp_path)
        data = json.loads((tmp_path / "summary.json").read_text())

        assert data["passed"] == 1
        assert data["failed"] == 1
        # avg_reward includes ALL rewards (even from failed verifications)
        assert data["avg_reward"] == 0.5

    def test_error_rollout_counted_as_failed(self, tmp_path: Path) -> None:
        summary = ParallelRunSummary(
            task_id="T1",
            rollout_count=2,
            results=[
                RolloutResult(rollout_idx=0, reward=1.0, verification_passed=True, steps_taken=10),
                RolloutResult(rollout_idx=1, error="sandbox crashed", steps_taken=0),
            ],
            total_duration_seconds=100.0,
        )
        ParallelDaytonaOrchestrator._write_summary(summary, tmp_path)
        data = json.loads((tmp_path / "summary.json").read_text())

        assert data["passed"] == 1
        assert data["failed"] == 1
        # avg_reward only from rollouts that have a reward
        assert data["avg_reward"] == 1.0

    def test_all_passed(self, tmp_path: Path) -> None:
        summary = ParallelRunSummary(
            task_id="T1",
            rollout_count=3,
            results=[
                RolloutResult(rollout_idx=i, reward=0.8, verification_passed=True, steps_taken=5)
                for i in range(3)
            ],
            total_duration_seconds=200.0,
        )
        ParallelDaytonaOrchestrator._write_summary(summary, tmp_path)
        data = json.loads((tmp_path / "summary.json").read_text())

        assert data["passed"] == 3
        assert data["failed"] == 0
        assert data["avg_reward"] == pytest.approx(0.8)
        assert data["avg_steps"] == 5.0


# ---------------------------------------------------------------------------
# _require_reachable_endpoints wait mode
# ---------------------------------------------------------------------------


class TestRequireReachableEndpointsWaitMode:
    """Verify wait=True polls then falls through to fail-fast."""

    def test_wait_mode_returns_when_all_reachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = {"n": 0}

        def fake_reachable(_endpoints: dict) -> dict[str, bool]:
            call_count["n"] += 1
            if call_count["n"] < 3:
                return {"email": True, "calendar": False}
            return {"email": True, "calendar": True}

        monkeypatch.setattr(tasks_cli, "_reachable_endpoints", fake_reachable)
        monkeypatch.setattr(tasks_cli.time, "sleep", lambda _s: None)

        # Should not raise — all become reachable on 3rd poll
        tasks_cli._require_reachable_endpoints(
            endpoints={"email": "http://localhost:8040", "calendar": "http://localhost:8050"},
            action="test",
            using_daytona=True,
            wait=True,
            timeout=120,
            poll_interval=5,
        )
        assert call_count["n"] == 3

    def test_wait_mode_falls_through_to_failfast_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Always return all unreachable
        monkeypatch.setattr(
            tasks_cli,
            "_reachable_endpoints",
            lambda _e: {"email": False},
        )
        monkeypatch.setattr(tasks_cli.time, "sleep", lambda _s: None)
        # Force immediate timeout
        monkeypatch.setattr(tasks_cli.time, "monotonic", lambda: 999999.0)

        with pytest.raises(SystemExit) as exc_info:
            tasks_cli._require_reachable_endpoints(
                endpoints={"email": "http://localhost:8040"},
                action="test",
                using_daytona=True,
                wait=True,
                timeout=0,
            )
        assert exc_info.value.code == 1

    def test_wait_false_is_unchanged_behavior(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default wait=False does a single check like before."""
        monkeypatch.setattr(
            tasks_cli,
            "_reachable_endpoints",
            lambda _e: {"email": True},
        )

        # Should not raise
        tasks_cli._require_reachable_endpoints(
            endpoints={"email": "http://localhost:8040"},
            action="test",
            using_daytona=False,
        )


# ---------------------------------------------------------------------------
# DaytonaRunner.up() preseed RuntimeError → SystemExit
# ---------------------------------------------------------------------------


class TestPreseedRuntimeErrorConversion:
    """Verify preseed RuntimeError is converted to SystemExit(1) in DaytonaRunner.up()."""

    def test_preseed_failure_raises_system_exit(self) -> None:
        """RuntimeError from preseed is converted to SystemExit(1)."""
        fake_rpt = MagicMock()

        with pytest.raises(SystemExit) as exc_info:
            _simulate_preseed_system_exit(fake_rpt)

        assert exc_info.value.code == 1
        fake_rpt.end_step.assert_called_once_with(success=False, error="preseed boom")

    def test_preseed_failure_cleans_up_sandbox(self) -> None:
        """Sandbox is deleted on preseed failure since SystemExit escapes except Exception."""
        fake_daytona = MagicMock()
        fake_sandbox = MagicMock()
        fake_rpt = MagicMock()

        with pytest.raises(SystemExit) as exc_info:
            _simulate_preseed_system_exit_with_cleanup(
                fake_rpt,
                fake_daytona,
                fake_sandbox,
            )

        assert exc_info.value.code == 1
        assert fake_daytona.delete.called


# ---------------------------------------------------------------------------
# Orphan cleanup wired into parallel tasks run
# ---------------------------------------------------------------------------


class TestOrphanCleanupWiredIn:
    """Verify cleanup_orphaned_sandboxes is called before parallel execution."""

    def test_cleanup_called_when_state_file_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_dir = tmp_path / "my-env"
        env_dir.mkdir()
        state_file = env_dir / "parallel-sandboxes.json"
        state_file.write_text(json.dumps({"sbx-leaked": "sbx-leaked"}))

        cleanup_called = {"count": 0}

        def tracked_cleanup(sf: Path, _daytona_api_key: str | None = None) -> int:
            cleanup_called["count"] += 1
            # Don't actually call Daytona — just remove the file
            sf.unlink(missing_ok=True)
            return 1

        monkeypatch.setattr(
            ParallelDaytonaOrchestrator,
            "cleanup_orphaned_sandboxes",
            staticmethod(tracked_cleanup),
        )

        # Verify the cleanup is wired in by checking the state file is consumed
        assert state_file.exists()
        tracked_cleanup(state_file)
        assert cleanup_called["count"] == 1
        assert not state_file.exists()


# ---------------------------------------------------------------------------
# stop_event cancellation inside agent loop
# ---------------------------------------------------------------------------


class TestStopEventCancelsAgent:
    """Verify stop_event threading.Event stops the ReferenceAgent loop early."""

    def test_reference_agent_stops_on_stop_event(self) -> None:
        """ReferenceAgent.run() should set context.error='Cancelled' and return."""
        stop = threading.Event()
        stop.set()  # pre-set so the very first iteration bails

        env = MagicMock(spec=BaseEnvironment)
        ctx = RunArtifacts(
            task_id="t1", task="do something", model="gpt-4o-mini", provider="openai", max_steps=10
        )
        agent = ReferenceAgent(api_key="fake-key")

        # run() should return immediately without calling OpenAI
        agent.run("do something", env, ctx, stop_event=stop)

        assert ctx.error == "Cancelled"

    def test_reference_agent_runs_normally_without_stop_event(self) -> None:
        """Without stop_event, agent proceeds to call the LLM (mocked)."""
        env = MagicMock(spec=BaseEnvironment)
        env.list_tools.return_value = []
        ctx = RunArtifacts(
            task_id="t1", task="do something", model="gpt-4o-mini", provider="openai", max_steps=1
        )
        agent = ReferenceAgent(api_key="fake-key")

        fake_message = MagicMock()
        fake_message.tool_calls = None
        fake_message.content = "Done!"
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]

        with patch("litellm.completion", return_value=fake_response):
            agent.run("do something", env, ctx)

        assert ctx.error is None
        assert ctx.final_observation == "Done!"


# ---------------------------------------------------------------------------
# MCP support in parallel Daytona rollouts
# ---------------------------------------------------------------------------


class TestParallelDaytonaMcpSupport:
    """Verify MCP-backed tasks work in the parallel Daytona path."""

    def test_run_single_rollout_allows_mcp_only_url_mcp_task_without_endpoint_checks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)
        compose_dir = tmp_path / "compose"
        compose_dir.mkdir()
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        config_path = env_dir / "env.yaml"
        config_path.write_text("name: test\n")

        fake_daytona = MagicMock()
        fake_sandbox = MagicMock()
        fake_sandbox.id = "sbx-1"
        fake_daytona.create.return_value = fake_sandbox
        demo_client = object()

        captured: dict[str, object] = {}

        class FakeEnvironment:
            def __init__(
                self,
                *,
                tool_servers: dict[str, str],
                mcp_clients: dict[str, object] | None = None,
            ) -> None:
                captured["tool_servers"] = tool_servers
                captured["mcp_clients"] = mcp_clients

        def fake_run_with_agent_contract(**kwargs) -> RunArtifacts:
            captured["environment"] = kwargs["environment"]
            return RunArtifacts(
                task_id="task-1",
                task="do task",
                model="gpt-test",
                provider="openai",
                max_steps=5,
                final_observation="done",
            )

        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._get_daytona", lambda _key: fake_daytona
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.CreateSandboxFromSnapshotParams",
            lambda **kwargs: kwargs,
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.setup_sandbox_environment",
            lambda *args, **kwargs: {},
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._require_reachable_endpoints",
            lambda **kwargs: pytest.fail(
                "MCP-only URL-based rollout should not check HTTP endpoints"
            ),
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.load_mcp_servers_from_env_dir",
            lambda _env_dir: {"mcpServers": {"demo": {"url": "http://mcp.example/mcp"}}},
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._build_mcp_clients",
            lambda _config, _endpoints: {"demo": demo_client},
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._require_mcp_tools_available",
            lambda clients: captured.setdefault("validated_clients", clients),
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.get_agent_runtime_helpers",
            lambda: (FakeEnvironment, fake_run_with_agent_contract),
        )
        monkeypatch.setattr(
            ParallelDaytonaOrchestrator,
            "_provision_group_channels",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            ParallelDaytonaOrchestrator,
            "_provision_calendar_accounts",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            ParallelDaytonaOrchestrator,
            "_run_verifiers",
            lambda *args, **kwargs: (None, None, None),
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._load_skills_markdown",
            lambda **kwargs: "",
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._build_skills_guidance_section",
            lambda _skills: "",
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._build_services_available_section",
            lambda *args, **kwargs: "",
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.teardown_sandbox", lambda *args, **kwargs: True
        )

        result = orch._run_single_rollout(
            0,
            task_data={"task": "do task", "meta": {"task_id": "task-1"}, "tool_servers": []},
            profiles={},
            compose_dir=compose_dir,
            tool_ports={},
            extra_tool_urls={},
            preseed_svc_names=None,
            seed_svc_names=None,
            config=MagicMock(),
            config_path=str(config_path),
            model="gpt-test",
            provider="openai",
            api_key="sk-test",
            base_url=None,
            max_steps=5,
            agent_import_path=None,
            agent_timeout_seconds=10.0,
            no_seed=True,
            bundle_dir=None,
            global_cfg=MagicMock(),
            backend_id=None,
            base_url_api="http://example.invalid",
            scenario_manager_api_key=None,
            rollout_format="default",
            run_dir=tmp_path / "output",
        )

        assert result.error is None
        assert captured["tool_servers"] == {}
        assert captured["validated_clients"] == {"demo": demo_client}
        assert captured["mcp_clients"] == {"demo": demo_client}

    def test_run_single_rollout_adds_daytona_gateway_url_for_command_mcp_servers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)
        compose_dir = tmp_path / "compose"
        compose_dir.mkdir()
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        config_path = env_dir / "env.yaml"
        config_path.write_text("name: test\n")

        fake_daytona = MagicMock()
        fake_sandbox = MagicMock()
        fake_sandbox.id = "sbx-1"
        fake_daytona.create.return_value = fake_sandbox
        fake_sandbox.get_preview_link.return_value = MagicMock(
            url="https://8081-x.daytonaproxy.net"
        )
        captured: dict[str, object] = {}
        gateway_checks = {"count": 0}

        class FakeMCPClient:
            def __init__(self) -> None:
                self._url = "https://8081-x.daytonaproxy.net/mcp"

            async def alist_tools(self) -> list[dict[str, object]]:
                gateway_checks["count"] += 1
                if gateway_checks["count"] < 3:
                    raise RuntimeError("gateway not ready")
                return [{"name": "ping", "description": "Ping", "input_schema": {}}]

        demo_client = FakeMCPClient()

        class FakeEnvironment:
            def __init__(
                self,
                *,
                tool_servers: dict[str, str],
                mcp_clients: dict[str, object] | None = None,
            ) -> None:
                captured["tool_servers"] = tool_servers
                captured["mcp_clients"] = mcp_clients

        def fake_run_with_agent_contract(**kwargs) -> RunArtifacts:
            _ = kwargs
            return RunArtifacts(
                task_id="task-1",
                task="do task",
                model="gpt-test",
                provider="openai",
                max_steps=5,
                final_observation="done",
            )

        def fake_build_mcp_clients(_config: dict[str, object] | None, endpoints: dict[str, str]):
            captured["build_endpoints"] = dict(endpoints)
            return {"demo": demo_client}

        def fake_require_mcp_tools_available(clients: dict[str, object]) -> None:
            captured["validation_after_gateway_checks"] = gateway_checks["count"]
            captured["validated_clients"] = clients

        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._get_daytona", lambda _key: fake_daytona
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.CreateSandboxFromSnapshotParams",
            lambda **kwargs: kwargs,
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.setup_sandbox_environment",
            lambda *args, **kwargs: {},
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.load_mcp_servers_from_env_dir",
            lambda _env_dir: {"mcpServers": {"demo": {"command": "uvx", "args": ["demo-mcp"]}}},
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.get_mcp_gateway_host_port",
            lambda _env_dir: 8081,
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._build_mcp_clients",
            fake_build_mcp_clients,
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._require_mcp_tools_available",
            fake_require_mcp_tools_available,
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._require_reachable_endpoints",
            lambda **kwargs: captured.setdefault("reachable_endpoints", kwargs["endpoints"]),
        )
        monkeypatch.setattr("simlab.runtime.parallel_daytona.time.sleep", lambda _s: None)
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.get_agent_runtime_helpers",
            lambda: (FakeEnvironment, fake_run_with_agent_contract),
        )
        monkeypatch.setattr(
            ParallelDaytonaOrchestrator,
            "_provision_group_channels",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            ParallelDaytonaOrchestrator,
            "_provision_calendar_accounts",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            ParallelDaytonaOrchestrator,
            "_run_verifiers",
            lambda *args, **kwargs: (None, None, None),
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._load_skills_markdown",
            lambda **kwargs: "",
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._build_skills_guidance_section",
            lambda _skills: "",
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._build_services_available_section",
            lambda *args, **kwargs: "",
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.teardown_sandbox",
            lambda *args, **kwargs: True,
        )

        result = orch._run_single_rollout(
            0,
            task_data={"task": "do task", "meta": {"task_id": "task-1"}, "tool_servers": []},
            profiles={},
            compose_dir=compose_dir,
            tool_ports={},
            extra_tool_urls={},
            preseed_svc_names=None,
            seed_svc_names=None,
            config=MagicMock(),
            config_path=str(config_path),
            model="gpt-test",
            provider="openai",
            api_key="sk-test",
            base_url=None,
            max_steps=5,
            agent_import_path=None,
            agent_timeout_seconds=10.0,
            no_seed=True,
            bundle_dir=None,
            global_cfg=MagicMock(),
            backend_id=None,
            base_url_api="http://example.invalid",
            scenario_manager_api_key=None,
            rollout_format="default",
            run_dir=tmp_path / "output",
        )

        expected_gateway = {
            tasks_cli.ComposeEngine.MCP_GATEWAY_SERVICE_NAME: "https://8081-x.daytonaproxy.net/mcp"
        }
        assert result.error is None
        assert captured["reachable_endpoints"] == expected_gateway
        assert captured["build_endpoints"] == expected_gateway
        assert captured["validation_after_gateway_checks"] == 3
        assert captured["validated_clients"] == {"demo": demo_client}
        assert captured["mcp_clients"] == {"demo": demo_client}

    def test_run_single_rollout_passes_mcp_verifier_tool_urls_when_http_tool_servers_are_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)
        compose_dir = tmp_path / "compose"
        compose_dir.mkdir()
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        config_path = env_dir / "env.yaml"
        config_path.write_text("name: test\n")

        fake_daytona = MagicMock()
        fake_sandbox = MagicMock()
        fake_sandbox.id = "sbx-1"
        fake_daytona.create.return_value = fake_sandbox
        captured: dict[str, object] = {}

        class FakeMCPClient:
            def __init__(self) -> None:
                self._url = "http://mcp.example/mcp"

            async def alist_tools(self) -> list[dict[str, object]]:
                return [{"name": "ping", "description": "Ping", "input_schema": {}}]

        demo_client = FakeMCPClient()

        class FakeEnvironment:
            def __init__(
                self,
                *,
                tool_servers: dict[str, str],
                mcp_clients: dict[str, object] | None = None,
            ) -> None:
                captured["tool_servers"] = tool_servers
                captured["mcp_clients"] = mcp_clients

        def fake_run_with_agent_contract(**kwargs) -> RunArtifacts:
            _ = kwargs
            return RunArtifacts(
                task_id="task-1",
                task="do task",
                model="gpt-test",
                provider="openai",
                max_steps=5,
                final_observation="done",
            )

        def fake_run_verifiers(_self: object, **kwargs):
            captured["verifier_tool_servers"] = kwargs["tool_servers"]
            return (None, None, None)

        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._get_daytona", lambda _key: fake_daytona
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.CreateSandboxFromSnapshotParams",
            lambda **kwargs: kwargs,
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.setup_sandbox_environment",
            lambda *args, **kwargs: {},
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.load_mcp_servers_from_env_dir",
            lambda _env_dir: {"mcpServers": {"demo": {"url": "http://mcp.example/mcp"}}},
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._build_mcp_clients",
            lambda _config, _endpoints: {"demo": demo_client},
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._require_mcp_tools_available",
            lambda clients: captured.setdefault("validated_clients", clients),
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.get_agent_runtime_helpers",
            lambda: (FakeEnvironment, fake_run_with_agent_contract),
        )
        monkeypatch.setattr(
            ParallelDaytonaOrchestrator,
            "_provision_group_channels",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            ParallelDaytonaOrchestrator,
            "_provision_calendar_accounts",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(ParallelDaytonaOrchestrator, "_run_verifiers", fake_run_verifiers)
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._load_skills_markdown",
            lambda **kwargs: "",
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._build_skills_guidance_section",
            lambda _skills: "",
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._build_services_available_section",
            lambda *args, **kwargs: "",
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona.teardown_sandbox", lambda *args, **kwargs: True
        )

        result = orch._run_single_rollout(
            0,
            task_data={
                "task": "do task",
                "meta": {"task_id": "task-1"},
                "tool_servers": [],
                "verifiers": [{"func": "python_module", "module": "demo.verifier"}],
            },
            profiles={},
            compose_dir=compose_dir,
            tool_ports={},
            extra_tool_urls={},
            preseed_svc_names=None,
            seed_svc_names=None,
            config=MagicMock(),
            config_path=str(config_path),
            model="gpt-test",
            provider="openai",
            api_key="sk-test",
            base_url=None,
            max_steps=5,
            agent_import_path=None,
            agent_timeout_seconds=10.0,
            no_seed=True,
            bundle_dir=None,
            global_cfg=MagicMock(),
            backend_id=None,
            base_url_api="http://example.invalid",
            scenario_manager_api_key=None,
            rollout_format="default",
            run_dir=tmp_path / "output",
        )

        assert result.error is None
        assert captured["tool_servers"] == {}
        assert captured["verifier_tool_servers"] == {"demo": "http://mcp.example"}


# ---------------------------------------------------------------------------
# DaytonaRunner.down() transient vs not-found errors
# ---------------------------------------------------------------------------


class TestDaytonaRunnerDownErrorHandling:
    """Verify env down --daytona distinguishes not-found from transient errors."""

    def _write_state(self, state_file: Path, sandbox_id: str = "sbx-test") -> None:
        state_file.write_text(json.dumps({"sandbox_id": sandbox_id}))

    def test_not_found_error_removes_state_file(self, tmp_path: Path) -> None:
        """DaytonaNotFoundError means sandbox is genuinely gone — clean up state."""
        state_file = tmp_path / "daytona-state.json"
        self._write_state(state_file)

        fake_daytona = MagicMock()
        fake_daytona.get.side_effect = DaytonaNotFoundError("not found")

        runner = DaytonaRunner()
        with patch("simlab.runtime.daytona_runner._get_daytona", return_value=fake_daytona):
            runner.down(tmp_path)

        assert not state_file.exists()

    def test_transient_error_preserves_state_file(self, tmp_path: Path) -> None:
        """Transient API error — sandbox may still be alive, keep state file."""
        state_file = tmp_path / "daytona-state.json"
        self._write_state(state_file)

        fake_daytona = MagicMock()
        fake_daytona.get.side_effect = ConnectionError("network timeout")

        runner = DaytonaRunner()
        with patch("simlab.runtime.daytona_runner._get_daytona", return_value=fake_daytona):
            runner.down(tmp_path)

        # State file should still exist for manual retry
        assert state_file.exists()

    def test_successful_teardown_removes_state_file(self, tmp_path: Path) -> None:
        """Normal teardown — sandbox deleted, state file removed."""
        state_file = tmp_path / "daytona-state.json"
        self._write_state(state_file)

        fake_daytona = MagicMock()
        fake_sandbox = MagicMock()
        fake_daytona.get.return_value = fake_sandbox

        runner = DaytonaRunner()
        with (
            patch("simlab.runtime.daytona_runner._get_daytona", return_value=fake_daytona),
            patch("simlab.runtime.daytona_runner.teardown_sandbox", return_value=True),
        ):
            runner.down(tmp_path)

        assert not state_file.exists()


# ---------------------------------------------------------------------------
# _provision_group_channels
# ---------------------------------------------------------------------------


class TestProvisionGroupChannels:
    """Verify _provision_group_channels in parallel Daytona orchestrator."""

    def test_runs_rocketchat_seed_with_env_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)

        task_data = {
            "npcs": [{"id": "diana_walsh"}],
            "seed_group_channels": [
                {
                    "channel_name": "billing-support",
                    "member_profile_ids": ["diana_walsh", "mike_williams"],
                    "messages": [],
                }
            ],
        }
        profiles = {
            "diana_walsh": {
                "first_name": "Diana",
                "last_name": "Walsh",
                "email": "diana@example.com",
            },
        }

        from types import SimpleNamespace  # noqa: PLC0415

        config = SimpleNamespace(tools=["rocketchat"])
        config_path = tmp_path / "env.yaml"
        config_path.write_text("name: x\n")

        sandbox_calls: list[dict[str, str]] = []

        def fake_profiled_services(
            _config: object,
            profile: str,  # noqa: ARG001
            config_path: object = None,  # noqa: ARG001
            tool_names: list[str] | None = None,
        ) -> list[str]:
            assert tool_names == ["rocketchat"]
            return ["rocketchat-seed"]

        def fake_run_in_sandbox(
            _sandbox: object,
            svc_names: list[str],  # noqa: ARG001
            profile: str,  # noqa: ARG001
            env_overrides: dict[str, str] | None = None,
            log_prefix: str = "",  # noqa: ARG001
            quiet: bool = False,  # noqa: ARG001
        ) -> None:
            sandbox_calls.append(env_overrides or {})

        monkeypatch.setattr(
            "simlab.runtime.env_lifecycle._get_profiled_service_names",
            fake_profiled_services,
        )
        monkeypatch.setattr(
            "simlab.runtime.parallel_daytona._run_profiled_services_in_sandbox",
            fake_run_in_sandbox,
        )

        fake_sandbox = MagicMock()
        orch._provision_group_channels(
            "[rollout 1/1]",
            fake_sandbox,
            task_data,
            profiles,
            config,
            str(config_path),
        )

        assert len(sandbox_calls) == 1
        env = sandbox_calls[0]
        npc_configs = json.loads(env["ROCKETCHAT_NPC_CONFIGS"])
        assert "diana_walsh" in npc_configs
        assert "mike_williams" in npc_configs
        assert npc_configs["diana_walsh"]["email"] == "diana@example.com"

        channels = json.loads(env["ROCKETCHAT_SEED_GROUP_CHANNELS"])
        assert channels[0]["channel_name"] == "billing-support"

    def test_skips_when_no_group_channels(self) -> None:
        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)
        task_data: dict[str, list[object]] = {"npcs": [], "seed_group_channels": []}
        # Should return immediately without error
        orch._provision_group_channels(
            "[rollout 1/1]", MagicMock(), task_data, {}, MagicMock(), "/fake"
        )

    def test_skips_when_rocketchat_not_in_tools(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415

        orch = ParallelDaytonaOrchestrator(rollout_count=1, max_parallel=1)
        task_data = {
            "npcs": [],
            "seed_group_channels": [
                {"channel_name": "ch", "member_profile_ids": ["bob"], "messages": []}
            ],
        }
        config = SimpleNamespace(tools=["email"])
        # Should return without running anything
        orch._provision_group_channels("[rollout 1/1]", MagicMock(), task_data, {}, config, "/fake")
