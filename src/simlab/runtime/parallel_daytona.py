"""Parallel rollout orchestrator for Daytona sandboxes."""

from __future__ import annotations

import atexit
import contextlib
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import click

from simlab.cli.progress import ParallelRolloutProgressLike
from simlab.composer.engine import ComposeEngine
from simlab.composer.engine import get_mcp_gateway_host_port
from simlab.mcp_config import get_mcp_command_servers
from simlab.mcp_config import load_mcp_servers_from_env_dir
from simlab.npc_chat.activation import NpcChatSession
from simlab.runtime.adapters.harbor import trajectory as harbor_trajectory
from simlab.runtime.daytona_runner import _SNAPSHOT_NAME
from simlab.runtime.daytona_runner import CreateSandboxFromSnapshotParams
from simlab.runtime.daytona_runner import _get_daytona
from simlab.runtime.daytona_runner import _run_profiled_services_in_sandbox
from simlab.runtime.daytona_runner import setup_sandbox_environment
from simlab.runtime.daytona_runner import teardown_sandbox
from simlab.runtime.rollout_runner import CALENDAR_DEFAULT_USERNAME
from simlab.runtime.rollout_runner import ROLLOUT_FORMAT_ATIF
from simlab.runtime.rollout_runner import ROLLOUT_FORMAT_DEFAULT
from simlab.runtime.rollout_runner import apply_verifier_env_overrides
from simlab.runtime.rollout_runner import build_mcp_clients
from simlab.runtime.rollout_runner import build_services_available_section
from simlab.runtime.rollout_runner import build_skills_guidance_section
from simlab.runtime.rollout_runner import collect_task_calendar_accounts
from simlab.runtime.rollout_runner import effective_tool_servers
from simlab.runtime.rollout_runner import ensure_task_calendar_accounts
from simlab.runtime.rollout_runner import get_agent_runtime_helpers
from simlab.runtime.rollout_runner import get_env_runtime_helpers
from simlab.runtime.rollout_runner import get_local_verifier_file_path
from simlab.runtime.rollout_runner import get_verifier_runtime_helpers
from simlab.runtime.rollout_runner import load_skills_markdown
from simlab.runtime.rollout_runner import maybe_run_rubric_judge
from simlab.runtime.rollout_runner import require_mcp_tools_available
from simlab.runtime.rollout_runner import require_reachable_endpoints
from simlab.runtime.rollout_runner import restore_env
from simlab.runtime.rollout_runner import rewrite_tool_server_urls
from simlab.runtime.rollout_runner import seed_task_data
from simlab.runtime.rollout_runner import task_uses_calendar
from simlab.runtime.rollout_runner import verifier_tool_servers
from simlab.runtime.rollout_runner import wait_for_mcp_tools_available

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """Outcome of a single parallel rollout."""

    rollout_idx: int
    sandbox_id: str = ""
    reward: float | None = None
    verification_passed: bool | None = None
    error: str | None = None
    steps_taken: int = 0
    duration_seconds: float = 0.0
    artifacts_path: Path | None = None


@dataclass
class ParallelRunSummary:
    """Aggregated results across all parallel rollouts."""

    task_id: str
    rollout_count: int
    results: list[RolloutResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    output_dir: Path | None = None


def _rollout_result_label(result: RolloutResult) -> str:
    if result.error:
        return "ERR"
    if result.verification_passed is True:
        return "PASS"
    if result.verification_passed is False:
        return "FAIL"
    return "N/A"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ParallelDaytonaOrchestrator:
    """Run N rollouts of the same task across isolated Daytona sandboxes."""

    def __init__(  # noqa: D107
        self,
        rollout_count: int,
        max_parallel: int,
        daytona_api_key: str | None = None,
    ) -> None:
        self._rollout_count = rollout_count
        self._max_parallel = max_parallel
        self._daytona_api_key = daytona_api_key

        # Thread-safe tracking of active sandboxes for cleanup.
        self._lock = threading.Lock()
        self._active_sandboxes: dict[int, tuple[Any, Any]] = {}  # idx -> (client, sandbox)

        # Durable state file tracking sandbox IDs for crash recovery.
        self._state_file: Path | None = None

        # Serialize verifier env overrides (os.environ is process-wide).
        self._verifier_env_lock = threading.Lock()

        self._cleanup_registered = False

        # Cancellation event — set on Ctrl+C so in-flight workers exit early.
        self._cancel = threading.Event()

    # -- durable sandbox state -----------------------------------------------

    def _persist_sandbox(self, sandbox_id: str) -> None:
        """Record a sandbox ID on disk so it can be cleaned up after a crash."""
        if self._state_file is None:
            return
        with self._lock:
            state = self._read_state_file()
            state[sandbox_id] = sandbox_id
            self._write_state_file(state)

    def _unpersist_sandbox(self, sandbox_id: str) -> None:
        """Remove a sandbox ID from the durable state after successful teardown."""
        if self._state_file is None:
            return
        with self._lock:
            state = self._read_state_file()
            state.pop(sandbox_id, None)
            self._write_state_file(state)

    def _read_state_file(self) -> dict[str, str]:
        if self._state_file is None or not self._state_file.exists():
            return {}
        with contextlib.suppress(json.JSONDecodeError, OSError):
            return json.loads(self._state_file.read_text())
        return {}

    def _write_state_file(self, state: dict[str, str]) -> None:
        if self._state_file is None:
            return
        if state:
            self._state_file.write_text(json.dumps(state, indent=2))
        else:
            self._state_file.unlink(missing_ok=True)

    @staticmethod
    def cleanup_orphaned_sandboxes(
        state_file: Path,
        daytona_api_key: str | None = None,
    ) -> int:
        """Clean up sandboxes leaked by a crashed parallel run.

        Reads *state_file* (``parallel-sandboxes.json``) and attempts to
        delete every sandbox listed.  Successfully deleted entries are removed
        from the file.  Returns the number of sandboxes deleted.
        """
        if not state_file.exists():
            return 0

        state: dict[str, str] = {}
        with contextlib.suppress(json.JSONDecodeError, OSError):
            state = json.loads(state_file.read_text())
        if not state:
            return 0

        daytona = _get_daytona(daytona_api_key)
        deleted = 0
        failed: dict[str, str] = {}

        for sandbox_id in list(state.values()):
            try:
                sandbox = daytona.get(sandbox_id)
                daytona.delete(sandbox)
                click.echo(f"  Deleted orphaned sandbox {sandbox_id}")
                deleted += 1
            except Exception as exc:
                click.echo(
                    click.style(
                        f"  Failed to delete sandbox {sandbox_id}: {exc}",
                        fg="yellow",
                    ),
                    err=True,
                )
                failed[sandbox_id] = sandbox_id

        if failed:
            state_file.write_text(json.dumps(failed, indent=2))
        else:
            state_file.unlink(missing_ok=True)

        return deleted

    # -- public API ----------------------------------------------------------

    def execute(
        self,
        *,
        task_id: str,
        task_data: dict[str, Any],
        profiles: dict[str, dict[str, Any]],
        compose_dir: Path,
        tool_ports: dict[str, int],
        extra_tool_urls: dict[str, str] | None = None,
        preseed_svc_names: list[str] | None = None,
        seed_svc_names: list[str] | None = None,
        config: Any,  # noqa: ANN401
        config_path: str,
        model: str,
        provider: str,
        api_key: str | None,
        base_url: str | None,
        max_steps: int,
        agent_import_path: str | None,
        agent_timeout_seconds: float,
        no_seed: bool,
        bundle_dir: Path | None,
        global_cfg: Any,  # noqa: ANN401
        backend_id: str | None,
        base_url_api: str,
        scenario_manager_api_key: str | None,
        rollout_format: str,
        output_root: Path = Path("output"),
        progress: ParallelRolloutProgressLike | None = None,
    ) -> ParallelRunSummary:
        """Execute all rollouts and return aggregated summary."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"parallel_run_{task_id}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self._state_file = compose_dir / "parallel-sandboxes.json"

        # Register cleanup once.
        if not self._cleanup_registered:
            atexit.register(self._cleanup_all)
            self._cleanup_registered = True

        summary = ParallelRunSummary(
            task_id=task_id,
            rollout_count=self._rollout_count,
            output_dir=run_dir,
        )

        if progress is None:
            click.echo(
                click.style(
                    f"\nStarting {self._rollout_count} parallel rollout(s) "
                    f"(max {self._max_parallel} concurrent)\n",
                    bold=True,
                )
            )
        start = time.monotonic()

        kwargs = {
            "task_data": task_data,
            "profiles": profiles,
            "compose_dir": compose_dir,
            "tool_ports": tool_ports,
            "extra_tool_urls": extra_tool_urls or {},
            "preseed_svc_names": preseed_svc_names,
            "seed_svc_names": seed_svc_names,
            "config": config,
            "config_path": config_path,
            "model": model,
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "max_steps": max_steps,
            "agent_import_path": agent_import_path,
            "agent_timeout_seconds": agent_timeout_seconds,
            "no_seed": no_seed,
            "bundle_dir": bundle_dir,
            "global_cfg": global_cfg,
            "backend_id": backend_id,
            "base_url_api": base_url_api,
            "scenario_manager_api_key": scenario_manager_api_key,
            "rollout_format": rollout_format,
            "run_dir": run_dir,
            "progress": progress,
        }

        executor = ThreadPoolExecutor(max_workers=self._max_parallel)
        futures = {
            executor.submit(self._run_single_rollout, idx, **kwargs): idx
            for idx in range(self._rollout_count)
        }

        try:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except (Exception, SystemExit) as exc:
                    result = RolloutResult(
                        rollout_idx=idx,
                        error=str(exc),
                    )
                summary.results.append(result)
        except KeyboardInterrupt:
            click.echo(
                click.style(
                    "\nInterrupted — cancelling pending rollouts and cleaning up...",
                    fg="yellow",
                )
            )
            # Signal in-flight workers to exit at their next checkpoint.
            self._cancel.set()
            # Cancel queued (not-yet-started) futures.
            for fut in futures:
                fut.cancel()
            # Don't block on long-running Daytona/agent calls — shut down
            # immediately and let _cleanup_all tear down any active sandboxes.
            executor.shutdown(wait=False, cancel_futures=True)
            self._cleanup_all()
            raise
        else:
            executor.shutdown(wait=True)

        summary.results.sort(key=lambda r: r.rollout_idx)
        summary.total_duration_seconds = time.monotonic() - start

        # Write summary.json
        self._write_summary(summary, run_dir)

        return summary

    def _check_cancelled(self, tag: str) -> None:
        """Raise if cancellation was requested (Ctrl+C)."""
        if self._cancel.is_set():
            raise RuntimeError(f"{tag} Cancelled")

    @staticmethod
    def _wait_for_mcp_gateway(
        gateway_url: str,
        *,
        mcp_clients: dict[str, Any] | None = None,
        timeout: int = 120,
        poll_interval: int = 5,
        log_prefix: str = "",
        quiet: bool = False,
    ) -> None:
        """Wait until the gateway-backed MCP namespaces expose tools."""
        if mcp_clients:
            wait_for_mcp_tools_available(
                mcp_clients,
                timeout=timeout,
                poll_interval=poll_interval,
                log_prefix=log_prefix,
                quiet=quiet,
            )
            return

        raise RuntimeError(f"{log_prefix} No MCP clients were available for gateway {gateway_url}")

    # -- single rollout lifecycle --------------------------------------------

    def _run_single_rollout(
        self,
        rollout_idx: int,
        *,
        task_data: dict[str, Any],
        profiles: dict[str, dict[str, Any]],
        compose_dir: Path,
        tool_ports: dict[str, int],
        extra_tool_urls: dict[str, str],
        preseed_svc_names: list[str] | None,
        seed_svc_names: list[str] | None,
        config: Any,  # noqa: ANN401
        config_path: str,
        model: str,
        provider: str,
        api_key: str | None,
        base_url: str | None,
        max_steps: int,
        agent_import_path: str | None,
        agent_timeout_seconds: float,
        no_seed: bool,
        bundle_dir: Path | None,
        global_cfg: Any,  # noqa: ANN401
        backend_id: str | None,
        base_url_api: str,
        scenario_manager_api_key: str | None,
        rollout_format: str,
        run_dir: Path,
        progress: ParallelRolloutProgressLike | None = None,
    ) -> RolloutResult:
        """Full lifecycle for one rollout: create -> setup -> seed -> run -> verify -> teardown."""
        tag = f"[rollout {rollout_idx + 1}/{self._rollout_count}]"
        result = RolloutResult(rollout_idx=rollout_idx)
        if progress is not None:
            progress.update(rollout_idx, status="Starting")
        rollout_start = time.monotonic()
        env_dir = Path(config_path).parent
        mcp_config = load_mcp_servers_from_env_dir(env_dir)

        daytona_client = _get_daytona(self._daytona_api_key)
        sandbox = None

        try:
            # 1. Create sandbox
            self._check_cancelled(tag)
            if progress is None:
                click.echo(f"{tag} Creating Daytona sandbox...")
            sandbox = daytona_client.create(
                CreateSandboxFromSnapshotParams(
                    snapshot=_SNAPSHOT_NAME,
                    public=True,
                ),
            )
            result.sandbox_id = sandbox.id

            with self._lock:
                self._active_sandboxes[rollout_idx] = (daytona_client, sandbox)
            self._persist_sandbox(sandbox.id)

            # 2. Setup environment
            self._check_cancelled(tag)
            endpoints = setup_sandbox_environment(
                sandbox,
                compose_dir,
                tool_ports,
                preseed_svc_names,
                log_prefix=tag,
                quiet=progress is not None,
            )

            # Include external URL-backed tools in endpoints.
            if extra_tool_urls:
                endpoints.update(extra_tool_urls)

            command_mcp_servers = get_mcp_command_servers(mcp_config) if mcp_config else {}
            if command_mcp_servers:
                preview = sandbox.get_preview_link(get_mcp_gateway_host_port(env_dir))
                endpoints[ComposeEngine.MCP_GATEWAY_SERVICE_NAME] = preview.url.rstrip("/") + "/mcp"

            # Wait for all resolved HTTP tool servers (and gateway, if present) to become reachable.
            if endpoints:
                if progress is None:
                    click.echo(f"{tag} Waiting for tool servers...")
                require_reachable_endpoints(
                    endpoints=endpoints,
                    action="tool server readiness",
                    using_daytona=True,
                    wait=True,
                    timeout=120,
                    poll_interval=5,
                    log_prefix=tag,
                    quiet=progress is not None,
                )

            if not no_seed:
                self._check_cancelled(tag)
                if seed_svc_names:
                    if progress is None:
                        click.echo(f"{tag} Running seed services...")
                    _run_profiled_services_in_sandbox(
                        sandbox,
                        seed_svc_names,
                        profile="seed",
                        log_prefix=tag,
                        quiet=progress is not None,
                    )

                self._check_cancelled(tag)
                self._provision_group_channels(
                    tag,
                    sandbox,
                    task_data,
                    profiles,
                    config,
                    config_path,
                    log=progress is None,
                )
                self._provision_calendar_accounts(
                    tag,
                    sandbox,
                    task_data,
                    profiles,
                    endpoints,
                    config,
                    config_path,
                    log=progress is None,
                )
                self._seed_rollout(tag, task_data, profiles, endpoints, log=progress is None)

            # 4. Build environment + run agent
            self._check_cancelled(tag)
            mcp_clients = build_mcp_clients(mcp_config, endpoints)
            gateway_url = endpoints.get(ComposeEngine.MCP_GATEWAY_SERVICE_NAME)
            if command_mcp_servers and gateway_url:
                self._wait_for_mcp_gateway(
                    gateway_url,
                    mcp_clients=mcp_clients,
                    timeout=120,
                    poll_interval=5,
                    log_prefix=tag,
                    quiet=progress is not None,
                )
            require_mcp_tools_available(mcp_clients)
            rewritten = rewrite_tool_server_urls(task_data, endpoints)
            tool_namespace_endpoints = effective_tool_servers(rewritten, endpoints)

            # Auto-activate NPC chat tool if any NPC has chat personality fields.
            npc_session = NpcChatSession.from_task_data(
                task_data,
                model=model,
                api_key=api_key,
                base_url=base_url,
                provider=provider,
            )
            if npc_session is not None:
                npc_url = npc_session.start()
                tool_namespace_endpoints["npc-chat"] = npc_url
                if progress is None:
                    click.echo(f"{tag} NPC chat tool auto-activated")

            artifacts = None
            try:
                if not tool_namespace_endpoints and not mcp_clients:
                    raise RuntimeError(
                        "Task has no valid tool_servers and no MCP servers after URL resolution"
                    )

                unified_tool_env_cls, run_with_agent_contract = get_agent_runtime_helpers()
                environment = unified_tool_env_cls(
                    tool_servers=tool_namespace_endpoints,
                    mcp_clients=mcp_clients or None,
                )

                instruction = rewritten.get("task", "")
                skills_section = build_skills_guidance_section(
                    load_skills_markdown(config=config, bundle_dir=bundle_dir)
                )
                if skills_section:
                    instruction = f"{instruction}\n\n{skills_section}"
                services_section = build_services_available_section(
                    config,
                    daytona=True,
                    config_path=config_path,
                    endpoints=endpoints,
                )
                if services_section:
                    instruction = f"{instruction}\n\n{services_section}"

                meta = task_data.get("meta", {})
                on_step_callback = None
                if progress is None:
                    click.echo(f"{tag} Running agent ({model} via {provider})...")
                else:
                    progress.update(rollout_idx, status="Running", steps_taken=0)

                    def on_step(steps_taken: int) -> None:
                        progress.update(rollout_idx, steps_taken=steps_taken)

                    on_step_callback = on_step

                artifacts = run_with_agent_contract(
                    task_id=meta.get("task_id", ""),
                    instruction=instruction,
                    model=model,
                    provider=provider,
                    max_steps=max_steps,
                    environment=environment,
                    agent_import_path=agent_import_path,
                    timeout_seconds=agent_timeout_seconds,
                    api_key=api_key,
                    base_url=base_url,
                    stop_event=self._cancel,
                    on_step=on_step_callback,
                )
            finally:
                if npc_session is not None:
                    if artifacts is not None:
                        npc_session.attach_to_artifacts(artifacts)
                    npc_session.stop()

            result.steps_taken = artifacts.steps_taken

            # 5. Save artifacts
            rollout_dir = run_dir / f"rollout_{rollout_idx}"
            rollout_dir.mkdir(parents=True, exist_ok=True)

            if npc_session is not None:
                npc_session.save_transcripts(rollout_dir)

            if rollout_format == ROLLOUT_FORMAT_DEFAULT:
                output_path = rollout_dir / "artifacts.json"
                artifacts.dump(output_path)
                result.artifacts_path = output_path

            if artifacts.error and progress is None:
                click.echo(click.style(f"{tag} Agent error: {artifacts.error}", fg="yellow"))

            # 6. Run verifiers
            self._check_cancelled(tag)
            if progress is not None:
                progress.update(rollout_idx, status="Verifying")
            result.reward, result.verification_passed, reward_payload = self._run_verifiers(
                tag=tag,
                task_data=task_data,
                artifacts=artifacts,
                tool_servers=verifier_tool_servers(rewritten, endpoints, mcp_clients),
                rollout_dir=rollout_dir,
                bundle_dir=bundle_dir,
                global_cfg=global_cfg,
                backend_id=backend_id,
                base_url_api=base_url_api,
                scenario_manager_api_key=scenario_manager_api_key,
                env_dir=env_dir,
                log=progress is None,
            )

            if rollout_format == ROLLOUT_FORMAT_ATIF:
                output_path = rollout_dir / "agent" / "trajectory.json"
                harbor_trajectory.write_atif_trajectory(
                    output_path,
                    artifacts=artifacts,
                    run_id=rollout_dir.name,
                    verification_passed=result.verification_passed,
                    reward=result.reward,
                    reward_payload=reward_payload,
                )
                result.artifacts_path = output_path

            # Match single-rollout behavior: agent error + no verifiers = failure.
            if artifacts.error and result.verification_passed is None:
                result.error = f"Agent error (no verifiers): {artifacts.error}"

            if progress is not None:
                progress.update(
                    rollout_idx,
                    status="Done",
                    result=_rollout_result_label(result),
                )
            else:
                status = "passed" if result.verification_passed else "failed"
                if result.verification_passed is None:
                    status = "no verifiers"
                click.echo(
                    click.style(
                        f"{tag} Completed — reward={result.reward}, "
                        f"steps={result.steps_taken}, {status}",
                        fg="green" if result.verification_passed else "yellow",
                    )
                )

        except (Exception, SystemExit) as exc:
            result.error = str(exc)
            if progress is not None:
                progress.update(rollout_idx, status="Done", result="ERR")
            else:
                click.echo(click.style(f"{tag} Failed: {exc}", fg="red"), err=True)

        finally:
            # 7. Teardown sandbox
            if sandbox is not None:
                if progress is None:
                    click.echo(f"{tag} Tearing down sandbox...")
                deleted = teardown_sandbox(daytona_client, sandbox, log_prefix=tag)
                if deleted:
                    with self._lock:
                        self._active_sandboxes.pop(rollout_idx, None)
                    self._unpersist_sandbox(sandbox.id)
                else:
                    click.echo(
                        click.style(
                            f"{tag} Sandbox delete failed; will retry in cleanup.",
                            fg="yellow",
                        ),
                        err=True,
                    )

            result.duration_seconds = time.monotonic() - rollout_start

        return result

    # -- helpers -------------------------------------------------------------

    def _provision_calendar_accounts(
        self,
        tag: str,
        sandbox: Any,  # noqa: ANN401
        task_data: dict[str, Any],
        profiles: dict[str, dict[str, Any]],
        endpoints: dict[str, str],
        config: Any,  # noqa: ANN401
        config_path: str,
        *,
        log: bool = True,
    ) -> None:
        """Provision CalDAV users and register calendar accounts in ephemeral sandbox."""
        if not task_uses_calendar(task_data):
            return

        accounts = [
            a
            for a in collect_task_calendar_accounts(task_data)
            if a and a != CALENDAR_DEFAULT_USERNAME
        ]
        if not accounts:
            return

        if log:
            click.echo(f"{tag} Provisioning calendar users: {', '.join(accounts)}")

        get_profiled_service_names, _ = get_env_runtime_helpers()
        config_file = Path(config_path)

        preseed_svc_names = get_profiled_service_names(
            config,
            profile="preseed",
            config_path=config_file,
            tool_names=["calendar"],
        )
        seed_svc_names = get_profiled_service_names(
            config,
            profile="seed",
            config_path=config_file,
            tool_names=["calendar"],
        )

        env_overrides = {"CALDAV_USERS": ",".join(accounts)}
        if preseed_svc_names:
            _run_profiled_services_in_sandbox(
                sandbox,
                preseed_svc_names,
                profile="preseed",
                env_overrides=env_overrides,
                log_prefix=tag,
                quiet=not log,
            )
        if seed_svc_names:
            _run_profiled_services_in_sandbox(
                sandbox,
                seed_svc_names,
                profile="seed",
                env_overrides=env_overrides,
                log_prefix=tag,
                quiet=not log,
            )

        # Register accounts with the calendar tool server
        log_fn = None if log else (lambda _msg: None)
        ensure_task_calendar_accounts(task_data, profiles, endpoints, config, log=log_fn)

    def _provision_group_channels(
        self,
        tag: str,
        sandbox: Any,  # noqa: ANN401
        task_data: dict[str, Any],
        profiles: dict[str, dict[str, Any]],
        config: Any,  # noqa: ANN401
        config_path: str,
        *,
        log: bool = True,
    ) -> None:
        """Provision RocketChat NPC users and group channels in ephemeral sandbox."""
        group_channels = task_data.get("seed_group_channels") or []
        if not group_channels:
            return

        if "rocketchat" not in config.tools:
            return

        # Collect all NPC profile IDs from npcs + group channel members
        npc_ids: list[str] = []
        for npc in task_data.get("npcs", []):
            if isinstance(npc, dict):
                npc_id = str(npc.get("id") or "").strip()
                if npc_id:
                    npc_ids.append(npc_id)
        for ch in group_channels:
            if isinstance(ch, dict):
                for mid in ch.get("member_profile_ids", []):
                    mid_str = str(mid).strip()
                    if mid_str:
                        npc_ids.append(mid_str)
        npc_ids = list(dict.fromkeys(npc_ids))
        if not npc_ids:
            return

        # Build NPC credentials matching server-side format
        npc_configs: dict[str, dict[str, Any]] = {}
        for pid in npc_ids:
            profile = profiles.get(pid, {})
            first = str(profile.get("first_name") or "").strip()
            last = str(profile.get("last_name") or "").strip()
            display_name = (
                f"{first} {last}".strip() if (first or last) else pid.replace("_", " ").title()
            )
            email = str(profile.get("email") or f"{pid}@example.com").strip()
            npc_configs[pid] = {
                "username": pid,
                "password": "npc123",
                "name": display_name,
                "email": email,
            }

        channel_names = [
            ch.get("channel_name", "?") for ch in group_channels if isinstance(ch, dict)
        ]
        if log:
            click.echo(
                f"{tag} Provisioning group channels: {', '.join(channel_names)} "
                f"({len(npc_ids)} NPC user(s))"
            )

        get_profiled_service_names, _ = get_env_runtime_helpers()
        config_file = Path(config_path)

        seed_svc_names = get_profiled_service_names(
            config,
            profile="seed",
            config_path=config_file,
            tool_names=["rocketchat"],
        )
        if not seed_svc_names:
            return

        env_overrides = {
            "ROCKETCHAT_NPC_CONFIGS": json.dumps(npc_configs),
            "ROCKETCHAT_SEED_GROUP_CHANNELS": json.dumps(group_channels),
        }
        _run_profiled_services_in_sandbox(
            sandbox,
            seed_svc_names,
            profile="seed",
            env_overrides=env_overrides,
            log_prefix=tag,
            quiet=not log,
        )

    def _seed_rollout(
        self,
        tag: str,
        task_data: dict[str, Any],
        profiles: dict[str, dict[str, Any]],
        endpoints: dict[str, str],
        *,
        log: bool = True,
    ) -> None:
        """Seed emails and calendar events for a single rollout.

        Group channels are provisioned separately by ``_provision_group_channels``
        before this method runs.
        """
        emails = task_data.get("seed_emails", [])
        cal_events = task_data.get("seed_calendar_events", [])
        if not emails and not cal_events:
            return

        if log:
            click.echo(f"{tag} Seeding task data...")
        log_fn = None if log else (lambda _msg: None)
        ok, fail = seed_task_data(task_data, profiles, endpoints, log=log_fn)
        if log:
            if fail:
                click.echo(click.style(f"{tag} Seeding had {fail} failure(s).", fg="yellow"))
            else:
                click.echo(click.style(f"{tag} {ok} item(s) seeded.", fg="green"))

    def _run_verifiers(
        self,
        *,
        tag: str,
        task_data: dict[str, Any],
        artifacts: Any,  # noqa: ANN401
        tool_servers: dict[str, str],
        rollout_dir: Path,
        bundle_dir: Path | None,
        global_cfg: Any,  # noqa: ANN401
        backend_id: str | None,
        base_url_api: str,
        scenario_manager_api_key: str | None,
        env_dir: Path,
        log: bool = True,
    ) -> tuple[float | None, bool | None, dict[str, Any] | None]:
        """Run verifiers and return ``(reward, passed, reward_payload)``."""
        evaluators = task_data.get("verifiers") or task_data.get("evaluators")
        if not evaluators:
            return None, None, None

        build_verifier_artifacts, infer_scenario_from_evaluator, run_verifier = (
            get_verifier_runtime_helpers()
        )
        verifier_scenario = infer_scenario_from_evaluator(evaluators[0]) or backend_id or ""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(task_data, f, indent=2)
            task_file = Path(f.name)

        try:
            adapter = build_verifier_artifacts(artifacts, task_file, tool_servers)
            verifier_results: list[dict[str, Any]] = []
            if log:
                click.echo(f"{tag} Running verifiers...")
            with self._verifier_env_lock:
                original_env, applied_env = apply_verifier_env_overrides(global_cfg)
                try:
                    for _i, ev in enumerate(evaluators, 1):
                        if ev.get("func") != "python_module" or not ev.get("module"):
                            continue
                        mod_path = ev["module"]
                        v_result = run_verifier(
                            mod_path,
                            adapter,
                            verifier_scenario,
                            scenario_manager_base_url=base_url_api,
                            scenario_manager_api_key=scenario_manager_api_key,
                            local_verifier_path=(
                                get_local_verifier_file_path(bundle_dir, mod_path)
                                if bundle_dir is not None
                                else None
                            ),
                            verifier_cache_root=env_dir / "verifiers",
                        )
                        verifier_results.append(
                            {
                                "module": mod_path,
                                "success": v_result.success,
                                "message": v_result.message or "",
                                "output": v_result.output or "",
                            }
                        )
                finally:
                    restore_env(original_env, applied_env)

            all_passed = all(r["success"] for r in verifier_results)
            reward = 1.0 if all_passed else 0.0

            # Rubric judge (optional)
            rubric_result_dict = maybe_run_rubric_judge(
                task_data=task_data,
                bundle_dir=bundle_dir,
                messages=list(artifacts.messages),
                global_cfg=global_cfg,
                log=log,
            )

            # Write verifier output files
            verifier_dir = rollout_dir / "verifier"
            verifier_dir.mkdir(parents=True, exist_ok=True)
            (verifier_dir / "reward.txt").write_text(
                "1" if all_passed else "0",
                encoding="utf-8",
            )
            reward_payload: dict[str, Any] = {
                "reward": reward,
                "verifier_results": verifier_results,
            }
            if rubric_result_dict is not None:
                reward_payload["rubric_result"] = rubric_result_dict
            (verifier_dir / "reward.json").write_text(
                json.dumps(reward_payload, indent=2),
                encoding="utf-8",
            )

            return reward, all_passed, reward_payload

        finally:
            task_file.unlink(missing_ok=True)

    def _cleanup_all(self) -> None:
        """Tear down any sandboxes still running (called via atexit)."""
        with self._lock:
            remaining = dict(self._active_sandboxes)

        if not remaining:
            return

        click.echo(
            click.style(
                f"\nCleaning up {len(remaining)} sandbox(es)...",
                fg="yellow",
            )
        )
        for idx, (client, sandbox) in remaining.items():
            try:
                deleted = teardown_sandbox(client, sandbox, log_prefix=f"[cleanup rollout {idx}]")
            except Exception:
                deleted = False
            if not deleted:
                # Teardown failed — keep in state file for manual recovery.
                click.echo(
                    click.style(
                        f"  [cleanup rollout {idx}] Teardown failed; sandbox {sandbox.id} "
                        f"preserved in state file for manual cleanup.",
                        fg="yellow",
                    ),
                    err=True,
                )
                continue
            # Only remove from tracking after successful deletion.
            with self._lock:
                self._active_sandboxes.pop(idx, None)
            self._unpersist_sandbox(sandbox.id)

    @staticmethod
    def _write_summary(summary: ParallelRunSummary, run_dir: Path) -> None:
        """Write summary.json to the output directory."""
        # A rollout is "passed" only when it has no error AND verification
        # did not explicitly fail.  This matches the CLI exit-code logic
        # which treats ``verification_passed is False`` as a failure.
        passed = [
            r for r in summary.results if r.error is None and r.verification_passed is not False
        ]
        failed = [
            r for r in summary.results if r.error is not None or r.verification_passed is False
        ]
        # Include all rewards (even 0.0 from failed verifications) so the
        # average reflects actual agent performance, not just the successes.
        rewards = [r.reward for r in summary.results if r.reward is not None]

        payload = {
            "task_id": summary.task_id,
            "rollout_count": summary.rollout_count,
            "passed": len(passed),
            "failed": len(failed),
            "avg_reward": (sum(rewards) / len(rewards)) if rewards else None,
            "avg_steps": (sum(r.steps_taken for r in passed) / len(passed) if passed else None),
            "total_duration_seconds": round(summary.total_duration_seconds, 1),
            "results": [
                {
                    "rollout_idx": r.rollout_idx,
                    "sandbox_id": r.sandbox_id,
                    "reward": r.reward,
                    "verification_passed": r.verification_passed,
                    "steps": r.steps_taken,
                    "duration": round(r.duration_seconds, 1),
                    "error": r.error,
                }
                for r in summary.results
            ],
        }
        (run_dir / "summary.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
