"""Rollout execution for autoresearch.

This runner reuses the same task execution codepaths as `simlab tasks run`, while
allowing the autoresearch loop to:

- Inject a runtime scenario prompt without rewriting `env.yaml`.
- Write rollout artifacts under a caller-provided output root.

The task set, environment, agent model, and evaluation are held fixed by the
run config. Only `scenario_guidance_md` changes between iterations.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from simlab.cli.progress import ParallelRolloutProgress
from simlab.cli.progress import ParallelRolloutProgressLike
from simlab.cli.progress import StepProgressLike
from simlab.composer.engine import EnvConfig
from simlab.config import GlobalConfig
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_daytona_api_key
from simlab.config import resolve_env_dir
from simlab.config import resolve_scenario_manager_api_url
from simlab.env_artifacts import ensure_env_artifacts_current
from simlab.env_artifacts import load_env_config
from simlab.env_registry import build_registry
from simlab.runtime.daytona_runner import DAYTONA_IMPORT_ERROR
from simlab.runtime.env_lifecycle import ensure_env_started_local
from simlab.runtime.env_lifecycle import env_down_local
from simlab.runtime.env_lifecycle import env_has_local_services
from simlab.runtime.env_lifecycle import is_env_running_local
from simlab.runtime.env_lifecycle import run_env_seed_local
from simlab.runtime.env_lifecycle import validate_daytona_coding_assets
from simlab.runtime.parallel_daytona import ParallelDaytonaOrchestrator
from simlab.runtime.parallel_daytona import ParallelRunSummary
from simlab.runtime.rollout_runner import SingleRolloutOutcome
from simlab.runtime.rollout_runner import get_env_runtime_helpers
from simlab.runtime.rollout_runner import load_local_task
from simlab.runtime.rollout_runner import resolve_agent_runtime_settings
from simlab.runtime.rollout_runner import resolve_rollout_format
from simlab.runtime.rollout_runner import run_single_rollout


@dataclass(frozen=True)
class RolloutRunInfo:
    """Summary of one task run written under an output root."""

    task_id: str
    output_path: Path
    runtime: str
    rollout_count: int
    duration_seconds: float


@dataclass(frozen=True)
class TaskSetProgressAdapter:
    """Map a single-rollout progress stream onto a fixed row in a task-set table."""

    progress: ParallelRolloutProgress
    task_idx: int

    def update(
        self,
        rollout_idx: int,
        *,
        status: str | None = None,
        steps_taken: int | None = None,
        result: str | None = None,
    ) -> None:
        """Update the task-set row with the latest rollout progress."""
        _ = rollout_idx
        self.progress.update(
            self.task_idx,
            status=status,
            steps_taken=steps_taken,
            result=result,
        )


@dataclass(frozen=True)
class TaskSetRowStepContext:
    """Context object passed to single-rollout steps so we can update a task-set row."""

    progress: ParallelRolloutProgress
    task_idx: int

    def detail(self, message: str) -> None:
        """Accept step detail updates for compatibility with StepContext."""
        _ = message

    def flush_buffer(self) -> None:
        """Flush any buffered output for compatibility with StepContext."""
        return

    def update(self, message: str) -> None:
        """Accept step status updates for compatibility with StepContext."""
        _ = message

    def on_step(self, steps_taken: int) -> None:
        """Mark the task row as running with the updated step count."""
        self.progress.update(
            self.task_idx,
            status="Running",
            steps_taken=steps_taken,
        )


@dataclass(frozen=True)
class TaskSetRowStepProgress:
    """StepProgress-like adapter that keeps a fixed task-set table row updated."""

    progress: ParallelRolloutProgress
    task_idx: int

    @contextmanager
    def step(
        self,
        label: str,
        *,
        success_label: str | None = None,
    ) -> Generator[TaskSetRowStepContext, None, None]:
        """Translate step lifecycle events into task-set row status updates."""
        _ = success_label
        normalized = label.strip().lower()
        status = None
        steps_taken = None
        if normalized in {"seeded", "services started"}:
            status = "Starting"
        elif normalized == "agent running":
            status = "Running"
            steps_taken = 0
        elif normalized == "verifying":
            status = "Verifying"

        if status is not None:
            self.progress.update(
                self.task_idx,
                status=status,
                steps_taken=steps_taken,
            )

        ctx = TaskSetRowStepContext(progress=self.progress, task_idx=self.task_idx)
        try:
            yield ctx
        except (Exception, SystemExit):
            self.progress.update(
                self.task_idx,
                status="Done",
                result="ERR",
            )
            raise


def run_rollouts(
    *,
    env_name: str,
    tasks_dir: Path,
    task_ids: list[str],
    runtime: str,
    scenario_prompt: str,
    model: str | None,
    provider: str | None,
    api_key: str | None,
    base_url: str | None,
    rollout_count: int,
    max_parallel: int,
    max_steps: int,
    agent_timeout_seconds: float,
    no_seed: bool,
    global_cfg: GlobalConfig,
    ctx: click.Context,
    output_root: Path,
) -> list[RolloutRunInfo]:
    """Run the fixed task set and write rollouts under `output_root`."""
    if runtime == "daytona":
        require_daytona_runtime(global_cfg=global_cfg)

    env_dir = resolve_env_dir(env_name, ctx=ctx)
    ensure_env_artifacts_current(env_dir, action_label="autoresearch run")

    config_path = str(env_dir / "env.yaml")
    env_cfg = load_env_config(env_dir)
    env_cfg = _override_scenario_prompt(env_cfg, scenario_prompt)

    base_url_api = resolve_scenario_manager_api_url(
        config_path=Path(config_path),
        config=env_cfg,
        base_url=resolve_scenario_manager_api_url(config=global_cfg),
    )
    scenario_manager_api_key = resolve_collinear_api_key(config=global_cfg)

    rollout_format = resolve_rollout_format(
        requested=None,
        config=env_cfg,
        global_cfg=global_cfg,
        harbor=False,
    )

    resolved_model, resolved_provider, resolved_api_key, resolved_base_url = (
        resolve_agent_runtime_settings(global_cfg, model, provider, api_key, base_url)
    )

    infos: list[RolloutRunInfo] = []
    if runtime != "daytona":
        managed_env = env_has_local_services(env_dir) and not is_env_running_local(env_dir)

        progress = (
            ParallelRolloutProgress(
                rollout_count=len(task_ids),
                task_name="",
                max_steps=max_steps,
                row_labels=task_ids,
                index_label="#",
                transient=False,
            )
            if sys.stdout.isatty()
            else None
        )

        def run_local_one(task_idx: int, task_id: str) -> RolloutRunInfo:
            task_data, profiles, _task_file = load_local_task(tasks_dir, task_id)
            start = time.monotonic()
            task_output = _run_local_task(
                env_dir=env_dir,
                config_path=config_path,
                config=env_cfg,
                global_cfg=global_cfg,
                task_id=task_id,
                task_data=task_data,
                profiles=profiles,
                bundle_dir=tasks_dir,
                model=resolved_model,
                provider=resolved_provider,
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                max_steps=max_steps,
                agent_timeout_seconds=agent_timeout_seconds,
                no_seed=no_seed,
                base_url_api=base_url_api,
                scenario_manager_api_key=scenario_manager_api_key,
                rollout_format=rollout_format,
                output_root=output_root,
                progress=(
                    TaskSetRowStepProgress(progress, task_idx) if progress is not None else None
                ),
            )
            if progress is not None:
                result = format_outcome_result(task_output)
                progress.update(
                    task_idx,
                    status="Done",
                    steps_taken=task_output.steps_taken,
                    result=result,
                )
            return RolloutRunInfo(
                task_id=task_id,
                output_path=task_output.output_dir,
                runtime=runtime,
                rollout_count=rollout_count,
                duration_seconds=time.monotonic() - start,
            )

        try:
            if managed_env:
                ensure_env_started_local(env_dir, env_cfg, env_dir / "env.yaml")
                if not no_seed:
                    run_env_seed_local(env_dir, env_cfg, env_dir / "env.yaml")

            if progress is None:
                for idx, task_id in enumerate(task_ids):
                    infos.append(run_local_one(idx, task_id))
                return infos

            with progress:
                for idx, task_id in enumerate(task_ids):
                    progress.update(idx, status="Starting")
                    infos.append(run_local_one(idx, task_id))
            return infos
        finally:
            if managed_env:
                with suppress(SystemExit):
                    env_down_local(env_dir)

    progress = (
        ParallelRolloutProgress(
            rollout_count=len(task_ids),
            task_name="",
            max_steps=max_steps,
            row_labels=task_ids,
            index_label="#",
            transient=False,
        )
        if sys.stdout.isatty()
        else None
    )

    def run_daytona_one(task_idx: int, task_id: str) -> RolloutRunInfo:
        task_data, profiles, _task_file = load_local_task(tasks_dir, task_id)
        start = time.monotonic()
        task_output_dir = _run_daytona_task(
            env_dir=env_dir,
            config_path=config_path,
            config=env_cfg,
            global_cfg=global_cfg,
            task_id=task_id,
            task_data=task_data,
            profiles=profiles,
            bundle_dir=tasks_dir,
            rollout_count=rollout_count,
            max_parallel=max_parallel,
            model=resolved_model,
            provider=resolved_provider,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            max_steps=max_steps,
            agent_timeout_seconds=agent_timeout_seconds,
            no_seed=no_seed,
            base_url_api=base_url_api,
            scenario_manager_api_key=scenario_manager_api_key,
            rollout_format=rollout_format,
            output_root=output_root,
            progress=TaskSetProgressAdapter(progress, task_idx) if progress is not None else None,
        )
        return RolloutRunInfo(
            task_id=task_id,
            output_path=task_output_dir,
            runtime=runtime,
            rollout_count=rollout_count,
            duration_seconds=time.monotonic() - start,
        )

    if progress is None:
        for idx, task_id in enumerate(task_ids):
            infos.append(run_daytona_one(idx, task_id))
        return infos

    with progress:
        for idx, task_id in enumerate(task_ids):
            infos.append(run_daytona_one(idx, task_id))
    return infos


def require_daytona_runtime(*, global_cfg: GlobalConfig) -> None:
    """Fail fast when runtime=daytona but the SDK or API key is missing."""
    if DAYTONA_IMPORT_ERROR is not None:
        raise click.ClickException(
            "Runtime 'daytona' requires the optional 'daytona' package.\n"
            "Source checkout: uv sync --extra daytona\n"
            'Installed CLI: uv tool install --python 3.13 "simulationlab[daytona]"'
        )

    api_key = resolve_daytona_api_key(config=global_cfg)
    if not api_key:
        raise click.ClickException(
            "Runtime 'daytona' requires a Daytona API key. "
            "Set SIMLAB_DAYTONA_API_KEY or DAYTONA_API_KEY, or pass --daytona-api-key."
        )


def _run_daytona_task(
    *,
    env_dir: Path,
    config_path: str,
    config: EnvConfig,
    global_cfg: GlobalConfig,
    task_id: str,
    task_data: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    bundle_dir: Path,
    rollout_count: int,
    max_parallel: int,
    model: str,
    provider: str,
    api_key: str | None,
    base_url: str | None,
    max_steps: int,
    agent_timeout_seconds: float,
    no_seed: bool,
    base_url_api: str,
    scenario_manager_api_key: str | None,
    rollout_format: str,
    output_root: Path,
    progress: ParallelRolloutProgressLike | None = None,
) -> Path:
    validate_daytona_coding_assets(config, env_dir)

    parallel_state = env_dir / "parallel-sandboxes.json"
    if parallel_state.exists():
        ParallelDaytonaOrchestrator.cleanup_orphaned_sandboxes(
            parallel_state,
            daytona_api_key=global_cfg.daytona_api_key,
        )

    registry = build_registry(env_dir=env_dir)
    tool_ports: dict[str, int] = {}
    extra_tool_urls: dict[str, str] = {}
    for tool_name in config.tools:
        tool = registry.get_tool(tool_name)
        if tool is None:
            continue
        if tool.tool_server_port is not None:
            tool_ports[tool_name] = tool.tool_server_port
        elif tool.tool_server_url:
            extra_tool_urls[tool_name] = tool.tool_server_url

    get_profiled_service_names, _ = get_env_runtime_helpers()

    config_path_path = Path(config_path)
    preseed_svc_names = get_profiled_service_names(
        config,
        profile="preseed",
        config_path=config_path_path,
    )
    seed_svc_names = get_profiled_service_names(
        config,
        profile="seed",
        config_path=config_path_path,
    )

    orchestrator = ParallelDaytonaOrchestrator(
        rollout_count=rollout_count,
        max_parallel=max_parallel,
        daytona_api_key=global_cfg.daytona_api_key,
    )

    meta_obj = task_data.get("meta")
    meta = meta_obj if isinstance(meta_obj, dict) else {}
    parallel_task_id = meta.get("task_id", task_id)
    created_progress = False
    parallel_progress: ParallelRolloutProgressLike | None = progress
    if parallel_progress is None and sys.stdout.isatty():
        parallel_progress = ParallelRolloutProgress(
            rollout_count=rollout_count,
            task_name=str(parallel_task_id),
            max_steps=max_steps,
        )
        created_progress = True

    def execute_parallel() -> ParallelRunSummary:
        return orchestrator.execute(
            task_id=str(parallel_task_id),
            task_data=task_data,
            profiles=profiles,
            compose_dir=env_dir,
            tool_ports=tool_ports,
            extra_tool_urls=extra_tool_urls,
            preseed_svc_names=preseed_svc_names,
            seed_svc_names=seed_svc_names,
            config=config,
            config_path=config_path,
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            max_steps=max_steps,
            agent_import_path=None,
            agent_timeout_seconds=agent_timeout_seconds,
            no_seed=no_seed,
            bundle_dir=bundle_dir,
            global_cfg=global_cfg,
            backend_id=None,
            base_url_api=base_url_api,
            scenario_manager_api_key=scenario_manager_api_key,
            rollout_format=rollout_format,
            output_root=output_root,
            progress=parallel_progress,
        )

    if created_progress and isinstance(parallel_progress, ParallelRolloutProgress):
        with parallel_progress:
            summary = execute_parallel()
    else:
        summary = execute_parallel()
    return summary.output_dir


def _run_local_task(
    *,
    env_dir: Path,
    config_path: str,
    config: EnvConfig,
    global_cfg: GlobalConfig,
    task_id: str,
    task_data: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    bundle_dir: Path,
    model: str,
    provider: str,
    api_key: str | None,
    base_url: str | None,
    max_steps: int,
    agent_timeout_seconds: float,
    no_seed: bool,
    base_url_api: str,
    scenario_manager_api_key: str | None,
    rollout_format: str,
    output_root: Path,
    progress: StepProgressLike | None = None,
) -> SingleRolloutOutcome:
    has_local_services = env_has_local_services(env_dir)
    managed_env = False
    try:
        if has_local_services:
            env_running = is_env_running_local(env_dir)
            if not env_running:
                managed_env = True
                ensure_env_started_local(env_dir, config, env_dir / "env.yaml")
                if not no_seed:
                    run_env_seed_local(env_dir, config, env_dir / "env.yaml")

        meta_obj = task_data.get("meta")
        meta = meta_obj if isinstance(meta_obj, dict) else {}
        display = meta.get("display_name", meta.get("task_id", task_id))
        outcome: SingleRolloutOutcome = run_single_rollout(
            env_dir=env_dir,
            config=config,
            config_path=config_path,
            global_cfg=global_cfg,
            task_data=task_data,
            task_id=task_id,
            profiles=profiles,
            meta=meta,
            display=str(display),
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            max_steps=max_steps,
            agent_import_path=None,
            agent_timeout_seconds=agent_timeout_seconds,
            no_seed=no_seed,
            verbose=False,
            daytona=False,
            tasks_dir=str(bundle_dir),
            bundle_dir=bundle_dir,
            backend_id=None,
            base_url_api=base_url_api,
            scenario_manager_api_key=scenario_manager_api_key,
            managed_env=managed_env,
            keep_alive=False,
            skip_env_setup=True,
            rollout_format=rollout_format,
            progress=progress,
            output_root=output_root,
        )
        return outcome
    finally:
        if managed_env:
            with suppress(SystemExit):
                env_down_local(env_dir)


def format_outcome_result(outcome: SingleRolloutOutcome) -> str:
    """Format a rollout outcome into a compact status string for the task table."""
    if outcome.run_error:
        return "ERR"
    if outcome.verification_passed is False:
        return "FAIL"
    if outcome.exit_code != 0:
        return "FAIL"
    return "PASS"


def _override_scenario_prompt(config: EnvConfig, scenario_prompt: str) -> EnvConfig:
    updated_prompt = scenario_prompt.strip()
    return config.model_copy(update={"scenario_guidance_md": updated_prompt})
