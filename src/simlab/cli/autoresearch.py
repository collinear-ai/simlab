"""CLI entry points for `simlab autoresearch`."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import click

from simlab.autoresearch.config import AutoresearchRunConfig
from simlab.autoresearch.config import load_run_config
from simlab.autoresearch.config import render_run_toml_template
from simlab.autoresearch.manager import run_autoresearch
from simlab.autoresearch.objectives import objective_direction
from simlab.autoresearch.report import write_end_of_run_reports
from simlab.cli.questionary import Choice
from simlab.cli.questionary import checkbox
from simlab.cli.questionary import confirm
from simlab.cli.questionary import select
from simlab.cli.questionary import text
from simlab.config import get_environments_dir
from simlab.config import resolve_daytona_api_key
from simlab.runtime.daytona_runner import DAYTONA_IMPORT_ERROR
from simlab.runtime.rollout_runner import normalize_tasks_bundle_dir
from simlab.telemetry import TelemetryCaptureConfig
from simlab.telemetry import emit_cli_event
from simlab.telemetry import normalize_config_path
from simlab.telemetry import resolve_scenario_manager_capture_config
from simlab.telemetry import with_command_telemetry

TASK_SELECT_ALL_VALUE = "__simlab_select_all__"
WIZARD_RULE_WIDTH = 72


@click.group()
def autoresearch() -> None:
    """Iterate on the runtime scenario prompt and measure whether it improves eval."""


def autoresearch_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for autoresearch commands when a Collinear API key is configured."""
    _ = args
    return resolve_scenario_manager_capture_config(
        ctx,
        config_path=normalize_config_path(kwargs.get("config_path")),
    )


@autoresearch.command("init")
@click.option("--env", "env_name", default=None, help="Saved setup name.")
@click.option(
    "--environments-dir",
    "environments_dir",
    default=None,
    type=click.Path(path_type=Path, exists=False, file_okay=False),
    help="Root directory for environments (overrides SIMLAB_ENVIRONMENTS_DIR).",
)
@click.option(
    "--tasks-dir",
    "tasks_dir",
    default=None,
    type=click.Path(path_type=Path, exists=False, file_okay=False),
    help="Path to a frozen local tasks bundle directory.",
)
@click.option("--task", "task_ids", multiple=True, help="Task ID to include (repeatable).")
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path, exists=False, dir_okay=False),
    default=Path("run.toml"),
    show_default=True,
    help="Where to write the config TOML.",
)
@with_command_telemetry("autoresearch init", resolver=autoresearch_capture_config)
@click.pass_context
def init(
    ctx: click.Context,
    env_name: str | None,
    environments_dir: Path | None,
    tasks_dir: Path | None,
    task_ids: tuple[str, ...],
    out_path: Path,
) -> None:
    """Write a starter run.toml so you do not have to memorize flags."""
    apply_environments_dir_override(ctx=ctx, environments_dir=environments_dir)
    env = _resolve_env_name(ctx=ctx, env_name=env_name)
    if not env:
        raise click.ClickException("Provide --env or run in a TTY to be prompted.")

    env_root = get_environments_dir(ctx=ctx)
    setup_dir = (env_root / env).resolve() if env and env_root.is_dir() else None
    tasks_dir_str = _resolve_tasks_dir(
        tasks_dir=tasks_dir,
        require_exists=False,
        setup_dir=setup_dir,
    )
    if not tasks_dir_str:
        raise click.ClickException("Provide --tasks-dir or run in a TTY to be prompted.")

    resolved_tasks_dir = resolve_relative_path(path=Path(tasks_dir_str), base_dir=None)
    if resolved_tasks_dir is not None:
        tasks_dir_str = str(normalize_tasks_bundle_dir(resolved_tasks_dir))

    task_list = _resolve_task_ids(task_ids=task_ids, tasks_dir=Path(tasks_dir_str))
    if not task_list:
        raise click.ClickException("Provide --task at least once or run in a TTY to be prompted.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_run_toml_template(
            env=env,
            tasks_dir=tasks_dir_str,
            task_ids=task_list,
            environments_dir=str(env_root.resolve()),
        ),
        encoding="utf-8",
    )
    click.echo(f"Wrote {out_path}")
    click.echo("Next: pick models by editing the [agent], [proposer], and [verifier] sections.")
    click.echo("Next: simlab autoresearch run --config " + str(out_path))


@autoresearch.command("run")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Path to run.toml (recommended).",
)
@click.option("--env", "env_name", default=None, help="Quick start. Saved setup name.")
@click.option(
    "--environments-dir",
    "environments_dir",
    default=None,
    type=click.Path(path_type=Path, exists=False, file_okay=False),
    help="Root directory for environments (overrides SIMLAB_ENVIRONMENTS_DIR).",
)
@click.option(
    "--tasks-dir",
    "tasks_dir",
    default=None,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Quick start. Local tasks bundle directory.",
)
@click.option("--task", "task_ids", multiple=True, help="Quick start. Task ID (repeatable).")
@with_command_telemetry("autoresearch run", resolver=autoresearch_capture_config)
@click.pass_context
def run(
    ctx: click.Context,
    config_path: Path | None,
    env_name: str | None,
    environments_dir: Path | None,
    tasks_dir: Path | None,
    task_ids: tuple[str, ...],
) -> None:
    """Run the sequential prompt optimization loop."""
    apply_environments_dir_override(ctx=ctx, environments_dir=environments_dir)
    if config_path is not None:
        try:
            cfg = load_run_config(config_path)
        except Exception as exc:
            raise click.ClickException(f"Invalid config: {exc}") from exc
        if (
            not cfg.agent.model.strip()
            or not cfg.proposer.model.strip()
            or not cfg.verifier.model.strip()
        ):
            raise click.ClickException(
                "Run config is missing model names. "
                "Fill in [agent].model, [proposer].model, and [verifier].model, then rerun."
            )

        if environments_dir is None and cfg.run.environments_dir:
            apply_environments_dir_override(
                ctx=ctx,
                environments_dir=Path(cfg.run.environments_dir),
            )
    else:
        interactive = sys_tty()
        if not interactive:
            has_env = bool((env_name or "").strip())
            has_tasks_dir = tasks_dir is not None
            has_tasks = any((t or "").strip() for t in task_ids)
            if not (has_env and has_tasks_dir and has_tasks):
                raise click.ClickException(
                    "Run without --config requires a TTY unless you pass --env, --tasks-dir, "
                    "and at least one --task. Run in a terminal, or create a config with "
                    "`simlab autoresearch init` and rerun with "
                    "`simlab autoresearch run --config run.toml`."
                )
        if interactive:
            _print_wizard_intro()
            emit_cli_event(
                "autoresearch_wizard_started",
                {
                    "config_provided": False,
                    "quick_start_env": bool((env_name or "").strip()),
                    "quick_start_tasks_dir": tasks_dir is not None,
                    "quick_start_task_count": len([t for t in task_ids if (t or "").strip()]),
                },
            )
        wizard_attempt = 1
        while True:
            if interactive and not (env_name or "").strip():
                _print_environment_step()
                emit_cli_event(
                    "autoresearch_wizard_step_shown",
                    {"step": "setup", "step_index": 1, "wizard_attempt": wizard_attempt},
                )
            env = _resolve_env_name(ctx=ctx, env_name=env_name)
            emit_cli_event(
                "autoresearch_wizard_step_completed",
                {
                    "step": "setup",
                    "step_index": 1,
                    "wizard_attempt": wizard_attempt,
                    "prompted": bool(interactive and not (env_name or "").strip()),
                },
            )
            setup_dir = (
                (get_environments_dir(ctx=ctx) / env).resolve()
                if env and get_environments_dir(ctx=ctx).is_dir()
                else None
            )

            if interactive and tasks_dir is None:
                _print_tasks_dir_step()
                emit_cli_event(
                    "autoresearch_wizard_step_shown",
                    {"step": "task_folder", "step_index": 2, "wizard_attempt": wizard_attempt},
                )
            tasks_dir_str = _resolve_tasks_dir(
                tasks_dir=tasks_dir,
                require_exists=True,
                setup_dir=setup_dir,
            )
            emit_cli_event(
                "autoresearch_wizard_step_completed",
                {
                    "step": "task_folder",
                    "step_index": 2,
                    "wizard_attempt": wizard_attempt,
                    "prompted": bool(interactive and tasks_dir is None),
                },
            )
            if tasks_dir_str:
                resolved_tasks_dir = resolve_relative_path(path=Path(tasks_dir_str), base_dir=None)
                if resolved_tasks_dir is not None:
                    tasks_dir_str = str(normalize_tasks_bundle_dir(resolved_tasks_dir))
            if interactive and tasks_dir_str:
                discovered_tasks = _discover_tasks(Path(tasks_dir_str))
                _print_task_selection_step(
                    tasks_dir=Path(tasks_dir_str),
                    discovered=discovered_tasks,
                )
            if interactive and not task_ids:
                emit_cli_event(
                    "autoresearch_wizard_step_shown",
                    {"step": "tasks", "step_index": 3, "wizard_attempt": wizard_attempt},
                )
            task_list = _resolve_task_ids(
                task_ids=task_ids,
                tasks_dir=Path(tasks_dir_str) if tasks_dir_str else Path(),
            )
            emit_cli_event(
                "autoresearch_wizard_step_completed",
                {
                    "step": "tasks",
                    "step_index": 3,
                    "wizard_attempt": wizard_attempt,
                    "prompted": bool(interactive and not task_ids),
                    "task_count": len(task_list),
                },
            )

            if not env or not tasks_dir_str or not task_list:
                raise click.ClickException(
                    "Run needs a saved setup, a task folder, and at least one task. "
                    "Run: simlab autoresearch init"
                )

            run_section: dict[str, object] = {
                "env": env,
                "tasks_dir": tasks_dir_str,
                "task_ids": task_list,
                "runtime": "local",
                "rollout_count": 1,
                "max_parallel": 1,
            }
            agent_section: dict[str, object] = {
                "model": "gpt-4o-mini",
                "provider": "openai",
                "api_key_env": "OPENAI_API_KEY",
                "base_url_env": "OPENAI_API_BASE",
            }
            proposer_section: dict[str, object] = {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "OPENAI_API_KEY",
                "base_url_env": "OPENAI_API_BASE",
            }
            verifier_section: dict[str, object] = {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "OPENAI_API_KEY",
                "base_url_env": "OPENAI_API_BASE",
            }
            objective_section: dict[str, object] = {"type": "pass_rate", "target": 0.8}
            budget_section: dict[str, object] = {}

            if interactive:
                daytona_available, daytona_note = _daytona_runtime_available(ctx=ctx)
                _print_runtime_step(daytona_note=daytona_note)
                if daytona_available:
                    emit_cli_event(
                        "autoresearch_wizard_step_shown",
                        {"step": "runtime", "step_index": 4, "wizard_attempt": wizard_attempt},
                    )
                runtime = "local"
                if daytona_available:
                    runtime = select(
                        "Where should SimLab run each trial?",
                        choices=[
                            Choice(title="local - Docker on this machine", value="local"),
                            Choice(title="daytona - remote sandboxes", value="daytona"),
                        ],
                        default="local",
                        instruction="Use Up/Down to compare options. Enter picks one.",
                    ).ask()
                run_section["runtime"] = runtime
                run_section["rollout_count"] = 1
                run_section["max_parallel"] = 1
                default_iterations = 1 if runtime == "local" else 20
                emit_cli_event(
                    "autoresearch_wizard_step_completed",
                    {
                        "step": "runtime",
                        "step_index": 4,
                        "wizard_attempt": wizard_attempt,
                        "prompted": bool(daytona_available),
                        "runtime": runtime,
                    },
                )

                _print_iteration_budget_step(
                    runtime=runtime,
                    default_iterations=default_iterations,
                )
                emit_cli_event(
                    "autoresearch_wizard_step_shown",
                    {"step": "iterations", "step_index": 5, "wizard_attempt": wizard_attempt},
                )
                max_iterations = int(
                    text(
                        "How many prompt edits should SimLab try?",
                        default=str(default_iterations),
                        validate=lambda v: (
                            None
                            if v.strip().isdigit() and int(v) >= 0
                            else "Enter an integer >= 0."
                        ),
                    ).ask()
                )
                budget_section["max_iterations"] = max_iterations
                emit_cli_event(
                    "autoresearch_wizard_step_completed",
                    {
                        "step": "iterations",
                        "step_index": 5,
                        "wizard_attempt": wizard_attempt,
                        "prompted": True,
                        "iteration_count": max_iterations,
                    },
                )

                _print_model_step()
                emit_cli_event(
                    "autoresearch_wizard_step_shown",
                    {"step": "models", "step_index": 6, "wizard_attempt": wizard_attempt},
                )
                _configure_model_sections(
                    agent_section=agent_section,
                    proposer_section=proposer_section,
                    verifier_section=verifier_section,
                )
                emit_cli_event(
                    "autoresearch_wizard_step_completed",
                    {
                        "step": "models",
                        "step_index": 6,
                        "wizard_attempt": wizard_attempt,
                        "prompted": True,
                        "agent_provider": str(agent_section.get("provider") or ""),
                        "proposer_provider": str(proposer_section.get("provider") or ""),
                        "verifier_provider": str(verifier_section.get("provider") or ""),
                    },
                )

                default_target = "0.8"
                default_max_minutes = "90"
                default_no_improvement = "2"
                _print_scoring_step()
                _print_default_success_goal(
                    target=default_target,
                    max_minutes=default_max_minutes,
                    no_improvement_window=default_no_improvement,
                )
                emit_cli_event(
                    "autoresearch_wizard_step_shown",
                    {"step": "success_goal", "step_index": 7, "wizard_attempt": wizard_attempt},
                )
                show_more = confirm(
                    "Would you like to adjust the success goal or stop rules?",
                    default=False,
                ).ask()
                changed_success_goal = bool(show_more)
                if show_more:
                    objective_section["type"] = select(
                        "Which goal should SimLab optimize for?",
                        choices=[
                            Choice(
                                title="Task success rate [0..1], higher is better - pass_rate",
                                value="pass_rate",
                            ),
                            Choice(
                                title="Average reward, higher is better - avg_reward",
                                value="avg_reward",
                            ),
                            Choice(
                                title="Check pass rate [0..1], higher is better - check_pass_rate",
                                value="check_pass_rate",
                            ),
                            Choice(
                                title=(
                                    "Rubric judge score [0..1], higher is better "
                                    "- reward_model_score_mean"
                                ),
                                value="reward_model_score_mean",
                            ),
                            Choice(
                                title="Tool error rate [0..1], lower is better - tool_error_rate",
                                value="tool_error_rate",
                            ),
                        ],
                        default=str(objective_section["type"]),
                        instruction=(
                            "Choose the score SimLab should optimize across the whole run."
                        ),
                    ).ask()

                    default_target = _default_target_for_objective(str(objective_section["type"]))
                    stop_mode = select(
                        "Should SimLab stop early once it reaches a target score?",
                        choices=[
                            Choice(
                                title="Yes, stop once the target goal is reached",
                                value="target",
                            ),
                            Choice(
                                title="No, continue optimizing until a stop condition is reached",
                                value="none",
                            ),
                        ],
                        default="target",
                        instruction=("You will pick the target value in the next step."),
                    ).ask()
                    if stop_mode == "target":
                        direction = objective_direction(str(objective_section["type"]))
                        target_label = str(objective_section["type"])
                        op = ">=" if direction == "max" else "<="
                        target_prompt = f"Stop early once {target_label} is {op} this score"
                        target_raw = text(
                            target_prompt,
                            default=str(default_target),
                            validate=lambda v: (
                                None if v.strip() and is_float(v) else "Enter a number."
                            ),
                        ).ask()
                        try:
                            objective_section["target"] = float(target_raw)
                        except ValueError as exc:  # pragma: no cover
                            raise click.ClickException("Enter a number.") from exc
                    else:
                        objective_section["target"] = None

                    budget_section["max_minutes"] = int(
                        text(
                            "How many minutes should SimLab spend at most? (-1 for no limit)",
                            default=str(default_max_minutes),
                            validate=lambda v: (
                                None
                                if v.strip() == "-1" or (v.strip().isdigit() and int(v) >= 1)
                                else "Enter -1 or an integer >= 1."
                            ),
                        ).ask()
                    )
                    budget_section["no_improvement_window"] = int(
                        text(
                            "Stop after N rejected edits in a row (-1 for no limit)",
                            default=str(default_no_improvement),
                            validate=lambda v: (
                                None
                                if v.strip() == "-1" or (v.strip().isdigit() and int(v) >= 0)
                                else "Enter -1 or an integer >= 0."
                            ),
                        ).ask()
                    )

                summary = _render_run_preview(
                    run_section=run_section,
                    agent_section=agent_section,
                    proposer_section=proposer_section,
                    verifier_section=verifier_section,
                    objective_section=objective_section,
                    budget_section=budget_section,
                )
                click.echo()
                click.echo(click.style("Review this run plan", bold=True))
                click.echo(summary)
                emit_cli_event(
                    "autoresearch_wizard_step_completed",
                    {
                        "step": "success_goal",
                        "step_index": 7,
                        "wizard_attempt": wizard_attempt,
                        "prompted": True,
                        "objective_type": str(objective_section.get("type") or ""),
                        "objective_has_target": objective_section.get("target") is not None,
                        "changed_success_goal": changed_success_goal,
                    },
                )

                emit_cli_event(
                    "autoresearch_wizard_step_shown",
                    {"step": "review", "step_index": 8, "wizard_attempt": wizard_attempt},
                )
                proceed = confirm("Proceed with this run?", default=True).ask()
                emit_cli_event(
                    "autoresearch_wizard_step_completed",
                    {
                        "step": "review",
                        "step_index": 8,
                        "wizard_attempt": wizard_attempt,
                        "prompted": True,
                        "proceed": bool(proceed),
                    },
                )
                if not proceed:
                    click.echo("Restarting setup.")
                    wizard_attempt += 1
                    continue
            else:
                # In non-interactive quick start mode we keep this run local and cheap.
                run_section["runtime"] = "local"
                run_section["rollout_count"] = 1
                run_section["max_parallel"] = 1
                budget_section["max_iterations"] = 1

            cfg = AutoresearchRunConfig.model_validate(
                {
                    "run": run_section,
                    "agent": agent_section,
                    "proposer": proposer_section,
                    "verifier": verifier_section,
                    "objective": objective_section,
                    "budget": budget_section,
                }
            )
            if interactive:
                emit_cli_event(
                    "autoresearch_wizard_completed",
                    {
                        "runtime": str(run_section.get("runtime") or ""),
                        "task_count": len(task_list),
                        "iteration_count": cfg.budget.max_iterations,
                        "objective_type": str(objective_section.get("type") or ""),
                        "objective_has_target": objective_section.get("target") is not None,
                        "wizard_attempts": wizard_attempt,
                    },
                )
            break

    ensure_autoresearch_environment_resolves(
        ctx=ctx,
        env_name=cfg.run.env,
        tasks_dir=Path(cfg.run.tasks_dir),
        base_dir=config_path.parent if config_path is not None else None,
    )
    environments_root = str(get_environments_dir(ctx=ctx).resolve())
    if cfg.run.environments_dir != environments_root:
        cfg = cfg.model_copy(
            update={
                "run": cfg.run.model_copy(
                    update={
                        "environments_dir": environments_root,
                    }
                )
            }
        )
    run_autoresearch(cfg=cfg, ctx=ctx)


@autoresearch.command("report")
@click.argument(
    "run_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
@with_command_telemetry("autoresearch report", resolver=autoresearch_capture_config)
def report(run_dir: Path) -> None:
    """Rebuild report.md and compare.md from an existing run directory."""
    cfg_path = run_dir / "run.toml"
    if not cfg_path.is_file():
        raise click.ClickException(f"Missing run.toml at {cfg_path}")

    cfg = load_run_config(cfg_path)
    baseline_prompt = (run_dir / "baseline" / "scenario_prompt.md").read_text(encoding="utf-8")
    best_prompt = (run_dir / "best" / "scenario_prompt.md").read_text(encoding="utf-8")
    baseline_result = json.loads((run_dir / "baseline" / "result.json").read_text(encoding="utf-8"))
    best_result = json.loads((run_dir / "best" / "result.json").read_text(encoding="utf-8"))
    best_iteration = int(best_result.get("iteration") or 0)

    history = _load_history(run_dir)
    baseline_output = run_dir / "baseline" / "output"
    best_output = (
        baseline_output
        if best_iteration == 0
        else (run_dir / "iterations" / f"iter_{best_iteration:03d}" / "output")
    )

    write_end_of_run_reports(
        run_dir=run_dir,
        cfg=cfg,
        baseline_output=baseline_output,
        best_output=best_output,
        history=history,
        best_iteration=best_iteration,
        best_result=best_result,
        baseline_result=baseline_result,
        best_prompt=best_prompt,
        baseline_prompt=baseline_prompt,
    )
    click.echo(f"Wrote {run_dir / 'report.md'} and {run_dir / 'compare.md'}")


def _load_history(run_dir: Path) -> list[dict[str, object]]:
    history: list[dict[str, object]] = []
    baseline = json.loads((run_dir / "baseline" / "result.json").read_text(encoding="utf-8"))
    history.append(
        {
            "iteration": 0,
            "surface": "baseline",
            "accepted": True,
            "objective_value": baseline.get("objective_value"),
            "change_type": None,
            "rationale": "baseline",
        }
    )
    iters_dir = run_dir / "iterations"
    if not iters_dir.is_dir():
        return history
    for child in sorted(iters_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        result_path = child / "result.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        history.append(
            {
                "iteration": payload.get("iteration"),
                "surface": "scenario_prompt",
                "accepted": payload.get("accepted"),
                "objective_value": payload.get("objective_value"),
                "change_type": payload.get("change_type"),
                "rationale": payload.get("rationale"),
            }
        )
    return history


def sys_tty() -> bool:
    """Return whether both stdin and stdout are attached to a TTY."""
    return click.get_text_stream("stdin").isatty() and click.get_text_stream("stdout").isatty()


def is_float(value: str) -> bool:
    """Return whether value parses as a float."""
    try:
        float(value)
    except ValueError:
        return False
    return True


def _default_target_for_objective(objective_type: str) -> str:
    if objective_type in {"pass_rate", "check_pass_rate", "reward_model_score_mean"}:
        return "0.8"
    if objective_type == "tool_error_rate":
        return "0.1"
    if objective_type == "avg_reward":
        return "0.0"
    return "0.8"


def _render_run_preview(
    *,
    run_section: dict[str, object],
    agent_section: dict[str, object],
    proposer_section: dict[str, object],
    verifier_section: dict[str, object],
    objective_section: dict[str, object],
    budget_section: dict[str, object],
) -> str:
    env = str(run_section.get("env") or "").strip()
    tasks_dir = str(run_section.get("tasks_dir") or "").strip()
    display_tasks_dir = _display_path(Path(tasks_dir)) if tasks_dir else ""
    task_ids_obj = run_section.get("task_ids")
    task_ids = task_ids_obj if isinstance(task_ids_obj, list) else []
    runtime = str(run_section.get("runtime") or "daytona")
    rollout_count = run_section.get("rollout_count")
    max_parallel = run_section.get("max_parallel")

    objective_type = str(objective_section.get("type") or "pass_rate")
    target = objective_section.get("target")
    target_text = "disabled" if target is None else str(target)

    max_iterations = budget_section.get("max_iterations", 6)
    max_minutes = budget_section.get("max_minutes", 90)
    no_improvement_window = budget_section.get("no_improvement_window", 2)
    display_max_minutes: object = max_minutes
    if isinstance(max_minutes, int) and max_minutes < 0:
        display_max_minutes = "no limit"
    display_no_improvement: object = no_improvement_window
    if isinstance(no_improvement_window, int) and no_improvement_window <= 0:
        display_no_improvement = "no limit"

    agent_model = _model_summary(agent_section, fallback_model="gpt-4o-mini")
    proposer_model = _model_summary(proposer_section, fallback_model="gpt-5.4")
    verifier_model = _model_summary(verifier_section, fallback_model="gpt-5.4")

    lines: list[str] = []
    lines.append(
        "SimLab will first run your current setup once to get a starting score, "
        "then try one prompt edit at a time against that same setup and task folder."
    )
    lines.append("")
    lines.extend(
        _render_preview_section(
            "Fixed setup",
            [
                ("Setup", env),
                ("Task folder", display_tasks_dir),
                ("Tasks", ", ".join(str(t) for t in task_ids)),
                ("Runtime", runtime),
            ],
        )
    )
    lines.append("")
    lines.extend(
        _render_preview_section(
            "Search loop",
            [
                ("Runs per task", rollout_count),
                ("Max parallel", max_parallel),
                ("Improvement iterations", max_iterations),
                ("Objective", objective_type),
                ("Early stop target", target_text),
                ("Time limit minutes", display_max_minutes),
                ("Non-improvement streak", display_no_improvement),
            ],
        )
    )
    lines.append("")
    lines.extend(
        _render_preview_section(
            "Models",
            [
                ("Agent to evaluate", agent_model),
                ("Prompt editor", proposer_model),
                ("Rubric verifier", verifier_model),
            ],
        )
    )
    return "\n".join(lines)


def _render_preview_section(title: str, rows: list[tuple[str, object]]) -> list[str]:
    lines = [title, "-" * len(title)]
    for label, value in rows:
        lines.append(f"  {label:<24} {value}")
    return lines


def _model_summary(section: dict[str, object], *, fallback_model: str) -> str:
    provider = str(section.get("provider") or "openai").strip() or "openai"
    model = str(section.get("model") or fallback_model).strip() or fallback_model
    api_key_env = str(section.get("api_key_env") or "").strip()
    summary = f"{provider}/{model}"
    if api_key_env:
        summary += f" via {api_key_env}"
    return summary


def apply_environments_dir_override(*, ctx: click.Context, environments_dir: Path | None) -> None:
    """Apply an environments root override to the shared global config context."""
    if environments_dir is None:
        return
    ctx.ensure_object(dict)
    overrides = dict(ctx.obj.get("global_config_overrides") or {})
    overrides["environments_dir"] = str(environments_dir.resolve())
    ctx.obj["global_config_overrides"] = overrides


def ensure_autoresearch_environment_resolves(
    *,
    ctx: click.Context,
    env_name: str,
    tasks_dir: Path,
    base_dir: Path | None,
) -> None:
    """Ensure the configured environment resolves, prompting for an env root when needed.

    This is intentionally specific to autoresearch to make the command easy to run from
    any directory. We keep the strict contract, but help the user pick the right
    environments root up front instead of failing deep inside the manager.
    """
    env_name = (env_name or "").strip()
    if not env_name:
        return

    env_root = get_environments_dir(ctx=ctx)
    if env_has_env_yaml(env_root=env_root, env_name=env_name):
        return

    candidates = find_environments_dir_candidates(
        env_name=env_name,
        tasks_dir=tasks_dir,
        base_dir=base_dir,
    )
    if len(candidates) == 1:
        picked = candidates[0]
        apply_environments_dir_override(ctx=ctx, environments_dir=picked)
        click.echo(click.style(f"Using saved setups from: {picked}", fg="yellow"), err=True)
        return

    if not sys_tty():
        hint = ""
        if candidates:
            preview = "\n".join(f"  - {path}" for path in candidates[:5])
            hint = f"\nPossible saved setup directories:\n{preview}"
        raise click.ClickException(
            f"Setup '{env_name}' not found under {env_root}.{hint}\n"
            "Set --environments-dir or SIMLAB_ENVIRONMENTS_DIR."
        )

    picked = prompt_for_environments_dir(
        env_name=env_name,
        default_root=env_root,
        candidates=candidates,
    )
    apply_environments_dir_override(ctx=ctx, environments_dir=picked)


def env_has_env_yaml(*, env_root: Path, env_name: str) -> bool:
    """Return whether env_root contains env_name/env.yaml."""
    env_dir = env_root / env_name
    return env_dir.is_dir() and (env_dir / "env.yaml").is_file()


def find_environments_dir_candidates(
    *,
    env_name: str,
    tasks_dir: Path,
    base_dir: Path | None,
    max_depth: int = 8,
) -> list[Path]:
    """Find plausible environment roots relative to the tasks bundle and cwd."""
    env_name = env_name.strip()
    roots: set[Path] = set()
    starts: list[Path] = []

    resolved_tasks_dir = resolve_relative_path(path=tasks_dir, base_dir=base_dir)
    if resolved_tasks_dir is not None:
        starts.append(resolved_tasks_dir)

    if base_dir is not None:
        resolved_base_dir = resolve_relative_path(path=base_dir, base_dir=None)
        if resolved_base_dir is not None:
            starts.append(resolved_base_dir)

    starts.append(Path.cwd())

    for start in starts:
        current = start
        for _depth in range(max_depth + 1):
            for candidate_root in (current / "environments",):
                if env_has_env_yaml(env_root=candidate_root, env_name=env_name):
                    try:
                        roots.add(candidate_root.resolve())
                    except OSError:
                        roots.add(candidate_root)
            if current.parent == current:
                break
            current = current.parent

    return sorted(roots)


def resolve_relative_path(*, path: Path, base_dir: Path | None) -> Path | None:
    """Resolve path relative to base_dir when provided."""
    if not isinstance(path, Path):
        return None
    path = path.expanduser()
    if base_dir is not None:
        base_dir = base_dir.expanduser()
    if path.is_absolute():
        try:
            return path.resolve()
        except OSError:
            return path
    if base_dir is not None:
        candidate = base_dir / path
        try:
            return candidate.resolve()
        except OSError:
            return candidate
    try:
        return path.resolve()
    except OSError:
        return path


def prompt_for_environments_dir(
    *,
    env_name: str,
    default_root: Path,
    candidates: list[Path],
) -> Path:
    """Prompt for a saved setups directory, optionally from detected candidates."""
    if candidates:
        picked = select(
            "Saved setups directory",
            choices=[
                *[Choice(title=str(path), value=path) for path in candidates],
                Choice(title="Enter a different path", value=None),
            ],
            default=candidates[0],
            instruction="Use Up/Down to navigate, Enter to select.",
        ).ask()
        if picked is None:
            picked = Path(
                text(
                    "Saved setups directory",
                    default=str(default_root),
                    validate=lambda v: None if v.strip() else "Enter a path.",
                ).ask()
            ).expanduser()
    else:
        picked = Path(
            text(
                "Saved setups directory",
                default=str(default_root),
                validate=lambda v: None if v.strip() else "Enter a path.",
            ).ask()
        ).expanduser()

    if not env_has_env_yaml(env_root=picked, env_name=env_name):
        raise click.ClickException(
            f"Setup '{env_name}' not found under {picked}. "
            "Provide a directory that contains <env_name>/env.yaml. "
            "This command does not create setups. Create one with `simlab env init` or "
            "paste a path to env.yaml."
        )
    return picked


def _resolve_env_name(*, ctx: click.Context, env_name: str | None) -> str:
    env = (env_name or "").strip()
    if env:
        resolved_env, resolved_root = _parse_setup_reference(ctx=ctx, raw=env)
        if resolved_root is not None:
            apply_environments_dir_override(ctx=ctx, environments_dir=resolved_root)
        return resolved_env
    if not sys_tty():
        return ""

    env_root = get_environments_dir(ctx=ctx)
    available: list[str] = []
    if env_root.is_dir():
        for child in sorted(env_root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "env.yaml").is_file():
                available.append(child.name)

    if available:
        picked = select(
            "Which setup should SimLab reuse for every trial?",
            choices=[
                *[Choice(title=name, value=name) for name in available],
                Choice(title="Enter manually", value="__manual__"),
            ],
            default=available[0],
            instruction=(
                "Type to filter saved setups. Enter picks the one SimLab will reuse each time."
            ),
        ).ask()
        if picked != "__manual__":
            return picked

    detected = find_setup_dir_candidates(search_root=Path.cwd())
    if detected:
        picked_setup_dir = select(
            "Choose detected setup",
            choices=[
                *[Choice(title=_display_dir(path), value=path) for path in detected],
                Choice(title="Enter a different path", value=None),
            ],
            default=detected[0],
            instruction="Use Up/Down to move, type to filter, and Enter to select.",
        ).ask()
        if picked_setup_dir is not None:
            resolved_env, resolved_root = _parse_setup_reference(
                ctx=ctx,
                raw=str(picked_setup_dir),
            )
            apply_environments_dir_override(ctx=ctx, environments_dir=resolved_root)
            return resolved_env

    def validate_setup(value: str) -> str | None:
        candidate = value.strip()
        if not candidate:
            return "Enter a setup path."
        try:
            parsed_env, parsed_root = _parse_setup_reference(ctx=ctx, raw=candidate)
        except click.ClickException as exc:
            return str(exc)
        if parsed_root is not None:
            return None
        if not env_root.is_dir():
            return (
                f"No saved setups found under {env_root}. "
                "Paste a path to env.yaml or set --environments-dir."
            )
        if not env_has_env_yaml(env_root=env_root, env_name=parsed_env):
            return f"Setup '{parsed_env}' not found under {env_root}."
        return None

    raw = text(
        "Paste a path to env.yaml",
        validate=validate_setup,
    ).ask()
    resolved_env, resolved_root = _parse_setup_reference(ctx=ctx, raw=str(raw))
    if resolved_root is not None:
        apply_environments_dir_override(ctx=ctx, environments_dir=resolved_root)
    return resolved_env


def _parse_setup_reference(*, ctx: click.Context, raw: str) -> tuple[str, Path | None]:
    raw = (raw or "").strip()
    if not raw:
        return "", None

    looks_like_path = any(sep in raw for sep in ("/", "\\", "~")) or raw.endswith((".yaml", ".yml"))
    if not looks_like_path:
        return raw, None

    candidate = Path(raw).expanduser()
    resolved = resolve_relative_path(
        path=candidate,
        base_dir=Path.cwd(),
    )
    candidate = resolved if resolved is not None else candidate

    if candidate.is_dir():
        env_yaml = candidate / "env.yaml"
        if not env_yaml.is_file():
            raise click.ClickException(
                "Setup directory must contain env.yaml. "
                f"Missing: {env_yaml}. "
                "Tip: point to a directory like ./examples/custom-cli-tools or a file like "
                "./examples/custom-cli-tools/env.yaml."
            )
        return candidate.name, candidate.parent

    if candidate.is_file():
        if candidate.name != "env.yaml":
            raise click.ClickException(
                "Setup file must be named env.yaml. "
                f"Got: {candidate}. "
                "Tip: point to a file like ./examples/custom-cli-tools/env.yaml."
            )
        return candidate.parent.name, candidate.parent.parent

    env_root = get_environments_dir(ctx=ctx)
    raise click.ClickException(
        f"Setup reference not found: {raw}\n"
        f"SimLab looks for <setup>/env.yaml under {env_root} by default.\n"
        "Tip: use --environments-dir or paste a path like ./examples/custom-cli-tools/env.yaml."
    )


def _resolve_tasks_dir(
    *,
    tasks_dir: Path | None,
    require_exists: bool,
    setup_dir: Path | None = None,
) -> str:
    if isinstance(tasks_dir, Path):
        tasks_dir_str = str(tasks_dir).strip()
        if tasks_dir_str:
            tasks_dir_path = tasks_dir.expanduser()
            if tasks_dir_path.name == "tasks":
                return str(tasks_dir_path.parent)
            return tasks_dir_str
    if not sys_tty():
        return ""

    def validate_dir(value: str) -> str | None:
        value = value.strip()
        if not value:
            return "Enter a directory path."
        if not require_exists:
            return None
        candidate = resolve_relative_path(path=Path(value).expanduser(), base_dir=Path.cwd())
        candidate = candidate if candidate is not None else Path(value).expanduser()
        if candidate.is_dir():
            return None
        if candidate.exists():
            return f"Expected a directory, got a file: {value}."
        return f"Folder does not exist: {value}. Pick an existing folder."

    detected: list[Path] = []
    if setup_dir is not None and setup_dir.is_dir():
        for root in (setup_dir, setup_dir.parent):
            detected = find_task_dir_candidates(search_root=root)
            if detected:
                break
    if not detected:
        detected = find_task_dir_candidates(search_root=Path.cwd())
    if detected and setup_dir is not None:
        resolved_setup_dir = setup_dir.resolve()

        def sort_key(candidate: Path) -> tuple[int, int, str]:
            try:
                rel = candidate.resolve().relative_to(resolved_setup_dir)
                return (0, len(rel.parts), str(rel))
            except ValueError:
                pass
            try:
                rel = candidate.resolve().relative_to(resolved_setup_dir.parent)
                return (1, len(rel.parts), str(rel))
            except ValueError:
                pass
            return (2, len(candidate.parts), str(candidate))

        detected = sorted(detected, key=sort_key)
    if detected:
        picked = select(
            "Task folder",
            choices=[
                *[Choice(title=_display_dir(path), value=path) for path in detected],
                Choice(title="Enter a different path", value=None),
            ],
            default=detected[0],
            instruction="Use Up/Down to move, type to filter, and Enter to select.",
        ).ask()
        if picked is not None:
            return str(picked.parent)

    default = "./generated-tasks"
    typed = text(
        "Task folder",
        default=default,
        validate=validate_dir,
    ).ask()
    picked = resolve_relative_path(path=Path(typed).expanduser(), base_dir=Path.cwd())
    picked = picked if picked is not None else Path(typed).expanduser()
    if picked.name == "tasks":
        return str(picked.parent)
    return str(picked)


def _resolve_task_ids(*, task_ids: Iterable[str], tasks_dir: Path) -> list[str]:
    task_list = [t.strip() for t in task_ids if isinstance(t, str) and t.strip()]
    if task_list:
        return task_list
    if not sys_tty():
        return []

    discovered = _discover_tasks(tasks_dir)
    if discovered:
        choices: list[Choice[str]] = []
        choices.append(Choice(title="Select all", value=TASK_SELECT_ALL_VALUE))
        for task_id, display_name, has_checks in discovered:
            clean_name = (display_name or "").replace("\n", " ").strip()
            if len(clean_name) > 80:
                clean_name = clean_name[:77] + "..."
            title = task_id if not clean_name else f"{task_id}  {clean_name}"
            if not has_checks:
                title = f"{title}  no checks"
            choices.append(Choice(title=title, value=task_id))
        return checkbox(
            "Select tasks to run",
            choices=choices,
            min_selected=1,
            select_all_value=TASK_SELECT_ALL_VALUE,
            instruction=(
                "Use Up/Down to move between tasks, Space to select, and Enter to confirm."
            ),
        ).ask()

    raise click.ClickException(
        f"No task JSON files found under `{_display_dir(_tasks_search_root(tasks_dir))}`."
    )


def _discover_tasks(tasks_dir: Path) -> list[tuple[str, str, bool]]:
    search_root = _tasks_search_root(tasks_dir)
    if not search_root.is_dir():
        return []

    discovered: list[tuple[str, str, bool]] = []
    for path in sorted(search_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        task_id = str(meta.get("task_id") or "").strip()
        display_name = str(meta.get("display_name") or "").strip()
        verifiers = payload.get("verifiers")
        has_checks = isinstance(verifiers, list) and bool(verifiers)
        if not task_id:
            task_id = path.stem.replace("_", "-")
        discovered.append((task_id, display_name, has_checks))

    return sorted(discovered, key=lambda pair: pair[0])


def _parse_task_selection(raw: str, discovered: list[tuple[str, str, bool]]) -> list[str]:
    tokens = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    selected: list[str] = []
    for token in tokens:
        if token.isdigit() and discovered:
            idx = int(token)
            if 1 <= idx <= len(discovered):
                selected.append(discovered[idx - 1][0])
                continue
        selected.append(token)
    return [t for t in selected if t]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _display_dir(path: Path) -> str:
    display = _display_path(path)
    if not display:
        return display
    return display if display.endswith("/") else f"{display}/"


def _tasks_search_root(tasks_dir: Path) -> Path:
    tasks_root = tasks_dir / "tasks"
    return tasks_root if tasks_root.is_dir() else tasks_dir


def find_task_dir_candidates(*, search_root: Path, max_depth: int = 8) -> list[Path]:
    """Find leaf `tasks/` directories that contain only JSON task files."""
    root = search_root.resolve()
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "output",
        "dist",
        "build",
    }
    candidates: set[Path] = set()

    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.parts) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue

        dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".")]

        if current.name != "tasks":
            continue

        if dirnames:
            continue

        if not filenames:
            continue

        if any(not name.endswith(".json") for name in filenames):
            continue

        candidates.add(current)

    return sorted(candidates)


def find_setup_dir_candidates(*, search_root: Path, max_depth: int = 8) -> list[Path]:
    """Find directories that contain an env.yaml file."""
    root = search_root.resolve()
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "output",
        "dist",
        "build",
    }
    candidates: set[Path] = set()

    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.parts) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue

        dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".")]

        if "env.yaml" not in filenames:
            continue

        candidates.add(current)

    return sorted(candidates)


def _print_wizard_intro() -> None:
    _print_guidance(
        "SimLab prompt optimization",
        [
            "SimLab improves the scenario instructions in an autonomous loop until it hits the "
            "goal you set. It first runs your tasks once to get a baseline score, then reruns "
            "the same tasks after each small prompt edit and keeps the best result in a "
            "before-and-after report.",
        ],
    )


def _print_environment_step() -> None:
    _print_guidance(
        "Choose setup",
        [
            "A setup is a folder named <setup>/ that contains env.yaml. If you type only a setup "
            "name, SimLab looks for <setup>/env.yaml under ./environments unless you pass "
            "--environments-dir. Example: ./examples/custom-cli-tools/",
        ],
    )


def _print_tasks_dir_step() -> None:
    _print_guidance(
        "Choose task folder",
        [
            "SimLab starts by looking for a task bundle near the setup you picked. If it cannot "
            "find one, it searches the current directory. Pick a `tasks/` folder from the list, "
            "or choose Enter a different path if your tasks live elsewhere.",
        ],
    )


def _print_task_selection_step(
    *,
    tasks_dir: Path,
    discovered: list[tuple[str, str, bool]],
) -> None:
    lines = [
        f"Using tasks from `{_display_dir(_tasks_search_root(tasks_dir))}`.",
    ]
    if discovered:
        count = len(discovered)
        noun = "task file" if count == 1 else "task files"
        lines.insert(1, f"Found {count} {noun} in this folder.")
        with_checks = sum(1 for _task_id, _name, has_checks in discovered if has_checks)
        without_checks = count - with_checks
        if without_checks:
            task_noun = "task" if without_checks == 1 else "tasks"
            lines.insert(
                2,
                f"{without_checks} {task_noun} have no checks, so task success rate will count "
                "them as failures.",
            )
    _print_guidance("Choose tasks", lines)


def _print_runtime_step(*, daytona_note: str | None) -> None:
    lines = [
        "Pick where each trial should run. Local uses Docker on this machine. Daytona uses "
        "remote sandboxes and is better for larger repeated comparisons.",
    ]
    if daytona_note:
        lines.append(daytona_note)
    _print_guidance("Choose where to run", lines)


def _daytona_runtime_available(*, ctx: click.Context) -> tuple[bool, str | None]:
    if DAYTONA_IMPORT_ERROR is not None:
        return (
            False,
            "Daytona is unavailable because the optional daytona package is not installed. "
            "Enable it with `uv sync --extra daytona`. This run will use local Docker.",
        )

    api_key = resolve_daytona_api_key(ctx=ctx)
    if not api_key:
        return (
            False,
            "Daytona is unavailable because no API key is configured. "
            "Set `SIMLAB_DAYTONA_API_KEY` or `DAYTONA_API_KEY`, or pass `--daytona-api-key`. "
            "This run will use local Docker.",
        )

    return True, None


def _print_iteration_budget_step(*, runtime: str, default_iterations: int) -> None:
    runtime_context = (
        "Local runs default small because Docker on one machine is best for quick checks."
        if runtime == "local"
        else (
            "Daytona defaults are larger because remote runs are better suited for longer searches."
        )
    )
    iteration_label = "iteration" if default_iterations == 1 else "iterations"
    _print_guidance(
        "Choose number of iterations",
        [
            f"{runtime_context} SimLab always runs your tasks once to get a starting score. "
            f"The default here is {default_iterations} {iteration_label}. Each iteration makes one "
            "prompt edit, reruns all selected tasks, and compares the new score to the best "
            "result so far.",
            "Even if you pick a large number, SimLab can stop early once it hits your goal, runs "
            "out of time, or goes several iterations in a row without improving the score.",
        ],
    )


def _print_model_step() -> None:
    _print_guidance(
        "Choose AI models",
        [
            "SimLab can use one model to do the work, one to suggest prompt changes, "
            "and one to score rubric-based tasks. You will pick all three now. "
            "You can reuse the same model for multiple roles.",
            "SimLab calls models through LiteLLM. When you edit a role, you will pick a "
            "model and provider from a filterable catalog.",
        ],
    )


def _print_scoring_step() -> None:
    _print_guidance(
        "Choose success goal",
        [
            "Tell SimLab which metric to optimize and when it should stop researching.",
            "By default, autoresearch optimizes for task success rate and stops when the "
            "target is reached. It will also stop when time is up or after a short streak "
            "of edits that do not improve the score.",
        ],
    )


def _print_guidance(title: str, lines: list[str]) -> None:
    click.echo()
    click.echo(click.style(title, bold=True))
    click.echo("-" * min(max(len(title), 12), WIZARD_RULE_WIDTH))
    for line in lines:
        click.echo(line)


def _print_default_success_goal(
    *,
    target: str,
    max_minutes: str,
    no_improvement_window: str,
) -> None:
    click.echo()
    click.echo("Defaults for this run")
    click.echo("  Metric                     Task success rate")
    click.echo(f"  Target                     {target}")
    click.echo(f"  Time limit                 {max_minutes} minutes")
    click.echo(f"  Non-improvement streak     {no_improvement_window}")


def _configure_model_sections(
    *,
    agent_section: dict[str, object],
    proposer_section: dict[str, object],
    verifier_section: dict[str, object],
) -> None:
    _prompt_model_section(
        role="agent",
        section=agent_section,
    )
    _prompt_model_section(
        role="proposer",
        section=proposer_section,
    )
    _prompt_model_section(
        role="verifier",
        section=verifier_section,
    )


def _prompt_model_section(
    *,
    role: str,
    section: dict[str, object],
) -> None:
    role_label = {
        "agent": "agent to evaluate",
        "proposer": "prompt editor",
        "verifier": "rubric verifier",
    }[role]
    role_heading = {
        "agent": "Edit agent to evaluate",
        "proposer": "Edit prompt editor",
        "verifier": "Edit rubric verifier",
    }[role]
    click.echo()
    click.echo(click.style(role_heading, bold=True))
    if role == "proposer":
        click.echo("This is the model that will propose changes to the Scenario prompt.")
    click.echo(
        "Pick a model, then pick a provider. Type to filter. "
        "These lists come from LiteLLM's built-in catalog."
    )

    current_provider = str(section.get("provider") or "openai").strip() or "openai"
    current_model = str(section.get("model") or "").strip()
    if current_provider and current_model.startswith(f"{current_provider}/"):
        current_model = current_model[len(current_provider) + 1 :]

    raw_model, is_custom = _pick_litellm_model(
        role_label=role_label,
        current_model=current_model,
    )
    provider_hint = None
    model = raw_model
    if is_custom:
        provider_hint, model = _split_provider_prefixed_model(raw_model)
    section["model"] = model

    provider, is_custom_provider = _pick_litellm_provider_for_model(
        role_label=role_label,
        model=model,
        current_provider=current_provider,
        provider_hint=provider_hint,
        allow_custom_provider=is_custom,
    )
    section["provider"] = provider

    if is_custom or is_custom_provider:
        section["api_key_env"] = (
            text(
                f"API key env var for the {role_label}",
                default=str(section.get("api_key_env") or _default_api_key_env(provider)),
                validate=lambda v: None if v.strip() else "Enter an env var name.",
            )
            .ask()
            .strip()
        )
        base_url_default = str(section.get("base_url_env") or _default_base_url_env(provider) or "")
        base_url_env = (
            text(
                f"Base URL env var for the {role_label}. Only needed for proxies or self-hosted "
                "endpoints. Leave blank to disable it.",
                default=base_url_default,
            )
            .ask()
            .strip()
        )
        section["base_url_env"] = base_url_env or None
    else:
        section["api_key_env"] = _default_api_key_env(provider)
        section["base_url_env"] = _default_base_url_env(provider)


def _provider_env_prefix(provider: str) -> str:
    provider = (provider or "").strip()
    if not provider:
        return "PROVIDER"
    cleaned: list[str] = []
    for ch in provider.upper():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    prefix = "".join(cleaned)
    while "__" in prefix:
        prefix = prefix.replace("__", "_")
    prefix = prefix.strip("_")
    return prefix or "PROVIDER"


def _default_api_key_env(provider: str) -> str:
    provider = (provider or "").strip()
    if not provider:
        return "OPENAI_API_KEY"

    normalized = provider.lower()
    candidates = {
        "azure": ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
    }.get(normalized)
    if candidates is None:
        candidates = [f"{_provider_env_prefix(provider)}_API_KEY"]
    for candidate in candidates:
        if os.environ.get(candidate):
            return candidate
    return candidates[0]


def _default_base_url_env(provider: str) -> str | None:
    provider = (provider or "").strip()
    if not provider:
        return "OPENAI_API_BASE"

    normalized = provider.lower()
    candidates = {
        "azure": ["AZURE_API_BASE"],
        "openai": ["OPENAI_API_BASE"],
    }.get(normalized)
    if candidates is None:
        candidates = [f"{_provider_env_prefix(provider)}_API_BASE"]
    for candidate in candidates:
        if os.environ.get(candidate):
            return candidate
    return candidates[0]


def _pick_litellm_model(*, role_label: str, current_model: str) -> tuple[str, bool]:
    _providers, providers_by_model = _list_litellm_catalog()
    models = sorted(providers_by_model, key=lambda name: ("/" in name, name))
    custom_value = "__custom_model__"
    choices: list[Choice[str]] = [Choice(title=model, value=model) for model in models]
    choices.append(Choice(title="Enter a custom model name", value=custom_value))
    current_model = (current_model or "").strip()
    default_value: str | None
    if not current_model:
        default_value = None
    elif current_model in models:
        default_value = current_model
    else:
        default_value = custom_value

    selected = select(
        f"Which model should SimLab use for the {role_label}?",
        choices=choices,
        default=default_value,
    ).ask()
    if selected != custom_value:
        return selected.strip(), False
    return (
        text(
            f"Model for the {role_label}",
            default=current_model,
            validate=lambda v: None if v.strip() else "Enter a model name.",
        )
        .ask()
        .strip()
    ), True


def _pick_litellm_provider_for_model(
    *,
    role_label: str,
    model: str,
    current_provider: str,
    provider_hint: str | None,
    allow_custom_provider: bool,
) -> tuple[str, bool]:
    providers, providers_by_model = _list_litellm_catalog()
    allowed_providers = sorted(providers_by_model.get(model) or providers)
    if not allowed_providers:
        allowed_providers = [current_provider or "openai"]
    if provider_hint and provider_hint not in allowed_providers:
        allowed_providers.append(provider_hint)
        allowed_providers.sort()

    choices: list[Choice[str]] = [
        Choice(
            title=f"{provider} {_litellm_provider_docs_url(provider)}",
            value=provider,
        )
        for provider in allowed_providers
    ]
    custom_value = "__custom_provider__"
    if allow_custom_provider:
        choices.append(Choice(title="Enter a provider not in this list", value=custom_value))

    if current_provider in allowed_providers:
        default_value = current_provider
    else:
        default_value = allowed_providers[0]
    if provider_hint and provider_hint in allowed_providers:
        default_value = provider_hint

    selected = select(
        f"Which provider should SimLab use for the {role_label}?",
        choices=choices,
        default=default_value,
    ).ask()
    if selected != custom_value or not allow_custom_provider:
        return selected.strip(), False
    return (
        text(
            f"Provider for the {role_label}",
            default=current_provider,
            validate=lambda v: None if v.strip() else "Enter a provider name.",
        )
        .ask()
        .strip()
    ), True


def _split_provider_prefixed_model(raw_model: str) -> tuple[str | None, str]:
    """Split a `provider/model` input into provider hint + model string."""
    raw_model = (raw_model or "").strip()
    if not raw_model or "/" not in raw_model:
        return None, raw_model

    providers, _providers_by_model = _list_litellm_catalog()
    provider_prefix, remainder = raw_model.split("/", 1)
    provider_prefix = provider_prefix.strip()
    remainder = remainder.strip()
    if provider_prefix and remainder and provider_prefix in set(providers):
        return provider_prefix, remainder
    return None, raw_model


def _list_litellm_catalog() -> tuple[list[str], dict[str, set[str]]]:
    """Return (providers, model -> providers) from LiteLLM's built-in catalog."""
    import litellm  # noqa: PLC0415

    def normalize(value: object) -> str:
        raw = getattr(value, "value", value)
        return str(raw).strip()

    by_provider = getattr(litellm, "models_by_provider", {})
    providers: set[str] = set()
    providers_by_model: dict[str, set[str]] = {}

    if isinstance(by_provider, dict):
        for provider, raw_models in by_provider.items():
            normalized_provider = normalize(provider)
            if not normalized_provider:
                continue
            providers.add(normalized_provider)
            for model in raw_models:
                normalized_model = normalize(model)
                if not normalized_model:
                    continue
                if normalized_model.startswith(f"{normalized_provider}/"):
                    normalized_model = normalized_model[len(normalized_provider) + 1 :]
                providers_by_model.setdefault(normalized_model, set()).add(normalized_provider)

    return sorted(providers), providers_by_model


def _litellm_provider_docs_url(provider: str) -> str:
    provider = (provider or "").strip()
    if not provider:
        return "https://docs.litellm.ai/docs/providers"

    slug_overrides = {
        "azure_anthropic": "azure",
        "azure_text": "azure",
        "cohere_chat": "cohere",
        "ollama_chat": "ollama",
        "text-completion-codestral": "codestral",
        "text-completion-openai": "openai",
        "together_ai": "togetherai",
        "vertex_ai": "vertex",
    }
    doc_slug = slug_overrides.get(provider)
    if doc_slug is None:
        doc_slug = provider

    # Some provider keys do not have a dedicated page. Fall back to the index.
    if provider in {
        "assemblyai",
        "cloudflare",
        "maritalk",
        "palm",
        "runwayml",
        "volcengine",
        "wandb",
    }:
        return "https://docs.litellm.ai/docs/providers"

    return f"https://docs.litellm.ai/docs/providers/{doc_slug}"
