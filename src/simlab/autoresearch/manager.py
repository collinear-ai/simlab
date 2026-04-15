"""Autoresearch manager loop.

This module owns the sequential baseline -> analyze -> propose -> rerun -> compare
loop for v1 prompt-only autoresearch.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path

import click

from simlab.autoresearch.analysis import build_analysis
from simlab.autoresearch.config import AutoresearchRunConfig
from simlab.autoresearch.objectives import extract_metrics
from simlab.autoresearch.objectives import extract_objective_value
from simlab.autoresearch.objectives import is_better_result
from simlab.autoresearch.objectives import objective_direction
from simlab.autoresearch.propose import propose_next_change
from simlab.autoresearch.report import write_end_of_run_reports
from simlab.autoresearch.runner import run_rollouts
from simlab.autoresearch.validate import parse_and_validate_proposal
from simlab.autoresearch.validate import validate_scenario_prompt
from simlab.cli.eval import render_markdown_report
from simlab.config import GlobalConfig
from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_env_dir
from simlab.env_artifacts import load_env_config
from simlab.evaluation import EvaluationError
from simlab.evaluation import build_report
from simlab.runtime.rollout_runner import find_rubric_file
from simlab.runtime.rollout_runner import load_local_task
from simlab.runtime.rollout_runner import normalize_tasks_bundle_dir


@dataclass(frozen=True)
class IterationPaths:
    """Filesystem paths for one baseline or iteration package."""

    iteration: int
    root: Path
    output_root: Path


def raise_missing_proposer_api_key(*, api_key_env: str) -> None:
    """Raise the standardized missing proposer API key error."""
    raise click.ClickException(f"Missing proposer API key in env var {api_key_env}")


def run_autoresearch(
    *,
    cfg: AutoresearchRunConfig,
    ctx: click.Context,
    output_base_dir: Path = Path("output"),
) -> Path:
    """Run one full autoresearch loop and write the output package to disk."""
    tasks_bundle_dir = normalize_tasks_bundle_dir(_resolve_tasks_dir_path(cfg.run.tasks_dir))
    if str(tasks_bundle_dir) != cfg.run.tasks_dir:
        cfg = cfg.model_copy(
            update={
                "run": cfg.run.model_copy(
                    update={
                        "tasks_dir": str(tasks_bundle_dir),
                    }
                )
            }
        )

    run_dir = _create_run_dir(output_base_dir)
    _write_text(run_dir / "run.toml", dump_run_config_toml(cfg))

    global_cfg = get_global_config_from_ctx(ctx)
    global_cfg = _apply_verifier_overrides(global_cfg, cfg)

    env_cfg = load_env_config(resolve_env_dir(cfg.run.env, ctx=ctx))
    baseline_prompt = (env_cfg.scenario_guidance_md or "").strip()
    validate_scenario_prompt(baseline_prompt or "")

    required_headings = _extract_required_headings(baseline_prompt)

    baseline_paths = IterationPaths(
        iteration=0,
        root=run_dir / "baseline",
        output_root=run_dir / "baseline" / "output",
    )
    baseline_paths.root.mkdir(parents=True, exist_ok=True)
    baseline_paths.output_root.mkdir(parents=True, exist_ok=True)

    _write_text(baseline_paths.root / "scenario_prompt.md", baseline_prompt + "\n")
    _write_json(
        baseline_paths.root / "harness.json",
        _harness_json(cfg, scenario_prompt=baseline_prompt),
    )

    agent_api_key = cfg.agent.resolve_api_key()
    if not agent_api_key:
        raise click.ClickException(f"Missing agent API key in env var {cfg.agent.api_key_env}")

    bundle_has_rubrics = _bundle_has_rubric_files(
        bundle_dir=tasks_bundle_dir, task_ids=cfg.run.task_ids
    )
    if cfg.objective.type == "reward_model_score_mean" and not bundle_has_rubrics:
        raise click.ClickException(
            "Objective reward_model_score_mean requires rubric files under <bundle_dir>/rubrics."
        )
    if bundle_has_rubrics:
        verifier_api_key = cfg.verifier.resolve_api_key()
        if not verifier_api_key:
            raise click.ClickException(
                f"Missing verifier API key in env var {cfg.verifier.api_key_env}"
            )

    start = time.monotonic()
    click.echo(click.style(f"Autoresearch run: {run_dir}", bold=True))

    best_iteration = 0
    best_prompt = baseline_prompt
    latest_iteration = 0
    baseline_eval: dict[str, object] | None = None
    baseline_result: dict[str, object] | None = None
    best_eval: dict[str, object] | None = None
    best_result: dict[str, object] | None = None
    latest_eval: dict[str, object] | None = None
    latest_result: dict[str, object] | None = None
    history: list[dict[str, object]] = []
    rejected_streak = 0
    finalize_started = False
    stop_reason = ""

    iters_dir = run_dir / "iterations"
    iters_dir.mkdir(parents=True, exist_ok=True)

    try:
        click.echo(
            "SimLab will run a baseline first, then try one prompt edit at a time "
            "against the same setup and task folder."
        )
        click.echo(
            "It keeps the best result so far and stops when it reaches your target, "
            "hits the time budget, or runs out of useful retries."
        )
        click.echo()
        _print_prompt_block(
            label="Starting prompt",
            prompt=baseline_prompt,
        )
        click.echo()
        click.echo("[baseline] Running the fixed setup to establish the score to beat...")
        baseline_eval, baseline_result = _run_and_eval_iteration(
            cfg=cfg,
            ctx=ctx,
            global_cfg=global_cfg,
            iteration_paths=baseline_paths,
            scenario_prompt=baseline_prompt,
            run_label="baseline",
        )

        best_eval = baseline_eval
        best_result = baseline_result
        latest_eval = baseline_eval
        latest_result = baseline_result
        history = [
            {
                "iteration": 0,
                "surface": "baseline",
                "accepted": True,
                "objective_value": baseline_result.get("objective_value"),
                "change_type": None,
                "rationale": "baseline",
            }
        ]
        click.echo(
            f"[baseline] {cfg.objective.type}={baseline_result.get('objective_value')} | "
            "starting improvement loop"
        )

        if _target_reached(cfg, best_result):
            stop_reason = f"Reached the target for {cfg.objective.type} ({cfg.objective.target})."
            _finalize_run(
                run_dir=run_dir,
                cfg=cfg,
                baseline_paths=baseline_paths,
                iters_dir=iters_dir,
                best_iteration=best_iteration,
                best_prompt=best_prompt,
                best_result=best_result,
                baseline_result=baseline_result,
                baseline_prompt=baseline_prompt,
                history=history,
                stop_reason=stop_reason,
            )
            finalize_started = True
            return run_dir

        if cfg.budget.max_iterations > 0:
            should_validate_proposer_key = cfg.budget.max_minutes <= 0
            if cfg.budget.max_minutes > 0:
                elapsed_minutes = (time.monotonic() - start) / 60.0
                should_validate_proposer_key = elapsed_minutes < cfg.budget.max_minutes

            if should_validate_proposer_key:
                proposer_api_key = cfg.proposer.resolve_api_key()
                if not proposer_api_key:
                    raise_missing_proposer_api_key(api_key_env=cfg.proposer.api_key_env)

        for iteration in range(1, cfg.budget.max_iterations + 1):
            if cfg.budget.max_minutes > 0:
                elapsed_minutes = (time.monotonic() - start) / 60.0
                if elapsed_minutes >= cfg.budget.max_minutes:
                    stop_reason = f"Reached the time budget of {cfg.budget.max_minutes} minute(s)."
                    break

            iter_paths = IterationPaths(
                iteration=iteration,
                root=iters_dir / f"iter_{iteration:03d}",
                output_root=iters_dir / f"iter_{iteration:03d}" / "output",
            )
            iter_paths.root.mkdir(parents=True, exist_ok=True)
            iter_paths.output_root.mkdir(parents=True, exist_ok=True)

            click.echo(
                f"[iter {iteration}] Analyzing prior results and drafting one prompt edit..."
            )
            analysis = build_analysis(
                objective_type=cfg.objective.type,
                objective_target=cfg.objective.target,
                best_iteration=best_iteration,
                best_result=best_result,
                best_eval=best_eval,
                latest_iteration=latest_iteration,
                latest_result=latest_result,
                latest_eval=latest_eval,
                history=history,
            )
            _write_json(iter_paths.root / "analysis.json", analysis)

            proposed_payload = propose_next_change(
                proposer=cfg.proposer,
                iteration=iteration,
                analysis=analysis,
                current_prompt=best_prompt,
                prompt_required_headings=required_headings,
            )
            proposal = parse_and_validate_proposal(
                proposed_payload,
                required_headings=required_headings,
            )
            _write_json(iter_paths.root / "proposal.json", proposal.model_dump(by_alias=True))

            candidate_prompt = proposal.changes.scenario_prompt.strip()
            _write_text(iter_paths.root / "scenario_prompt.md", candidate_prompt + "\n")
            _write_json(
                iter_paths.root / "harness.json",
                _harness_json(
                    cfg,
                    scenario_prompt=candidate_prompt,
                    change_type=proposal.change_type,
                    rationale=proposal.rationale,
                ),
            )

            click.echo()
            _print_prompt_block(
                label=f"[iter {iteration}] Candidate prompt",
                prompt=candidate_prompt,
            )
            click.echo()
            click.echo(
                f"[iter {iteration}] Running the same evaluation with the candidate prompt..."
            )
            candidate_eval, candidate_result = _run_and_eval_iteration(
                cfg=cfg,
                ctx=ctx,
                global_cfg=global_cfg,
                iteration_paths=iter_paths,
                scenario_prompt=candidate_prompt,
                run_label=f"iter_{iteration:03d}",
            )

            latest_iteration = iteration
            latest_eval = candidate_eval
            latest_result = candidate_result

            accepted = is_better_result(
                candidate=candidate_result,
                best=best_result,
                objective_type=cfg.objective.type,
            )
            candidate_result["accepted"] = accepted
            candidate_result["change_type"] = proposal.change_type
            candidate_result["rationale"] = proposal.rationale
            _write_json(iter_paths.root / "result.json", candidate_result)

            history.append(
                {
                    "iteration": iteration,
                    "surface": "scenario_prompt",
                    "accepted": accepted,
                    "objective_value": candidate_result.get("objective_value"),
                    "change_type": proposal.change_type,
                    "rationale": proposal.rationale,
                }
            )

            _print_iteration_line(
                iteration=iteration,
                cfg=cfg,
                best=best_result,
                candidate=candidate_result,
                rationale=proposal.rationale,
                accepted=accepted,
            )

            if accepted:
                best_iteration = iteration
                best_prompt = candidate_prompt
                best_eval = candidate_eval
                best_result = candidate_result
                rejected_streak = 0
            else:
                rejected_streak += 1

            if _target_reached(cfg, best_result):
                stop_reason = (
                    f"Reached the target for {cfg.objective.type} ({cfg.objective.target})."
                )
                break
            if should_stop_no_improvement(
                rejected_streak=rejected_streak,
                window=cfg.budget.no_improvement_window,
            ):
                stop_reason = f"Stopped after {rejected_streak} rejected iteration(s) in a row."
                break
        if not stop_reason:
            if cfg.budget.max_iterations == 0:
                stop_reason = "Baseline-only run complete."
            else:
                stop_reason = f"Reached max_iterations={cfg.budget.max_iterations}."
        _finalize_run(
            run_dir=run_dir,
            cfg=cfg,
            baseline_paths=baseline_paths,
            iters_dir=iters_dir,
            best_iteration=best_iteration,
            best_prompt=best_prompt,
            best_result=best_result,
            baseline_result=baseline_result,
            baseline_prompt=baseline_prompt,
            history=history,
            stop_reason=stop_reason,
        )
        finalize_started = True
    except click.ClickException as exc:
        if not finalize_started and baseline_result is not None and best_result is not None:
            try:
                _finalize_run(
                    run_dir=run_dir,
                    cfg=cfg,
                    baseline_paths=baseline_paths,
                    iters_dir=iters_dir,
                    best_iteration=best_iteration,
                    best_prompt=best_prompt,
                    best_result=best_result,
                    baseline_result=baseline_result,
                    baseline_prompt=baseline_prompt,
                    history=history,
                    stop_reason=stop_reason or str(exc) or "Run stopped early.",
                )
            except Exception as finalize_exc:
                click.echo(
                    click.style(
                        f"Failed to write final reports after click error: {finalize_exc}",
                        fg="red",
                    ),
                    err=True,
                )
        raise
    except (Exception, SystemExit) as exc:
        crash_payload: dict[str, object] = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        if isinstance(exc, SystemExit):
            crash_payload["exit_code"] = exc.code
        _write_json(run_dir / "crash.json", crash_payload)
        if not finalize_started and baseline_result is not None and best_result is not None:
            try:
                _finalize_run(
                    run_dir=run_dir,
                    cfg=cfg,
                    baseline_paths=baseline_paths,
                    iters_dir=iters_dir,
                    best_iteration=best_iteration,
                    best_prompt=best_prompt,
                    best_result=best_result,
                    baseline_result=baseline_result,
                    baseline_prompt=baseline_prompt,
                    history=history,
                    stop_reason=stop_reason or "Run crashed after baseline.",
                )
            except Exception as finalize_exc:
                click.echo(
                    click.style(
                        f"Failed to write final reports after crash: {finalize_exc}",
                        fg="red",
                    ),
                    err=True,
                )
        raise
    else:
        return run_dir


def should_stop_no_improvement(*, rejected_streak: int, window: int) -> bool:
    """Return whether the run should stop after repeated rejected iterations."""
    if window <= 0:
        return False
    return rejected_streak >= window


def _print_prompt_block(*, label: str, prompt: str) -> None:
    prompt_text = (prompt or "").rstrip()
    click.echo(click.style(label, bold=True))
    click.echo(prompt_text)


def _finalize_run(
    *,
    run_dir: Path,
    cfg: AutoresearchRunConfig,
    baseline_paths: IterationPaths,
    iters_dir: Path,
    best_iteration: int,
    best_prompt: str,
    best_result: dict[str, object],
    baseline_result: dict[str, object],
    baseline_prompt: str,
    history: list[dict[str, object]],
    stop_reason: str,
) -> None:
    _write_best(run_dir=run_dir, cfg=cfg, best_prompt=best_prompt, best_result=best_result)

    baseline_output = baseline_paths.output_root
    best_output = (
        baseline_paths.output_root
        if best_iteration == 0
        else (iters_dir / f"iter_{best_iteration:03d}" / "output")
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
        stop_reason=stop_reason,
    )

    best_obj = best_result.get("objective_value")
    if stop_reason:
        click.echo(f"Stop reason: {stop_reason}")
    click.echo(
        click.style(
            f"\nWrote autoresearch results to {run_dir} (best {cfg.objective.type}={best_obj})",
            fg="green",
        )
    )
    click.echo()
    click.echo("Rerun this exact setup without the wizard:")
    click.echo(f"  simlab autoresearch run --config {run_dir / 'run.toml'}")


def _run_and_eval_iteration(
    *,
    cfg: AutoresearchRunConfig,
    ctx: click.Context,
    global_cfg: GlobalConfig,
    iteration_paths: IterationPaths,
    scenario_prompt: str,
    run_label: str,
) -> tuple[dict[str, object], dict[str, object]]:
    tasks_bundle_dir = normalize_tasks_bundle_dir(Path(cfg.run.tasks_dir))
    run_rollouts(
        env_name=cfg.run.env,
        tasks_dir=tasks_bundle_dir.expanduser().resolve(),
        task_ids=cfg.run.task_ids,
        runtime=cfg.run.runtime,
        scenario_prompt=scenario_prompt,
        model=cfg.agent.model,
        provider=cfg.agent.provider,
        api_key=cfg.agent.resolve_api_key(),
        base_url=cfg.agent.resolve_base_url(),
        rollout_count=cfg.run.rollout_count,
        max_parallel=cfg.run.max_parallel,
        max_steps=cfg.run.max_steps,
        agent_timeout_seconds=cfg.run.agent_timeout_seconds,
        no_seed=cfg.run.no_seed,
        global_cfg=global_cfg,
        ctx=ctx,
        output_root=iteration_paths.output_root,
    )

    try:
        eval_report = build_report(iteration_paths.output_root)
    except EvaluationError as exc:
        failure_hint = _render_rollout_failure_hint(iteration_paths.output_root)
        raise click.ClickException(
            "Autoresearch could not score the run because no rollout artifacts were produced. "
            f"{exc}\n{failure_hint}"
        ) from exc
    _write_json(iteration_paths.root / "eval.json", eval_report)
    _write_text(iteration_paths.root / "eval-report.md", render_markdown_report(eval_report))

    metrics = extract_metrics(eval_report)
    objective_value = extract_objective_value(eval_report, cfg.objective.type)
    result: dict[str, object] = {
        "iteration": iteration_paths.iteration,
        "label": run_label,
        "objective_type": cfg.objective.type,
        "objective_value": objective_value,
        "pass_rate": metrics.get("pass_rate"),
        "avg_reward": metrics.get("avg_reward"),
        "check_pass_rate": metrics.get("check_pass_rate"),
        "reward_model_score_mean": metrics.get("reward_model_score_mean"),
        "tool_error_rate": metrics.get("tool_error_rate"),
    }
    if iteration_paths.iteration == 0:
        _write_json(iteration_paths.root / "result.json", result)
    return eval_report, result


def _render_rollout_failure_hint(output_root: Path) -> str:
    """Return a short hint from any run-set summary.json files under output_root."""
    errors: list[str] = []
    try:
        children = sorted(output_root.iterdir(), key=lambda item: item.name)
    except OSError:
        children = []
    for child in children:
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            continue
        for entry in results:
            if not isinstance(entry, dict):
                continue
            raw_error = entry.get("error")
            if not isinstance(raw_error, str) or not raw_error.strip():
                continue
            errors.append(raw_error.strip())
            if len(errors) >= 3:
                break
        if len(errors) >= 3:
            break

    if not errors:
        return f"Look for task logs under {output_root}."

    seen: set[str] = set()
    unique: list[str] = []
    for error in errors:
        if error in seen:
            continue
        seen.add(error)
        unique.append(error)

    lines: list[str] = []
    lines.append("Recent rollout failure(s):")
    for error in unique:
        clipped_lines = error.splitlines()
        clipped = "\n".join(clipped_lines[:8]).rstrip()
        if len(clipped_lines) > 8:
            clipped = clipped + "\n..."
        lines.append(f"- {clipped}")
    lines.append("")
    lines.append(
        "Fix: ensure your setup directory has generated files like docker-compose.yml and any "
        "build contexts it references (for example gateway/ for mcp-gateway)."
    )
    return "\n".join(lines)


def _print_iteration_line(
    *,
    iteration: int,
    cfg: AutoresearchRunConfig,
    best: dict[str, object],
    candidate: dict[str, object],
    rationale: str,
    accepted: bool,
) -> None:
    obj = cfg.objective.type
    best_val = best.get("objective_value")
    cand_val = candidate.get("objective_value")
    direction = objective_direction(obj)
    decision = "Updated best prompt" if accepted else "Kept best prompt"

    def fmt_score(value: object) -> str:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return "n/a"
        return f"{float(value):.4f}"

    target_obj = cfg.objective.target
    target_str = fmt_score(target_obj) if target_obj is not None else "none"
    goal_str = "Higher is better" if direction == "max" else "Lower is better"

    click.echo(f"[iter {iteration}] {decision}")
    click.echo(f"  Metric      {obj}")
    click.echo(f"  Goal        {goal_str}")
    click.echo(f"  Target      {target_str}")
    click.echo(f"  Best so far {fmt_score(best_val)}")
    click.echo(f"  Candidate   {fmt_score(cand_val)}")

    trimmed_rationale = rationale.strip()
    if trimmed_rationale:
        click.echo("  Proposer notes")
        for line in trimmed_rationale.splitlines():
            click.echo(f"    {line.rstrip()}")


def _target_reached(cfg: AutoresearchRunConfig, best_result: dict[str, object]) -> bool:
    target = cfg.objective.target
    if target is None:
        return False
    value_obj = best_result.get("objective_value")
    if not isinstance(value_obj, (int, float)) or isinstance(value_obj, bool):
        return False
    value = float(value_obj)
    direction = objective_direction(cfg.objective.type)
    if direction == "max":
        return value >= target
    return value <= target


def _write_best(
    *,
    run_dir: Path,
    cfg: AutoresearchRunConfig,
    best_prompt: str,
    best_result: dict[str, object],
) -> None:
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    _write_text(best_dir / "scenario_prompt.md", best_prompt + "\n")
    _write_json(best_dir / "harness.json", _harness_json(cfg, scenario_prompt=best_prompt))
    _write_json(best_dir / "result.json", best_result)


def _create_run_dir(output_base_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    run_dir = output_base_dir / f"autoresearch_run_{ts}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dump_run_config_toml(cfg: AutoresearchRunConfig) -> str:
    """Serialize a run config to TOML for recording inside the output package."""

    def q(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    task_list = ", ".join(f'"{q(task_id)}"' for task_id in cfg.run.task_ids)
    environments_dir_line = (
        f'environments_dir = "{q(cfg.run.environments_dir)}"\n' if cfg.run.environments_dir else ""
    )
    objective_block = (
        "[objective]\n"
        f'type = "{cfg.objective.type}"\n'
        + (f"target = {cfg.objective.target}\n" if cfg.objective.target is not None else "")
        + "\n"
    )

    parts = [
        "[run]\n",
        f'env = "{q(cfg.run.env)}"\n',
    ]
    if environments_dir_line:
        parts.append(environments_dir_line)
    parts.extend(
        [
            f'tasks_dir = "{q(cfg.run.tasks_dir)}"\n',
            f"task_ids = [{task_list}]\n",
            f'runtime = "{cfg.run.runtime}"\n',
            f"rollout_count = {cfg.run.rollout_count}\n",
            f"max_parallel = {cfg.run.max_parallel}\n",
            f"max_steps = {cfg.run.max_steps}\n",
            f"agent_timeout_seconds = {cfg.run.agent_timeout_seconds}\n",
            f"no_seed = {'true' if cfg.run.no_seed else 'false'}\n",
            "\n",
            "[agent]\n",
            f'model = "{q(cfg.agent.model)}"\n',
            f'provider = "{q(cfg.agent.provider)}"\n',
            f'api_key_env = "{q(cfg.agent.api_key_env)}"\n',
            f'base_url_env = "{q(cfg.agent.base_url_env or "")}"\n',
            "\n",
            "[proposer]\n",
            f'model = "{q(cfg.proposer.model)}"\n',
            f'provider = "{q(cfg.proposer.provider)}"\n',
            f'api_key_env = "{q(cfg.proposer.api_key_env)}"\n',
            f'base_url_env = "{q(cfg.proposer.base_url_env or "")}"\n',
            "\n",
            "[verifier]\n",
            f'model = "{q(cfg.verifier.model)}"\n',
            f'provider = "{q(cfg.verifier.provider)}"\n',
            f'api_key_env = "{q(cfg.verifier.api_key_env)}"\n',
            f'base_url_env = "{q(cfg.verifier.base_url_env or "")}"\n',
            "\n",
            objective_block,
            "[budget]\n",
            f"max_iterations = {cfg.budget.max_iterations}\n",
            f"max_minutes = {cfg.budget.max_minutes}\n",
            f"no_improvement_window = {cfg.budget.no_improvement_window}\n",
        ]
    )
    return "".join(parts)


def _apply_verifier_overrides(global_cfg: GlobalConfig, cfg: AutoresearchRunConfig) -> GlobalConfig:
    return global_cfg.model_copy(
        update={
            "verifier_model": cfg.verifier.model,
            "verifier_provider": cfg.verifier.provider,
            "verifier_api_key": cfg.verifier.resolve_api_key(),
            "verifier_base_url": cfg.verifier.resolve_base_url(),
        }
    )


def _extract_required_headings(prompt: str) -> list[str]:
    headings: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        headings.append(stripped)
    return headings


def _harness_json(
    cfg: AutoresearchRunConfig,
    *,
    scenario_prompt: str,
    change_type: str | None = None,
    rationale: str | None = None,
) -> dict[str, object]:
    return {
        "version": "0.1",
        "env": cfg.run.env,
        "tasks_dir": cfg.run.tasks_dir,
        "task_ids": list(cfg.run.task_ids),
        "runtime": cfg.run.runtime,
        "rollout_count": cfg.run.rollout_count,
        "agent": {
            "model": cfg.agent.model,
            "provider": cfg.agent.provider,
        },
        "objective": {
            "type": cfg.objective.type,
            "target": cfg.objective.target,
        },
        "scenario_prompt": scenario_prompt,
        "change_type": change_type,
        "rationale": rationale,
    }


def _resolve_tasks_dir_path(tasks_dir: str) -> Path:
    path = Path(tasks_dir).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _bundle_has_rubric_files(*, bundle_dir: Path, task_ids: list[str]) -> bool:
    rubrics_dir = bundle_dir / "rubrics"
    if not rubrics_dir.is_dir():
        return False

    for task_id in task_ids:
        resolved_task_id = _resolve_task_id(bundle_dir=bundle_dir, task_id=task_id)
        if find_rubric_file(bundle_dir=bundle_dir, task_id=resolved_task_id) is not None:
            return True
    return False


def _resolve_task_id(*, bundle_dir: Path, task_id: str) -> str:
    task_id = task_id.strip()
    if not task_id:
        return ""

    tasks_dir = bundle_dir / "tasks"
    if not tasks_dir.is_dir():
        return task_id

    task_data, _profiles, _task_file = load_local_task(bundle_dir, task_id)
    meta_obj = task_data.get("meta")
    meta = meta_obj if isinstance(meta_obj, dict) else {}
    resolved = meta.get("task_id", task_id)
    resolved_str = str(resolved).strip()
    return resolved_str or task_id
