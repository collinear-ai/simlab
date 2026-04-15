# ruff: noqa: ANN401
"""CLI entry point for ``simlab eval``."""

from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import click

from simlab.config import get_global_config_from_ctx
from simlab.config import resolve_agent_api_key
from simlab.evaluation import EvaluationError
from simlab.evaluation import build_report
from simlab.evaluation import parse_iso8601_timestamp
from simlab.evaluation import summarize_rollouts


@click.command("eval")
@click.argument(
    "path",
    required=False,
    default=Path("output"),
    type=click.Path(path_type=Path, exists=False, file_okay=False),
)
@click.option(
    "--compare",
    "compare_path",
    default=None,
    type=click.Path(path_type=Path, exists=False, file_okay=False),
    help="Compare PATH against another rollout directory.",
)
@click.option(
    "--task",
    "task_ids",
    multiple=True,
    help="Only include rollouts for the given task id (can repeat).",
)
@click.option(
    "--model",
    "model_names",
    multiple=True,
    help="Only include rollouts for the given model name (can repeat).",
)
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON output.")
@click.option(
    "--report",
    "report_path",
    type=click.Path(path_type=Path, exists=False, file_okay=True, dir_okay=False),
    flag_value="eval-report.md",
    is_flag=False,
    help="Write a Markdown report to PATH (default: ./eval-report.md).",
)
def eval_command(
    path: Path,
    compare_path: Path | None,
    task_ids: tuple[str, ...],
    model_names: tuple[str, ...],
    json_output: bool,
    report_path: Path | None,
) -> None:
    """Analyze local rollout artifacts under PATH. Defaults to ./output."""
    resolved_report_path = report_path.expanduser() if report_path is not None else None
    warn_report_overwrite(resolved_report_path)

    try:
        report = build_report(
            path,
            compare_path=compare_path,
            task_ids=task_ids,
            model_names=model_names,
        )
    except EvaluationError as exc:
        raise click.ClickException(str(exc)) from exc

    if json_output:
        click.echo(json.dumps(report, indent=2))
    else:
        if report["mode"] == "single_rollout":
            print_single_rollout(report["rollout"])
        elif report["mode"] == "compare":
            print_compare_report(report)
        else:
            print_multi_rollout(report["path"], report["summary"])

        print_warnings(report.get("warnings", []))

    report_written = False
    if resolved_report_path is not None:
        try:
            write_markdown_report(report, resolved_report_path)
            report_written = True
        except Exception as exc:  # pragma: no cover - click renders exception text
            raise click.ClickException(f"Failed to write Markdown report: {exc}") from exc

    if report_written:
        if json_output:
            click.echo(f"Evaluation report written to {resolved_report_path}", err=True)
            return
        click.echo("")
        click.echo(f"Evaluation report written to {resolved_report_path}")


def write_markdown_report(report: dict[str, Any], report_path: Path) -> None:
    """Write the Markdown evaluation report to disk."""
    report_text = render_markdown_report(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")


def warn_report_overwrite(report_path: Path | None) -> None:
    """Warn early when the report path already exists."""
    if report_path is None or not report_path.exists():
        return

    click.echo(
        click.style(
            f"WARNING: report file already exists and will be overwritten: {report_path}",
            fg="yellow",
            bold=True,
        ),
        err=True,
    )
    click.echo(
        click.style("Press Ctrl+C now to cancel.", fg="yellow", bold=True),
        err=True,
    )
    click.echo("", err=True)


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the evaluation report as shareable Markdown."""
    generated_at = datetime.now(timezone.utc)

    if report.get("mode") == "multi_rollout":
        summary = report["summary"]
        rollouts = report.get("rollouts", [])
        rollout_path = report.get("path")
    elif report.get("mode") == "single_rollout":
        rollout = report["rollout"]
        summary = summarize_rollouts([rollout])
        rollouts = [rollout]
        rollout_path = report.get("path") or rollout.get("path")
    elif report.get("mode") == "compare":
        return render_markdown_compare_report(report, generated_at)
    else:
        raise ValueError(
            "Markdown report generation currently supports single, multi, and compare eval only."
        )

    run_analysis = generate_run_analysis_markdown(
        summary,
        rollouts=rollouts,
        rollout_path=str(rollout_path or ""),
    )

    lines: list[str] = ["# Evaluation Report", ""]

    rollout_count = int(summary.get("rollout_count") or 0)
    task_count = int(summary.get("task_count") or 0)
    models = summary.get("models") or []
    lines.append(
        f"**{rollout_count} rollouts** | {task_count} tasks | {format_model_summary(models)}"
    )

    date_range = format_date_range(rollouts)
    if date_range:
        lines.append(f"**Date range:** {date_range}")

    passed_count = int(summary.get("passed_count") or 0)
    pass_rate = summary.get("pass_rate")
    lines.append(f"**Pass rate:** {format_percent(pass_rate)} ({passed_count}/{rollout_count})")

    reward_distribution = summary.get("reward_distribution") or {}
    lines.append(
        "**Reward:** "
        f"min={format_compact_number(reward_distribution.get('min'))} "
        f"· p50={format_compact_number(reward_distribution.get('p50'))} "
        f"· max={format_compact_number(reward_distribution.get('max'))}"
    )

    aggregate_metrics = summary.get("aggregate_metrics") or {}
    lines.append(
        "**Total cost:** "
        f"{format_money(aggregate_metrics.get('total_cost_usd'))} | "
        f"Avg per rollout: {format_money(aggregate_metrics.get('avg_cost_usd'))}"
    )
    lines.append(
        "**Avg duration:** "
        f"{format_duration(aggregate_metrics.get('avg_duration_seconds'))} | "
        "**Avg tokens:** "
        f"{format_tokens(aggregate_metrics.get('avg_prompt_tokens'))} in / "
        f"{format_tokens(aggregate_metrics.get('avg_completion_tokens'))} out"
    )
    lines.append("")

    results_by_task = summary.get("results_by_task") or []
    lines.extend(render_results_by_task_section(results_by_task))
    lines.append("")

    capability_analysis = summary.get("capability_analysis") or []
    lines.extend(render_capability_analysis_section(capability_analysis, rollout_count))
    lines.append("")

    if run_analysis:
        lines.append("## Run Analysis")
        lines.append("")
        lines.append(run_analysis.strip())
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by SimLab on {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}*")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown_compare_report(report: dict[str, Any], generated_at: datetime) -> str:
    """Render a compare-mode evaluation report in Markdown."""
    if report.get("compare_kind") == "rollout":
        return render_markdown_rollout_compare_report(report, generated_at)

    return render_markdown_dataset_compare_report(report, generated_at)


def render_markdown_rollout_compare_report(report: dict[str, Any], generated_at: datetime) -> str:
    """Render a rollout-level compare report in Markdown."""
    left_rollout = report["left"]["rollout"]
    right_rollout = report["right"]["rollout"]
    comparison = report.get("rollout_comparison") or {}

    lines: list[str] = ["# Comparison Report", ""]

    if left_rollout.get("task_id") == right_rollout.get("task_id"):
        lines.append(f"**Task:** {left_rollout.get('task_id')}")
    else:
        lines.append(f"**A task:** {left_rollout.get('task_id')}")
        lines.append(f"**B task:** {right_rollout.get('task_id')}")

    lines.append(f"**A path:** {left_rollout.get('path')}")
    lines.append(f"**B path:** {right_rollout.get('path')}")
    lines.append("")

    lines.extend(render_rollout_compare_summary_section(comparison.get("overview") or []))
    lines.append("")

    lines.extend(
        render_rollout_compare_tool_flow_section(
            comparison.get("left_errors") or [],
            comparison.get("right_errors") or [],
        )
    )
    lines.append("")

    lines.extend(render_rollout_compare_criteria_section(comparison.get("criteria") or []))
    lines.append("")

    lines.extend(
        render_rollout_compare_reward_model_dimensions_section(
            comparison.get("reward_model_dimensions") or []
        )
    )
    lines.append("")

    lines.extend(
        render_rollout_compare_tool_sequence_section(comparison.get("tool_sequence") or [])
    )
    lines.append("")

    lines.extend(
        render_rollout_compare_error_calls_section(
            "A Error Calls",
            comparison.get("left_errors") or [],
            empty_message="_No tool errors on A._",
        )
    )
    lines.append("")

    lines.extend(
        render_rollout_compare_error_calls_section(
            "B Error Calls",
            comparison.get("right_errors") or [],
            empty_message="_No tool errors on B._",
        )
    )
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by SimLab on {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}*")
    return "\n".join(lines).rstrip() + "\n"


def render_rollout_compare_summary_section(overview_rows: list[dict[str, Any]]) -> list[str]:
    """Render the rollout summary section as Markdown."""
    headers = ["Metric", "A", "B", "A-B", "Better"]
    rows: list[list[str]] = []
    for row in overview_rows:
        metric = row.get("metric")
        left_value = row.get("left_value")
        right_value = row.get("right_value")
        rows.append(
            [
                rollout_metric_label(metric),
                format_rollout_compare_value(metric, left_value),
                format_rollout_compare_value(metric, right_value),
                format_rollout_compare_delta(metric, row.get("delta")),
                better_side(metric, left_value, right_value),
            ]
        )

    section = ["## Summary", ""]
    if not rows:
        section.append("_No rollout comparison data recorded._")
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def render_rollout_compare_tool_flow_section(
    left_errors: list[dict[str, Any]],
    right_errors: list[dict[str, Any]],
) -> list[str]:
    """Render the tool flow summary section as Markdown."""
    section = ["## Tool Flow", ""]
    section.append(
        render_markdown_table(
            ["Metric", "Value"],
            [
                ["A tool errors", str(len(left_errors))],
                ["B tool errors", str(len(right_errors))],
            ],
        )
    )
    return section


def render_rollout_compare_criteria_section(criteria_rows: list[dict[str, Any]]) -> list[str]:
    """Render the programmatic verifier comparison section as Markdown."""
    headers = ["Criterion", "A", "B", "Notes"]
    rows: list[list[str]] = [
        [
            str(row.get("criterion") or ""),
            format_pass_fail(row.get("left_passed")),
            format_pass_fail(row.get("right_passed")),
            comparison_notes(row.get("left_detail"), row.get("right_detail")),
        ]
        for row in criteria_rows
    ]

    section = ["## Programmatic Verifier Comparison", ""]
    if not rows:
        section.append("_No programmatic verifier results recorded._")
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def render_rollout_compare_reward_model_dimensions_section(
    dimension_rows: list[dict[str, Any]],
) -> list[str]:
    """Render the reward model dimension comparison section as Markdown."""
    headers = ["Dimension", "A", "B", "A-B", "Better"]
    rows: list[list[str]] = [
        [
            str(row.get("dimension") or ""),
            format_number(row.get("left_score")),
            format_number(row.get("right_score")),
            format_signed(row.get("delta_score")),
            better_side("reward_model_score", row.get("left_score"), row.get("right_score")),
        ]
        for row in dimension_rows
    ]

    section = ["## Reward Model Dimension Comparison", ""]
    if not rows:
        section.append("_No reward model dimensions recorded._")
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def render_rollout_compare_tool_sequence_section(
    tool_sequence_rows: list[dict[str, Any]],
) -> list[str]:
    """Render the tool sequence comparison section as Markdown."""
    headers = ["#", "A tool", "B tool", "Match", "A status", "B status"]
    rows: list[list[str]] = [
        [
            str(row.get("step") or ""),
            str(row.get("left_tool") or ""),
            str(row.get("right_tool") or ""),
            "yes" if row.get("same_tool") else "no",
            "error" if row.get("left_error") else ("ok" if row.get("left_tool") else ""),
            "error" if row.get("right_error") else ("ok" if row.get("right_tool") else ""),
        ]
        for row in tool_sequence_rows[:15]
    ]

    section = ["## Tool Sequence", ""]
    if not rows:
        section.append("_No tool calls recorded._")
        return section
    section.append(render_markdown_table(headers, rows))
    if len(tool_sequence_rows) > 15:
        section.append("")
        section.append("_Showing first 15 tool calls._")
    return section


def render_rollout_compare_error_calls_section(
    title: str,
    error_rows: list[dict[str, Any]],
    *,
    empty_message: str,
) -> list[str]:
    """Render a per-side tool error call table as Markdown."""
    headers = ["#", "Tool", "Summary"]
    rows: list[list[str]] = [
        [
            str(row.get("index") or ""),
            str(row.get("tool") or ""),
            compact(row.get("summary"), 200),
        ]
        for row in error_rows
    ]

    section = [f"## {title}", ""]
    if not rows:
        section.append(empty_message)
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def render_markdown_dataset_compare_report(report: dict[str, Any], generated_at: datetime) -> str:
    """Render a dataset-level compare report in Markdown."""
    left = report["left"]
    right = report["right"]
    comparison = report.get("comparison") or {}

    lines: list[str] = ["# Comparison Report", ""]
    lines.append(f"**A:** {Path(str(left.get('path') or '')).name}")
    lines.append(f"**B:** {Path(str(right.get('path') or '')).name}")
    lines.append(f"**A path:** {left.get('path')}")
    lines.append(f"**B path:** {right.get('path')}")
    if report.get("compare_kind"):
        lines.append(f"**Compare kind:** {report.get('compare_kind')}")
    lines.append("")

    lines.extend(render_compare_overview_section(comparison.get("overview") or []))
    lines.append("")
    lines.extend(render_compare_task_section(comparison.get("results_by_task") or []))
    lines.append("")
    lines.extend(render_compare_model_section(comparison.get("model_comparison") or []))
    lines.append("")
    lines.extend(render_compare_system_section(comparison.get("system_failure_rates") or []))
    lines.append("")
    lines.extend(render_compare_capability_section(comparison.get("capability_analysis") or []))
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by SimLab on {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}*")
    return "\n".join(lines).rstrip() + "\n"


def render_compare_overview_section(overview_rows: list[dict[str, Any]]) -> list[str]:
    """Render a compare overview section as Markdown."""
    headers = ["Metric", "A", "B", "A-B", "Better"]
    rows: list[list[str]] = []
    for row in overview_rows:
        metric = row.get("metric")
        left_value = row.get("left_value")
        right_value = row.get("right_value")
        rows.append(
            [
                overview_metric_label(metric),
                format_overview_value(metric, left_value),
                format_overview_value(metric, right_value),
                format_overview_delta(metric, row.get("delta")),
                better_side(metric, left_value, right_value),
            ]
        )

    section = ["## Overview", ""]
    if not rows:
        section.append("_No comparison data recorded._")
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def render_compare_task_section(task_rows: list[dict[str, Any]]) -> list[str]:
    """Render the task comparison section as Markdown."""
    headers = [
        "Task",
        "A reward",
        "B reward",
        "A-B reward",
        "A reward model",
        "B reward model",
        "A-B reward model",
        "Better",
    ]
    rows: list[list[str]] = [
        [
            str(row.get("task_id") or ""),
            format_percent(row.get("left_pass_rate")),
            format_percent(row.get("right_pass_rate")),
            format_signed_percent(row.get("delta_pass_rate")),
            format_number(row.get("left_mean_reward_model_score")),
            format_number(row.get("right_mean_reward_model_score")),
            format_signed(row.get("delta_mean_reward_model_score")),
            better_side("pass_rate", row.get("left_pass_rate"), row.get("right_pass_rate")),
        ]
        for row in task_rows
    ]

    section = ["## Task Comparison", ""]
    if not rows:
        section.append("_No task comparison data recorded._")
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def render_compare_model_section(model_rows: list[dict[str, Any]]) -> list[str]:
    """Render the model comparison section as Markdown."""
    headers = [
        "Model",
        "A reward",
        "B reward",
        "A-B reward",
        "A reward model",
        "B reward model",
        "A-B reward model",
        "Better",
    ]
    rows: list[list[str]] = [
        [
            str(row.get("model") or ""),
            format_percent(row.get("left_pass_rate")),
            format_percent(row.get("right_pass_rate")),
            format_signed_percent(row.get("delta_pass_rate")),
            format_number(row.get("left_mean_reward_model_score")),
            format_number(row.get("right_mean_reward_model_score")),
            format_signed(row.get("delta_mean_reward_model_score")),
            better_side("pass_rate", row.get("left_pass_rate"), row.get("right_pass_rate")),
        ]
        for row in model_rows
    ]

    section = ["## Model Comparison", ""]
    if not rows:
        section.append("_No model comparison data recorded._")
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def render_compare_system_section(system_rows: list[dict[str, Any]]) -> list[str]:
    """Render the system failure rate comparison section as Markdown."""
    headers = ["System", "A fail", "B fail", "Delta fail", "Better"]
    rows: list[list[str]] = [
        [
            str(row.get("system") or ""),
            format_percent(row.get("left_fail_rate")),
            format_percent(row.get("right_fail_rate")),
            format_signed_percent(row.get("delta_fail_rate")),
            better_side("fail_rate", row.get("left_fail_rate"), row.get("right_fail_rate")),
        ]
        for row in system_rows
    ]

    section = ["## System Failure Rate Comparison", ""]
    if not rows:
        section.append("_No system comparison data recorded._")
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def render_compare_capability_section(capability_rows: list[dict[str, Any]]) -> list[str]:
    """Render the capability comparison section as Markdown."""
    headers = [
        "Capability",
        "A pass",
        "B pass",
        "A-B pass",
        "A results",
        "B results",
        "Better",
    ]
    rows: list[list[str]] = [
        [
            str(row.get("capability") or ""),
            format_percent(row.get("left_pass_rate")),
            format_percent(row.get("right_pass_rate")),
            format_signed_percent(row.get("delta_pass_rate")),
            format_compare_value(row.get("left_total_count")),
            format_compare_value(row.get("right_total_count")),
            better_side("pass_rate", row.get("left_pass_rate"), row.get("right_pass_rate")),
        ]
        for row in capability_rows
    ]

    section = ["## Capability Comparison", ""]
    if not rows:
        section.append("_No capability comparison data recorded._")
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def format_date_range(rollouts: list[dict[str, Any]]) -> str | None:
    """Return the date range spanned by the rollouts when timestamps exist."""
    timestamps = [
        parsed
        for rollout in rollouts
        if (parsed := parse_iso8601_timestamp(str(rollout.get("created_at") or ""))) is not None
    ]
    if not timestamps:
        return None
    start = min(timestamps).date().isoformat()
    end = max(timestamps).date().isoformat()
    if start == end:
        return start
    return f"{start} to {end}"


def format_compact_number(value: Any) -> str:
    """Format a float without trailing zeros for Markdown summaries."""
    if not isinstance(value, (int, float)):
        return "n/a"
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text or "0"


def render_results_by_task_section(results_by_task: list[dict[str, Any]]) -> list[str]:
    """Render the results-by-task Markdown section."""
    headers = ["Task", "Pass Rate", "Avg Cost", "Avg Duration"]
    rows: list[list[str]] = []
    for row in results_by_task:
        task_id = str(row.get("task_id") or "")
        passed_count = int(row.get("passed_count") or 0)
        rollout_count = int(row.get("rollout_count") or 0)
        pass_rate = row.get("pass_rate")
        rows.append(
            [
                task_id,
                f"{format_percent(pass_rate)} ({passed_count}/{rollout_count})",
                format_money(row.get("avg_cost_usd")),
                format_duration(row.get("avg_duration_seconds")),
            ]
        )

    section = ["## Results by Task", ""]
    if not rows:
        section.append("_No task results found._")
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def render_capability_analysis_section(
    capability_rows: list[dict[str, Any]],
    rollout_count: int,
) -> list[str]:
    """Render the capability analysis Markdown section."""
    headers = ["Capability", "Pass Rate", "Details"]
    rows: list[list[str]] = []
    for row in capability_rows:
        capability = str(row.get("capability") or "")
        pass_count = int(row.get("pass_count") or 0)
        total_count = int(row.get("total_count") or 0)
        pass_rate = row.get("pass_rate")
        rows.append(
            [
                capability,
                f"{format_percent(pass_rate)} ({pass_count}/{total_count})",
                format_capability_details(row, rollout_count),
            ]
        )

    section = ["## Capability Analysis", ""]
    if not rows:
        section.append("_No capability analysis found._")
        return section
    section.append(render_markdown_table(headers, rows))
    return section


def format_capability_details(row: dict[str, Any], rollout_count: int) -> str:
    """Describe cofailure and attempt patterns for one capability."""
    attempt_signal = row.get("attempt_signal")
    raw_patterns = row.get("cofailure_patterns")
    patterns = raw_patterns if isinstance(raw_patterns, list) else []
    top_pattern = patterns[0] if patterns else None

    details: list[str] = []
    if isinstance(top_pattern, dict):
        fail_count = int(top_pattern.get("fail_count") or 0)
        if fail_count:
            details.append(f"Fails together in {fail_count}/{rollout_count} runs")
    if isinstance(attempt_signal, str) and attempt_signal.strip():
        details.append(attempt_signal.strip())
    return " | ".join(details) if details else ""


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-compatible Markdown table."""
    lines = [
        "| " + " | ".join(escape_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |" for row in rows
    )
    return "\n".join(lines)


def escape_markdown_cell(value: Any) -> str:
    """Escape Markdown table cell text."""
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    return text.replace("|", "\\|")


def generate_run_analysis_markdown(
    summary: dict[str, Any],
    *,
    rollouts: list[dict[str, Any]],
    rollout_path: str,
) -> str | None:
    """Generate a short Markdown run analysis using an LLM when an API key is available."""
    global_cfg = get_global_config_from_ctx(click.get_current_context(silent=True))

    providers = {str(rollout.get("provider")) for rollout in rollouts if rollout.get("provider")}
    provider = (
        next(iter(providers))
        if len(providers) == 1
        else ((global_cfg.agent_provider or "openai").strip() or "openai")
    )
    provider = provider.strip() or "openai"

    api_key = resolve_agent_api_key(provider=provider, config=global_cfg)
    if not api_key:
        return None

    analysis_model: str | None = None
    raw_models = summary.get("models")
    models = (
        [model.strip() for model in raw_models if isinstance(model, str) and model.strip()]
        if isinstance(raw_models, list)
        else []
    )
    if len(models) == 1:
        analysis_model = models[0]
    if not analysis_model:
        analysis_model = (global_cfg.agent_model or "gpt-5.2").strip() or "gpt-5.2"

    base_url = (global_cfg.agent_base_url or "").strip() or None

    raw_results_by_task = summary.get("results_by_task")
    results_by_task: list[Any] = (
        raw_results_by_task if isinstance(raw_results_by_task, list) else []
    )
    raw_capability_analysis = summary.get("capability_analysis")
    capability_analysis: list[Any] = (
        raw_capability_analysis if isinstance(raw_capability_analysis, list) else []
    )
    raw_cofailure_patterns = summary.get("cofailure_patterns")
    cofailure_patterns: list[Any] = (
        raw_cofailure_patterns if isinstance(raw_cofailure_patterns, list) else []
    )

    analysis_input = {
        "path": rollout_path,
        "rollout_count": summary.get("rollout_count"),
        "task_count": summary.get("task_count"),
        "models": summary.get("models"),
        "pass_rate": summary.get("pass_rate"),
        "passed_count": summary.get("passed_count"),
        "reward_distribution": summary.get("reward_distribution"),
        "aggregate_metrics": summary.get("aggregate_metrics"),
        "results_by_task": results_by_task[:25],
        "capability_analysis": capability_analysis[:25],
        "cofailure_patterns": cofailure_patterns[:10],
    }

    system_prompt = (
        "You are SimLab. Write a short analysis of an evaluation run "
        "based only on the provided JSON. "
        "Focus on the biggest strengths, the biggest gaps, and what to change next. "
        "Output 2 to 4 short paragraphs plus one final line that starts with "
        "'**Recommendation:**'. "
        "Do not include headings, bullets, or code fences."
    )

    litellm_model = analysis_model
    if provider and not analysis_model.startswith(f"{provider}/"):
        litellm_model = f"{provider}/{analysis_model}"

    try:
        import litellm  # noqa: PLC0415

        analysis_payload = json.dumps(analysis_input, indent=2, ensure_ascii=False)
        response = litellm.completion(
            model=litellm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_payload},
            ],
            api_key=api_key,
            base_url=base_url,
        )
        content = response.choices[0].message.content
    except Exception as exc:
        click.echo(
            f"[simlab] Run analysis skipped due to LLM error ({type(exc).__name__}).",
            err=True,
        )
        return None

    text = (content or "").strip()
    return text or None


def print_single_rollout(rollout: dict[str, Any]) -> None:
    """Render the human-readable report for one rollout."""
    criteria = rollout.get("criteria", [])
    tool_calls = rollout.get("tool_calls", [])
    metrics = rollout.get("metrics", {})
    reward_model = rollout.get("reward_model") or {}
    passed_count = sum(1 for criterion in criteria if criterion.get("passed"))
    total_count = len(criteria)
    status = single_rollout_status(rollout, passed_count, total_count)
    tool_error_count = sum(1 for tool_call in tool_calls if tool_call.get("is_error"))

    click.echo("Rollout Detail")
    click.echo(f"Run: {Path(str(rollout.get('path') or '')).name}")
    click.echo(f"Task: {rollout.get('task_id')}")
    click.echo(f"Path: {rollout.get('path')}")
    model = rollout.get("model")
    provider = rollout.get("provider")
    if model and provider:
        click.echo(f"Model: {model} ({provider})")
    elif model:
        click.echo(f"Model: {model}")
    if rollout.get("created_at"):
        click.echo(f"Date: {format_timestamp(rollout['created_at'])}")
    if rollout.get("max_steps") is not None:
        click.echo(f"Steps: {rollout.get('steps_taken', 0)}/{rollout.get('max_steps')}")
    else:
        click.echo(f"Steps: {rollout.get('steps_taken', 0)}")
    click.echo(f"Status: {status}")

    print_metric_table(
        "Summary",
        [
            ("Termination", str(rollout.get("termination_reason") or "n/a")),
            ("Reward", format_number(rollout.get("reward"))),
            ("Programmatic verifier score", format_number(pass_rate(passed_count, total_count))),
            ("Reward model score", format_number(reward_model.get("score"))),
            (
                "Programmatic verifier results",
                f"{passed_count}/{total_count}" if total_count else "n/a",
            ),
            ("Tool calls", str(len(tool_calls))),
            ("Tool errors", str(tool_error_count)),
            ("Unique tools", str(len({tool_label(tool_call) for tool_call in tool_calls}))),
            ("Cost", format_money(metrics.get("estimated_cost_usd"))),
            ("Duration", format_duration(metrics.get("duration_seconds"))),
            (
                "Tokens",
                f"{format_tokens(metrics.get('prompt_tokens'))} in / "
                f"{format_tokens(metrics.get('completion_tokens'))} out",
            ),
        ],
    )

    print_table(
        f"Programmatic Verifier Results ({total_count})",
        ["Status", "Criterion", "Capability", "Detail"],
        [
            [
                "PASS" if criterion.get("passed") else "FAIL",
                str(criterion.get("name") or "unknown"),
                str(criterion.get("capability") or ""),
                compact(criterion.get("detail"), 56),
            ]
            for criterion in criteria
        ],
        empty_message="  No programmatic verifier results found in reward.json.",
    )

    if reward_model:
        print_metric_table(
            "Reward Model",
            [
                ("Source", str(reward_model.get("source") or "n/a")),
                ("Score", format_number(reward_model.get("score"))),
                ("Confidence", format_number(reward_model.get("confidence"))),
                ("Verdict", str(reward_model.get("verdict") or "n/a")),
            ],
        )
        print_table(
            "Reward Model Dimensions",
            ["Dimension", "Score", "Reason"],
            [
                [
                    compact(row.get("dimension"), 32),
                    format_number(row.get("score")),
                    compact(row.get("reason"), 48),
                ]
                for row in reward_model.get("dimension_scores", [])
            ],
            empty_message="  No reward model dimensions recorded.",
        )
        print_table(
            "Reward Model Failed Criteria",
            ["Criterion"],
            [[str(name)] for name in reward_model.get("failed_criteria", [])],
            empty_message="  No failed reward model criteria recorded.",
        )

    print_table(
        f"Tool Calls ({len(tool_calls)})",
        ["#", "Status", "Tool", "Args", "Summary"],
        [
            [
                str(tool_call.get("index") or ""),
                "error" if tool_call.get("is_error") else "ok",
                tool_label(tool_call),
                compact(compact_json(tool_call.get("parameters")), 28),
                compact(tool_call.get("summary") or "No summary available", 64),
            ]
            for tool_call in tool_calls
        ],
        empty_message="  No tool calls recorded.",
    )


def print_multi_rollout(path: str, summary: dict[str, Any]) -> None:
    """Render the human-readable report for a rollout directory."""
    click.echo("Evaluation Report")
    click.echo(f"Path: {path}")
    click.echo(
        f"{summary.get('rollout_count', 0)} rollouts | "
        f"{summary.get('task_count', 0)} tasks | "
        f"{format_model_summary(summary.get('models') or [])}"
    )

    overview = summary.get("overview", {})
    score_summary = summary.get("score_summary", {})
    reward_model_gap = summary.get("reward_model_gap", {})
    tool_error_taxonomy = summary.get("tool_error_taxonomy", {})
    call_patterns = summary.get("call_sequence_patterns", {})
    task_difficulty = summary.get("task_difficulty", {})

    print_metric_table(
        "High-Level Overview",
        [
            ("Total rollouts", str(overview.get("total_rollouts", 0))),
            ("Unique tasks", str(overview.get("unique_tasks", 0))),
            ("Total tool calls", str(overview.get("total_tool_calls", 0))),
            (
                "Total tool errors",
                f"{overview.get('total_tool_errors', 0)} "
                f"({format_percent(overview.get('tool_error_rate'))})",
            ),
            (
                "Rollouts with at least 1 error",
                f"{overview.get('rollouts_with_errors', 0)} "
                f"({format_percent(overview.get('rollouts_with_errors_rate'))})",
            ),
            ("Total recorded steps", str(overview.get("total_steps") or 0)),
            (
                "Rollouts with recorded steps",
                f"{overview.get('rollouts_with_steps', 0)} "
                f"({format_percent(overview.get('rollouts_with_steps_rate'))})",
            ),
            (
                "Average recorded steps per rollout",
                format_number(overview.get("avg_steps_per_rollout")),
            ),
            ("Median steps per rollout", format_number(overview.get("median_steps_per_rollout"))),
            ("Average duration", format_duration(overview.get("avg_duration_seconds"))),
            (
                "Termination reasons",
                ", ".join(
                    f"{row.get('reason')} {row.get('count')} ({format_percent(row.get('rate'))})"
                    for row in overview.get("termination_reasons", [])
                )
                or "n/a",
            ),
        ],
    )

    print_metric_table(
        "Score Summary",
        [
            (
                "Programmatic verifier all-checks pass rate",
                format_percent(score_summary.get("module_all_checks_pass_rate")),
            ),
            (
                "Programmatic verifier check pass rate",
                format_percent(score_summary.get("individual_module_check_pass_rate")),
            ),
            (
                "Mean programmatic verifier score",
                format_number(score_summary.get("mean_composite_score")),
            ),
            (
                "Mean reward model score",
                format_number(score_summary.get("mean_reward_model_score")),
            ),
            (
                "Reward model score variance",
                format_number(score_summary.get("reward_model_score_variance")),
            ),
            (
                "Reward model score stdev",
                format_number(score_summary.get("reward_model_score_stdev")),
            ),
            (
                "Rollouts scoring >0.8 with reward model",
                format_percent(score_summary.get("reward_model_rollouts_above_point_eight_rate")),
            ),
            ("Reward earned rate", format_percent(score_summary.get("reward_earned_rate"))),
        ],
    )

    print_metric_table(
        "Programmatic Verifier vs Reward Model Gap",
        [
            (
                "Programmatic verifier fails and reward model >0.8",
                f"{reward_model_gap.get('module_fails_reward_model_above_point_eight_count', 0)} "
                f"({format_percent(reward_model_gap.get('module_fails_reward_model_above_point_eight_rate'))})",
            ),
            (
                "Programmatic verifier passes and reward model <0.5",
                f"{reward_model_gap.get('module_passes_reward_model_below_point_five_count', 0)} "
                f"({format_percent(reward_model_gap.get('module_passes_reward_model_below_point_five_rate'))})",
            ),
        ],
    )

    print_table(
        "Programmatic Verifier Pass Rate vs Reward Model Score",
        ["Programmatic verifier pass rate", "N", "Mean reward model score"],
        [
            [
                str(row.get("module_pass_rate") or ""),
                str(row.get("rollout_count") or 0),
                format_number(row.get("mean_reward_model_score")),
            ]
            for row in reward_model_gap.get("module_pass_rate_bins", [])
        ],
        empty_message="  No programmatic-verifier-vs-reward-model data recorded.",
    )

    print_table(
        "Programmatic Verifier Failure Rates by System",
        ["System", "Passed", "Failed", "Fail rate"],
        [
            [
                str(row.get("system") or ""),
                str(row.get("passed_count") or 0),
                str(row.get("failed_count") or 0),
                format_percent(row.get("fail_rate")),
            ]
            for row in summary.get("system_failure_rates", [])
        ],
        empty_message="  No system-level programmatic verifier data recorded.",
    )

    print_table(
        "Programmatic Verifier Checks That Never Pass",
        ["Check", "Attempts", "Passes"],
        [
            [
                str(row.get("name") or ""),
                str(row.get("attempts") or 0),
                str(row.get("passed_count") or 0),
            ]
            for row in summary.get("never_passed_checks", [])[:30]
        ],
        empty_message="  Every observed programmatic verifier check passed at least once.",
    )

    print_table(
        "Lowest Pass Programmatic Verifier Checks",
        ["Check", "Attempts", "Passes", "Pass rate"],
        [
            [
                str(row.get("name") or ""),
                str(row.get("attempts") or 0),
                str(row.get("passed_count") or 0),
                format_percent(row.get("pass_rate")),
            ]
            for row in summary.get("lowest_pass_checks", [])[:20]
        ],
        empty_message="  No programmatic verifier results recorded.",
    )

    reward_model_dimensions = summary.get("reward_model_dimensions", {})
    print_table(
        "Lowest Reward Model Dimensions",
        ["Mean score", "Dimension", "Samples"],
        [
            [
                format_number(row.get("mean_score")),
                str(row.get("dimension") or ""),
                str(row.get("sample_count") or 0),
            ]
            for row in reward_model_dimensions.get("lowest", [])
        ],
        empty_message="  No reward model dimensions recorded.",
    )

    print_table(
        "Highest Reward Model Dimensions",
        ["Mean score", "Dimension", "Samples"],
        [
            [
                format_number(row.get("mean_score")),
                str(row.get("dimension") or ""),
                str(row.get("sample_count") or 0),
            ]
            for row in reward_model_dimensions.get("highest", [])
        ],
        empty_message="  No reward model dimensions recorded.",
    )

    print_table(
        "Reward Model Failed Criteria",
        ["Criterion", "Count"],
        [
            [str(row.get("criterion") or ""), str(row.get("count") or 0)]
            for row in summary.get("reward_model_failed_criteria", [])[:20]
        ],
        empty_message="  No failed reward model criteria recorded.",
    )

    print_table(
        "Tool Error Categories",
        ["Category", "Count", "Rate", "Description"],
        [
            [
                str(row.get("category") or ""),
                str(row.get("count") or 0),
                format_percent(row.get("rate")),
                str(row.get("description") or ""),
            ]
            for row in tool_error_taxonomy.get("categories", [])
        ],
        empty_message="  No tool errors recorded.",
    )

    print_table(
        "Most Error-Prone Tools",
        ["Tool", "Error rate", "Errors", "Calls"],
        [
            [
                str(row.get("tool") or ""),
                format_percent(row.get("error_rate")),
                str(row.get("error_count") or 0),
                str(row.get("call_count") or 0),
            ]
            for row in tool_error_taxonomy.get("most_error_prone_tools", [])[:15]
        ],
        empty_message="  No tool errors recorded.",
    )

    print_table(
        "Zero-Error Tools",
        ["Tool", "Calls"],
        [
            [str(row.get("tool") or ""), str(row.get("call_count") or 0)]
            for row in tool_error_taxonomy.get("zero_error_tools", [])[:15]
        ],
        empty_message="  No zero-error tools recorded.",
    )

    print_table(
        "Top Tool Error Messages",
        ["Count", "Message"],
        [
            [str(row.get("count") or 0), str(row.get("message") or "")]
            for row in tool_error_taxonomy.get("top_error_messages", [])[:10]
        ],
        empty_message="  No tool error messages recorded.",
    )

    print_table(
        "Most Common First Tool Calls",
        ["Tool", "Count", "Rate"],
        [
            [
                str(row.get("tool") or ""),
                str(row.get("count") or 0),
                format_percent(row.get("rate")),
            ]
            for row in call_patterns.get("first_calls", [])[:10]
        ],
        empty_message="  No tool call sequences recorded.",
    )

    print_table(
        "Most Common First Five Tool Patterns",
        ["Pattern", "Count", "Rate"],
        [
            [
                str(row.get("pattern") or ""),
                str(row.get("count") or 0),
                format_percent(row.get("rate")),
            ]
            for row in call_patterns.get("first_five_patterns", [])[:10]
        ],
        empty_message="  No tool call sequences recorded.",
    )

    print_table(
        "Results by Task",
        [
            "Task",
            "N",
            "Reward pass rate",
            "Mean reward model",
            "Mean programmatic",
            "Avg cost",
            "Avg duration",
        ],
        [
            [
                str(row.get("task_id") or ""),
                str(row.get("rollout_count") or 0),
                format_percent(row.get("pass_rate")),
                format_number(row.get("mean_reward_model_score")),
                format_number(row.get("mean_composite_score")),
                format_money(row.get("avg_cost_usd")),
                format_duration(row.get("avg_duration_seconds")),
            ]
            for row in summary.get("results_by_task", [])
        ],
        empty_message="  No task summary recorded.",
    )

    print_table(
        "Hardest Tasks",
        ["Task", "N", "Reward pass rate", "Mean reward model", "Mean programmatic"],
        [
            [
                str(row.get("task_id") or ""),
                str(row.get("rollout_count") or 0),
                format_percent(row.get("pass_rate")),
                format_number(row.get("mean_reward_model_score")),
                format_number(row.get("mean_composite_score")),
            ]
            for row in task_difficulty.get("hardest", [])
        ],
        empty_message="  No task-difficulty data recorded.",
    )

    print_table(
        "Easiest Tasks",
        ["Task", "N", "Reward pass rate", "Mean reward model", "Mean programmatic"],
        [
            [
                str(row.get("task_id") or ""),
                str(row.get("rollout_count") or 0),
                format_percent(row.get("pass_rate")),
                format_number(row.get("mean_reward_model_score")),
                format_number(row.get("mean_composite_score")),
            ]
            for row in task_difficulty.get("easiest", [])
        ],
        empty_message="  No task-difficulty data recorded.",
    )

    print_table(
        "Model Comparison",
        [
            "Model",
            "N",
            "Reward pass rate",
            "Mean reward model",
            "Mean programmatic",
            "Avg cost",
            "Avg duration",
        ],
        [
            [
                str(row.get("model") or ""),
                str(row.get("rollout_count") or 0),
                format_percent(row.get("pass_rate")),
                format_number(row.get("mean_reward_model_score")),
                format_number(row.get("mean_composite_score")),
                format_money(row.get("avg_cost_usd")),
                format_duration(row.get("avg_duration_seconds")),
            ]
            for row in summary.get("model_comparison", [])
        ],
        empty_message="  No model summary recorded.",
    )

    print_table(
        "Capability Analysis",
        ["Capability", "Pass rate", "Passes", "Checks", "Failed rollouts", "Attempt signal"],
        [
            [
                str(row.get("capability") or ""),
                format_percent(row.get("pass_rate")),
                str(row.get("pass_count") or 0),
                str(row.get("total_count") or 0),
                str(row.get("failed_rollout_count") or 0),
                str(row.get("attempt_signal") or ""),
            ]
            for row in summary.get("capability_analysis", [])
        ],
        empty_message="  No capability summary recorded.",
    )

    print_table(
        "Co-Failure Patterns",
        ["Capability", "Checks", "Fail count", "Rollouts"],
        [
            [
                str(row.get("capability") or ""),
                ", ".join(row.get("criteria_names", [])),
                str(row.get("fail_count") or 0),
                str(row.get("rollout_count") or 0),
            ]
            for row in summary.get("cofailure_patterns", [])
        ],
        empty_message="  No co-failure patterns recorded.",
    )


def print_compare_report(report: dict[str, Any]) -> None:
    """Dispatch to the rollout or dataset comparison renderer."""
    if report.get("compare_kind") == "rollout":
        print_rollout_compare(report)
        return

    print_dataset_compare(report)


def print_rollout_compare(report: dict[str, Any]) -> None:
    """Render the human-readable comparison for two single rollouts."""
    left_rollout = report["left"]["rollout"]
    right_rollout = report["right"]["rollout"]
    comparison = report["rollout_comparison"]

    click.echo("Rollout Comparison")
    if left_rollout.get("task_id") == right_rollout.get("task_id"):
        click.echo(f"Task: {left_rollout.get('task_id')}")
    else:
        click.echo(f"A task: {left_rollout.get('task_id')}")
        click.echo(f"B task: {right_rollout.get('task_id')}")
    click.echo(f"A path: {left_rollout.get('path')}")
    click.echo(f"B path: {right_rollout.get('path')}")

    print_table(
        "Summary",
        ["Metric", "A", "B", "A-B", "Better"],
        [
            [
                rollout_metric_label(row.get("metric")),
                format_rollout_compare_value(row.get("metric"), row.get("left_value")),
                format_rollout_compare_value(row.get("metric"), row.get("right_value")),
                format_rollout_compare_delta(row.get("metric"), row.get("delta")),
                better_side(row.get("metric"), row.get("left_value"), row.get("right_value")),
            ]
            for row in comparison.get("overview", [])
        ],
        empty_message="  No rollout comparison data recorded.",
    )

    print_metric_table(
        "Tool Flow",
        [
            ("A tool errors", str(len(comparison.get("left_errors", [])))),
            ("B tool errors", str(len(comparison.get("right_errors", [])))),
        ],
    )

    print_table(
        "Programmatic Verifier Comparison",
        ["Criterion", "A", "B", "Notes"],
        [
            [
                compact(row.get("criterion"), 32),
                format_pass_fail(row.get("left_passed")),
                format_pass_fail(row.get("right_passed")),
                compact(comparison_notes(row.get("left_detail"), row.get("right_detail")), 72),
            ]
            for row in comparison.get("criteria", [])
        ],
        empty_message="  No programmatic verifier results recorded.",
    )

    print_table(
        "Reward Model Dimension Comparison",
        ["Dimension", "A", "B", "A-B", "Better"],
        [
            [
                compact(row.get("dimension"), 28),
                format_number(row.get("left_score")),
                format_number(row.get("right_score")),
                format_signed(row.get("delta_score")),
                better_side("reward_model_score", row.get("left_score"), row.get("right_score")),
            ]
            for row in comparison.get("reward_model_dimensions", [])
        ],
        empty_message="  No reward model dimensions recorded.",
    )

    print_table(
        "Tool Sequence",
        ["#", "A tool", "B tool", "Match", "A status", "B status"],
        [
            [
                str(row.get("step") or ""),
                str(row.get("left_tool") or ""),
                str(row.get("right_tool") or ""),
                "yes" if row.get("same_tool") else "no",
                "error" if row.get("left_error") else ("ok" if row.get("left_tool") else ""),
                "error" if row.get("right_error") else ("ok" if row.get("right_tool") else ""),
            ]
            for row in comparison.get("tool_sequence", [])[:15]
        ],
        empty_message="  No tool calls recorded.",
    )

    print_table(
        "A Error Calls",
        ["#", "Tool", "Summary"],
        [
            [
                str(row.get("index") or ""),
                str(row.get("tool") or ""),
                compact(row.get("summary"), 72),
            ]
            for row in comparison.get("left_errors", [])
        ],
        empty_message="  No tool errors on A.",
    )

    print_table(
        "B Error Calls",
        ["#", "Tool", "Summary"],
        [
            [
                str(row.get("index") or ""),
                str(row.get("tool") or ""),
                compact(row.get("summary"), 72),
            ]
            for row in comparison.get("right_errors", [])
        ],
        empty_message="  No tool errors on B.",
    )


def print_dataset_compare(report: dict[str, Any]) -> None:
    """Render the human-readable comparison for two rollout sets."""
    left = report["left"]
    right = report["right"]
    comparison = report["comparison"]

    click.echo("Comparison Report")
    click.echo(f"A: {Path(left['path']).name}")
    click.echo(f"B: {Path(right['path']).name}")
    click.echo(f"A path: {left['path']}")
    click.echo(f"B path: {right['path']}")

    print_table(
        "Overview",
        ["Metric", "A", "B", "A-B", "Better"],
        [
            [
                overview_metric_label(row.get("metric")),
                format_overview_value(row.get("metric"), row.get("left_value")),
                format_overview_value(row.get("metric"), row.get("right_value")),
                format_overview_delta(row.get("metric"), row.get("delta")),
                better_side(row.get("metric"), row.get("left_value"), row.get("right_value")),
            ]
            for row in comparison.get("overview", [])
        ],
        empty_message="  No comparison data recorded.",
    )

    print_table(
        "Task Comparison",
        [
            "Task",
            "A reward",
            "B reward",
            "A-B reward",
            "A reward model",
            "B reward model",
            "A-B reward model",
            "Better",
        ],
        [
            [
                str(row.get("task_id") or ""),
                format_percent(row.get("left_pass_rate")),
                format_percent(row.get("right_pass_rate")),
                format_signed_percent(row.get("delta_pass_rate")),
                format_number(row.get("left_mean_reward_model_score")),
                format_number(row.get("right_mean_reward_model_score")),
                format_signed(row.get("delta_mean_reward_model_score")),
                better_side(
                    "pass_rate",
                    row.get("left_pass_rate"),
                    row.get("right_pass_rate"),
                ),
            ]
            for row in comparison.get("results_by_task", [])
        ],
        empty_message="  No task comparison data recorded.",
    )

    print_table(
        "Model Comparison",
        [
            "Model",
            "A reward",
            "B reward",
            "A-B reward",
            "A reward model",
            "B reward model",
            "A-B reward model",
            "Better",
        ],
        [
            [
                str(row.get("model") or ""),
                format_percent(row.get("left_pass_rate")),
                format_percent(row.get("right_pass_rate")),
                format_signed_percent(row.get("delta_pass_rate")),
                format_number(row.get("left_mean_reward_model_score")),
                format_number(row.get("right_mean_reward_model_score")),
                format_signed(row.get("delta_mean_reward_model_score")),
                better_side(
                    "pass_rate",
                    row.get("left_pass_rate"),
                    row.get("right_pass_rate"),
                ),
            ]
            for row in comparison.get("model_comparison", [])
        ],
        empty_message="  No model comparison data recorded.",
    )

    print_table(
        "System Failure Rate Comparison",
        ["System", "A fail", "B fail", "Delta fail", "Better"],
        [
            [
                str(row.get("system") or ""),
                format_percent(row.get("left_fail_rate")),
                format_percent(row.get("right_fail_rate")),
                format_signed_percent(row.get("delta_fail_rate")),
                better_side(
                    "fail_rate",
                    row.get("left_fail_rate"),
                    row.get("right_fail_rate"),
                ),
            ]
            for row in comparison.get("system_failure_rates", [])
        ],
        empty_message="  No system comparison data recorded.",
    )

    print_table(
        "Capability Comparison",
        [
            "Capability",
            "A pass",
            "B pass",
            "A-B pass",
            "A results",
            "B results",
            "Better",
        ],
        [
            [
                str(row.get("capability") or ""),
                format_percent(row.get("left_pass_rate")),
                format_percent(row.get("right_pass_rate")),
                format_signed_percent(row.get("delta_pass_rate")),
                format_compare_value(row.get("left_total_count")),
                format_compare_value(row.get("right_total_count")),
                better_side(
                    "pass_rate",
                    row.get("left_pass_rate"),
                    row.get("right_pass_rate"),
                ),
            ]
            for row in comparison.get("capability_analysis", [])
        ],
        empty_message="  No capability comparison data recorded.",
    )


def print_metric_table(title: str, rows: list[tuple[str, str]]) -> None:
    """Print a two-column metric table."""
    print_table(
        title,
        ["Metric", "Value"],
        [[label, value] for label, value in rows],
        empty_message="  No data recorded.",
    )


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    empty_message: str,
) -> None:
    """Print a text table with optional empty-state messaging."""
    click.echo("")
    click.echo(title)
    if not rows:
        click.echo(empty_message)
        return

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    click.echo(
        "  " + "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    click.echo("  " + "  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rows:
        click.echo("  " + "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def print_warnings(warnings: list[str]) -> None:
    """Print any warnings collected while building the report."""
    if not warnings:
        return
    click.echo("", err=True)
    click.echo("Warnings", err=True)
    for warning in warnings:
        click.echo(f"  - {warning}", err=True)


def single_rollout_status(
    rollout: dict[str, Any],
    passed_count: int,
    total_count: int,
) -> str:
    """Describe the overall outcome of one rollout."""
    if rollout.get("error"):
        return f"Error - {rollout['error']}"
    passed = rollout.get("passed")
    if passed is True and total_count:
        return f"Passed ({passed_count}/{total_count} programmatic results passed)"
    if passed is False and total_count:
        return f"Failed ({passed_count}/{total_count} programmatic results passed)"
    if passed is True:
        return "Passed"
    if passed is False:
        return "Failed"
    return "Unknown"


def tool_label(tool_call: dict[str, Any]) -> str:
    """Build the canonical tool label used in reports."""
    return f"{tool_call.get('tool_server') or 'unknown'}__{tool_call.get('tool_name') or 'unknown'}"


def format_model_summary(models: list[str]) -> str:
    """Format the model list for the report header."""
    if not models:
        return "unknown model"
    if len(models) == 1:
        return models[0]
    return f"{len(models)} models"


def format_percent(value: Any) -> str:
    """Format a numeric value as a percentage."""
    if not isinstance(value, (int, float)):
        return "n/a"
    pct = value * 100.0
    if abs(pct - round(pct)) < 0.05:
        return f"{pct:.0f}%"
    return f"{pct:.1f}%"


def format_signed_percent(value: Any) -> str:
    """Format a signed percentage delta."""
    if not isinstance(value, (int, float)):
        return "n/a"
    pct = value * 100.0
    sign = "+" if pct > 0 else ""
    if abs(pct - round(pct)) < 0.05:
        return f"{sign}{pct:.0f}%"
    return f"{sign}{pct:.1f}%"


def format_money(value: Any) -> str:
    """Format a dollar amount for display."""
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"${float(value):.2f}"


def format_duration(value: Any) -> str:
    """Format a duration in seconds for display."""
    if not isinstance(value, (int, float)):
        return "n/a"
    total_seconds = max(round(float(value)), 0)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def format_tokens(value: Any) -> str:
    """Format a token count for display."""
    if not isinstance(value, (int, float)):
        return "n/a"
    numeric = float(value)
    if numeric >= 1_000_000:
        return f"{numeric / 1_000_000:.1f}M"
    if numeric >= 1_000:
        return f"{numeric / 1_000:.1f}K"
    return str(round(numeric))


def format_number(value: Any) -> str:
    """Format a floating-point value for display."""
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.3f}"


def format_signed(value: Any) -> str:
    """Format a signed numeric delta."""
    if not isinstance(value, (int, float)):
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{float(value):.3f}"


def format_signed_money(value: Any) -> str:
    """Format a signed dollar delta."""
    if not isinstance(value, (int, float)):
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}${float(value):.2f}"


def format_compare_value(value: Any) -> str:
    """Format one value inside a rollout comparison table."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "n/a"
    return str(value)


def compact(value: Any, max_chars: int) -> str:
    """Trim long display text to the requested width."""
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def compact_json(value: Any) -> str:
    """Serialize JSON data into a compact display string."""
    if not value:
        return ""
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def pass_rate(passed_count: int, total_count: int) -> float | None:
    """Return a pass rate when a denominator is present."""
    if total_count <= 0:
        return None
    return passed_count / total_count


def format_pass_fail(value: Any) -> str:
    """Format a boolean pass or fail value for display."""
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "n/a"


def rollout_metric_label(metric: Any) -> str:
    """Translate rollout comparison metric keys into labels."""
    labels = {
        "passed": "Reward earned",
        "reward": "Reward",
        "composite_score": "Programmatic verifier score",
        "reward_model_score": "Reward model score",
        "steps_taken": "Steps",
        "tool_calls": "Tool calls",
        "tool_errors": "Tool errors",
        "duration_seconds": "Duration",
        "cost_usd": "Cost",
    }
    return labels.get(str(metric), str(metric))


def overview_metric_label(metric: Any) -> str:
    """Translate dataset comparison metric keys into labels."""
    labels = {
        "rollouts": "Rollouts",
        "tasks": "Tasks",
        "pass_rate": "Reward pass rate",
        "mean_composite_score": "Mean programmatic verifier score",
        "mean_reward_model_score": "Mean reward model score",
        "reward_model_score_variance": "Reward model score variance",
        "reward_rate": "Reward earned rate",
        "total_tool_errors": "Total tool errors",
        "avg_steps_per_rollout": "Average recorded steps per rollout",
        "avg_duration_seconds": "Average duration",
        "avg_cost_usd": "Average cost",
    }
    return labels.get(str(metric), str(metric))


def format_rollout_compare_value(metric: Any, value: Any) -> str:
    """Format a rollout comparison cell value."""
    if metric == "passed":
        return format_pass_fail(value)
    if metric in {"reward", "composite_score", "reward_model_score"}:
        return format_number(value)
    if metric in {"steps_taken", "tool_calls", "tool_errors"}:
        return format_compare_value(value)
    if metric == "duration_seconds":
        return format_duration(value)
    if metric == "cost_usd":
        return format_money(value)
    return format_compare_value(value)


def format_rollout_compare_delta(metric: Any, value: Any) -> str:
    """Format a rollout comparison delta value."""
    if metric == "passed":
        return "n/a"
    if metric in {"reward", "composite_score", "reward_model_score"}:
        return format_signed(value)
    if metric in {"steps_taken", "tool_calls", "tool_errors"}:
        return format_signed_int(value)
    if metric == "duration_seconds":
        return format_signed_duration(value)
    if metric == "cost_usd":
        return format_signed_money(value)
    return format_signed(value)


def format_signed_int(value: Any) -> str:
    """Format a signed integer delta."""
    if not isinstance(value, (int, float)):
        return "n/a"
    numeric = round(float(value))
    sign = "+" if numeric > 0 else ""
    return f"{sign}{numeric}"


def format_signed_duration(value: Any) -> str:
    """Format a signed duration delta."""
    if not isinstance(value, (int, float)):
        return "n/a"
    numeric = float(value)
    sign = "+" if numeric > 0 else "-"
    return sign + format_duration(abs(numeric))


def better_side(metric: Any, left_value: Any, right_value: Any) -> str:
    """Return which side performed better for a comparison metric."""
    if metric == "passed":
        if left_value is True and right_value is not True:
            return "A"
        if right_value is True and left_value is not True:
            return "B"
        return "=" if left_value == right_value else "n/a"
    if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
        return "n/a"
    if left_value == right_value:
        return "="
    higher_is_better = {
        "reward",
        "composite_score",
        "reward_model_score",
        "pass_rate",
        "reward_rate",
    }
    lower_is_better = {
        "fail_rate",
        "total_tool_errors",
        "tool_calls",
        "tool_errors",
        "avg_duration_seconds",
        "duration_seconds",
        "avg_cost_usd",
        "cost_usd",
        "avg_steps_per_rollout",
        "steps_taken",
    }
    if metric in higher_is_better:
        return "A" if left_value > right_value else "B"
    if metric in lower_is_better:
        return "A" if left_value < right_value else "B"
    return "n/a"


def comparison_notes(left_detail: Any, right_detail: Any) -> str:
    """Summarize differing detail text for a comparison row."""
    left_text = str(left_detail or "").strip()
    right_text = str(right_detail or "").strip()
    if not left_text and not right_text:
        return ""
    if left_text == right_text:
        return left_text
    if left_text and right_text:
        return f"A: {left_text} | B: {right_text}"
    if left_text:
        return f"A: {left_text}"
    return f"B: {right_text}"


def format_overview_value(metric: Any, value: Any) -> str:
    """Format a dataset comparison overview value."""
    if metric in {"pass_rate", "reward_rate"}:
        return format_percent(value)
    if metric == "avg_cost_usd":
        return format_money(value)
    if metric in {
        "mean_composite_score",
        "mean_reward_model_score",
        "reward_model_score_variance",
    }:
        return format_number(value)
    return format_compare_value(value)


def format_overview_delta(metric: Any, value: Any) -> str:
    """Format a dataset comparison overview delta."""
    if metric in {"pass_rate", "reward_rate"}:
        return format_signed_percent(value)
    if metric == "avg_cost_usd":
        return format_signed_money(value)
    return format_signed(value)


def format_timestamp(value: str) -> str:
    """Format an ISO timestamp for display."""
    return value.replace("T", " ").replace("Z", " UTC")
