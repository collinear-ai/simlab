# ruff: noqa: ANN401
"""CLI entry point for ``simlab eval``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from simlab.evaluation import EvaluationError
from simlab.evaluation import build_report


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
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON output.")
def eval_command(
    path: Path,
    compare_path: Path | None,
    json_output: bool,
) -> None:
    """Analyze local rollout artifacts under PATH. Defaults to ./output."""
    try:
        report = build_report(
            path,
            compare_path=compare_path,
        )
    except EvaluationError as exc:
        raise click.ClickException(str(exc)) from exc

    if json_output:
        click.echo(json.dumps(report, indent=2))
        return

    if report["mode"] == "single_rollout":
        print_single_rollout(report["rollout"])
    elif report["mode"] == "compare":
        print_compare_report(report)
    else:
        print_multi_rollout(report["path"], report["summary"])

    print_warnings(report.get("warnings", []))


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
