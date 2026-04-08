"""CLI commands for browsing local run output."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.table import Table

from simlab.runs_history import RunHistoryEntry
from simlab.runs_history import load_runs_history


@click.group()
def runs() -> None:
    """Inspect run output saved under the output directory."""


@runs.command("history")
@click.option(
    "--output-dir",
    default=Path("output"),
    type=click.Path(path_type=Path, exists=False, file_okay=False),
    help="Output directory to scan (default: ./output).",
)
@click.option(
    "--last",
    "last_n",
    default=None,
    type=click.IntRange(min=1),
    help="Show only the N most recent runs.",
)
@click.option("--model", "model_filter", default=None, help="Filter to runs using MODEL.")
@click.option("--task", "task_filter", default=None, help="Filter to runs matching TASK.")
@click.option(
    "--result",
    "result_filter",
    default=None,
    type=click.Choice(["pass", "fail"], case_sensitive=False),
    help="Filter to runs with the given result.",
)
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON output.")
def runs_history(
    output_dir: Path,
    last_n: int | None,
    model_filter: str | None,
    task_filter: str | None,
    result_filter: str | None,
    json_output: bool,
) -> None:
    """List past runs from OUTPUT_DIR in a compact table."""
    try:
        entries, warnings = load_runs_history(output_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise click.ClickException(f"Output directory not found: {exc}") from exc

    entries = apply_filters(
        entries,
        model_filter=model_filter,
        task_filter=task_filter,
        result_filter=(result_filter.lower() if result_filter else None),
    )
    if last_n is not None:
        entries = entries[:last_n]

    if json_output:
        payload = {
            "output_dir": str(output_dir.expanduser().resolve()),
            "count": len(entries),
            "runs": [entry.to_json() for entry in entries],
            "warnings": warnings,
        }
        click.echo(json.dumps(payload, indent=2))
        return

    print_history_table(entries)
    click.echo()
    click.echo(f"{len(entries)} runs shown. Run `simlab eval` for detailed analysis.")
    if warnings:
        click.echo("", err=True)
        click.echo("Warnings", err=True)
        for warning in warnings[:25]:
            click.echo(f"  - {warning}", err=True)
        if len(warnings) > 25:
            click.echo(f"  - ... and {len(warnings) - 25} more", err=True)


def apply_filters(
    entries: list[RunHistoryEntry],
    *,
    model_filter: str | None,
    task_filter: str | None,
    result_filter: str | None,
) -> list[RunHistoryEntry]:
    """Apply CLI filters to history entries."""
    model_value = model_filter.strip().lower() if isinstance(model_filter, str) else None
    task_value = task_filter.strip().lower() if isinstance(task_filter, str) else None
    result_value = result_filter.strip().lower() if isinstance(result_filter, str) else None

    filtered: list[RunHistoryEntry] = []
    for entry in entries:
        if model_value and (entry.model or "").lower() != model_value:
            continue
        if task_value and entry.task_id.lower() != task_value:
            continue
        if result_value and entry.result != result_value:
            continue
        filtered.append(entry)
    return filtered


def print_history_table(entries: list[RunHistoryEntry]) -> None:
    """Render the history table for human-readable output."""
    console = Console()
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("Date", no_wrap=True)
    table.add_column("Task", overflow="ellipsis")
    table.add_column("Model", no_wrap=True)
    table.add_column("Result", no_wrap=True)
    table.add_column("Duration", justify="right", no_wrap=True)

    for entry in entries:
        table.add_row(
            format_history_date(entry.created_at),
            entry.task_id,
            entry.model or "n/a",
            format_history_result(entry.result),
            format_history_duration(entry.duration_seconds),
        )

    console.print(table, highlight=False)


def format_history_date(value: datetime | None) -> str:
    """Format a run timestamp for the history table."""
    if value is None:
        return "unknown"
    local = value.astimezone()
    return local.strftime("%Y-%m-%d %H:%M")


def format_history_result(value: str) -> str:
    """Format a run pass/fail result for the history table."""
    if value == "pass":
        return "✓ Pass"
    if value == "fail":
        return "✗ Fail"
    return "n/a"


def format_history_duration(value: float | None) -> str:
    """Format a duration in seconds for the history table."""
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
