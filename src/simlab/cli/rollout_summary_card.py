"""Rich post-rollout summary cards for ``simlab tasks run``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

RUBRIC_PASS_THRESHOLD = 0.6
MAX_CHECK_ROWS = 10
CHECK_CONTAINER_KEYS = ("checks", "results", "criteria_results")


@dataclass(frozen=True)
class CheckRow:
    """A single check row shown in rollout summary cards."""

    name: str
    passed: bool
    detail: str | None = None
    source: str | None = None


def print_single_rollout_summary_card(
    *,
    task_id: str,
    model: str | None,
    provider: str | None,
    steps_taken: int,
    max_steps: int | None,
    duration_seconds: float | None,
    reward: float | None,
    verification_passed: bool | None,
    run_error: str | None,
    verifier_results: list[dict[str, Any]] | None,
    output_dir: Path,
    quiet: bool,
    console: Console | None = None,
) -> None:
    """Print a post-run summary card for a single rollout."""
    if quiet:
        return

    console = console or Console(stderr=True)
    checks = extract_checks(verifier_results or [])
    passed_checks = sum(1 for check in checks if check.passed)
    check_count = len(checks)

    header, border_style = single_rollout_header(
        run_error=run_error,
        verification_passed=verification_passed,
    )
    fields = Table.grid(padding=(0, 1))
    fields.add_column(justify="left", style="bold")
    fields.add_column(justify="left", overflow="fold")
    fields.add_row("Task", task_id)
    fields.add_row("Model", format_model(model=model, provider=provider))
    fields.add_row("Steps", format_steps(steps_taken=steps_taken, max_steps=max_steps))
    fields.add_row("Duration", format_duration(duration_seconds))
    fields.add_row("Reward", format_reward(reward))

    if check_count:
        fields.add_row("Checks", f"{passed_checks}/{check_count} passed")
    else:
        fields.add_row("Checks", "n/a")

    if run_error:
        fields.add_row("Error", compact(run_error, width=76))

    check_block = build_checks_block(checks)
    body: Group
    if check_block is None:
        body = Group(header, fields, Text(f"Output: {output_dir}", style="dim"))
    else:
        body = Group(
            header,
            fields,
            Text(""),
            check_block,
            Text(""),
            Text(f"Output: {output_dir}", style="dim"),
        )

    panel = Panel(
        body,
        box=box.SQUARE,
        border_style=border_style,
        padding=(0, 1),
        title="Rollout Summary",
        title_align="left",
    )
    console.print()
    console.print(panel, highlight=False)
    console.print(Text(f"Next: simlab eval {output_dir}", style="dim"), highlight=False)
    console.print()


def print_parallel_rollout_summary_card(
    *,
    task_id: str,
    rollout_count: int,
    results: list[Any],
    total_duration_seconds: float,
    output_dir: Path | None,
    quiet: bool,
    console: Console | None = None,
) -> None:
    """Print a post-run summary card for a parallel rollout run."""
    if quiet:
        return

    console = console or Console(stderr=True)

    has_failures = any(parallel_row_is_failure(r) for r in results)
    has_unverified = any(parallel_row_is_unverified(r) for r in results)
    passed = [r for r in results if parallel_row_is_pass(r)]
    rewards = [float(r.reward) for r in results if getattr(r, "reward", None) is not None]

    fields = Table.grid(padding=(0, 1))
    fields.add_column(justify="left", style="bold")
    fields.add_column(justify="left", overflow="fold")
    fields.add_row("Task", task_id)
    fields.add_row("Rollouts", str(rollout_count))
    fields.add_row("Passed", f"{len(passed)}/{rollout_count}")
    fields.add_row("Avg reward", f"{(sum(rewards) / len(rewards)):.2f}" if rewards else "n/a")
    fields.add_row("Duration", f"{format_duration(total_duration_seconds)} total")
    if output_dir is not None:
        fields.add_row("Output", str(output_dir))

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Reward", justify="right", no_wrap=True)
    table.add_column("Steps", justify="right", no_wrap=True)
    table.add_column("Time", justify="right", no_wrap=True)

    for r in results:
        status_text = parallel_rollout_status(r)
        reward_str = format_reward(getattr(r, "reward", None))
        table.add_row(
            str(getattr(r, "rollout_idx", "?")),
            status_text,
            reward_str,
            str(getattr(r, "steps_taken", 0)),
            format_duration(float(getattr(r, "duration_seconds", 0.0))),
        )

    error_rows = [
        (
            str(getattr(r, "rollout_idx", "?")),
            compact(str(getattr(r, "error", "")), width=120),
        )
        for r in results
        if getattr(r, "error", None)
    ]
    error_table = None
    if error_rows:
        error_table = Table(box=box.SIMPLE, show_header=True, header_style="bold", padding=(0, 1))
        error_table.add_column("#", justify="right", no_wrap=True)
        error_table.add_column("Error", overflow="fold")
        for idx, message in error_rows[:10]:
            error_table.add_row(idx, message)

    if error_table:
        body = Group(fields, Text(""), table, Text(""), error_table)
    else:
        body = Group(fields, Text(""), table)
    panel = Panel(
        body,
        box=box.SQUARE,
        border_style="red" if has_failures else "yellow" if has_unverified else "green",
        padding=(0, 1),
        title="Parallel Rollout Summary",
        title_align="left",
    )

    console.print()
    console.print(panel, highlight=False)
    if output_dir is not None:
        console.print(Text(f"Next: simlab eval {output_dir}", style="dim"), highlight=False)
    else:
        console.print(Text("Next: simlab eval ./output", style="dim"), highlight=False)
    console.print()


def single_rollout_header(
    *, run_error: str | None, verification_passed: bool | None
) -> tuple[Text, str]:
    """Return the header line and border style for a single rollout panel."""
    if verification_passed is True:
        return Text("✓ PASS", style="bold green"), "green"
    if verification_passed is False:
        return Text("✗ FAIL", style="bold red"), "red"
    if run_error:
        return Text("✗ ERROR", style="bold red"), "red"
    return Text("✓ DONE", style="bold yellow"), "yellow"


def parallel_rollout_status(result: object) -> Text:
    """Render a per-rollout status cell."""
    if getattr(result, "error", None):
        return Text("ERR", style="bold red")
    passed = getattr(result, "verification_passed", None)
    if passed is True:
        return Text("PASS", style="bold green")
    if passed is False:
        return Text("FAIL", style="bold red")
    return Text("N/A", style="bold yellow")


def extract_checks(verifier_results: list[dict[str, Any]]) -> list[CheckRow]:
    """Extract structured checks from verifier results, falling back to module rows."""
    checks: list[CheckRow] = []
    for result in verifier_results:
        module = coerce_str(result.get("module")) or "verifier"
        source = module_label(module)
        success = bool(result.get("success"))
        detail = coerce_str(result.get("message")) or coerce_str(result.get("output"))

        payload = parse_json_blob(coerce_str(result.get("output")))
        if payload is not None:
            extracted = extract_checks_from_payload(payload, source=source)
        else:
            extracted = []
        if extracted:
            checks.extend(extracted)
            if not success and all(check.passed for check in extracted):
                checks.append(CheckRow(name=source, passed=False, detail=detail, source=source))
            continue

        checks.append(CheckRow(name=source, passed=success, detail=detail, source=source))

    return dedupe_checks(checks)


def extract_checks_from_payload(payload: object, *, source: str) -> list[CheckRow]:
    """Extract checks from a verifier JSON payload.

    Supports both legacy ``{"criteria": "...", "pass": true}`` and
    structured ``{"name": "...", "passed": true}`` rows.
    """
    if isinstance(payload, list):
        checks: list[CheckRow] = []
        for item in payload:
            checks.extend(extract_checks_from_payload(item, source=source))
        return checks

    if not isinstance(payload, dict):
        return []

    check = leaf_check_from_payload(payload, source=source)
    if check is not None:
        return [check]

    for key in CHECK_CONTAINER_KEYS:
        child = payload.get(key)
        if not isinstance(child, list):
            continue
        nested: list[CheckRow] = []
        for item in child:
            nested.extend(extract_checks_from_payload(item, source=source))
        if nested:
            return nested

    dimension_scores = payload.get("dimension_scores")
    if isinstance(dimension_scores, list):
        checks = []
        for item in dimension_scores:
            if not isinstance(item, dict):
                continue
            dimension_name = coerce_str(item.get("dimension"))
            score = safe_float(item.get("score"))
            if not dimension_name or score is None:
                continue
            checks.append(
                CheckRow(
                    name=dimension_name,
                    passed=score >= RUBRIC_PASS_THRESHOLD,
                    detail=coerce_str(item.get("reason")),
                    source=source,
                )
            )
        if checks:
            return checks

    failed_criteria = payload.get("failed_criteria")
    if isinstance(failed_criteria, list):
        verdict = (
            coerce_str(payload.get("verdict"))
            or coerce_str(payload.get("message"))
            or coerce_str(payload.get("detail"))
        )
        checks = []
        for item in failed_criteria:
            name = coerce_str(item)
            if not name:
                continue
            checks.append(CheckRow(name=name, passed=False, detail=verdict, source=source))
        if checks:
            return checks

    return []


def dedupe_checks(checks: list[CheckRow]) -> list[CheckRow]:
    """Deduplicate checks by name while preserving first-seen order."""
    by_name: dict[str, CheckRow] = {}
    order: list[str] = []
    for check in checks:
        if check.name not in by_name:
            by_name[check.name] = check
            order.append(check.name)
            continue
        existing = by_name[check.name]
        if existing.passed and not check.passed:
            by_name[check.name] = check
    return [by_name[name] for name in order]


def leaf_check_from_payload(payload: dict[str, Any], *, source: str) -> CheckRow | None:
    """Extract a single check row from a JSON payload if present."""
    name = coerce_str(payload.get("name"))
    passed = payload.get("passed") if isinstance(payload.get("passed"), bool) else None
    if name and passed is not None:
        detail = (
            coerce_str(payload.get("detail"))
            or coerce_str(payload.get("message"))
            or coerce_str(payload.get("reason"))
        )
        return CheckRow(name=name, passed=passed, detail=detail, source=source)

    legacy_name = coerce_str(payload.get("criteria"))
    legacy_passed = payload.get("pass") if isinstance(payload.get("pass"), bool) else None
    if legacy_name and legacy_passed is not None:
        detail = (
            coerce_str(payload.get("detail"))
            or coerce_str(payload.get("message"))
            or coerce_str(payload.get("reason"))
        )
        return CheckRow(name=legacy_name, passed=legacy_passed, detail=detail, source=source)

    return None


def build_checks_block(checks: list[CheckRow]) -> Group | None:
    """Build the Key checks section for the summary panel."""
    if not checks:
        return None

    failures = [check for check in checks if not check.passed]
    passes = [check for check in checks if check.passed]

    display: list[CheckRow] = []
    display.extend(failures[:MAX_CHECK_ROWS])
    if len(display) < MAX_CHECK_ROWS:
        display.extend(passes[: MAX_CHECK_ROWS - len(display)])

    remaining = len(checks) - len(display)

    lines: list[Text] = []
    lines.append(Text("Key checks", style="bold"))
    for check in display:
        prefix = Text("✓ ", style="green") if check.passed else Text("✗ ", style="red")
        line = Text.assemble(prefix, Text(check.name))
        lines.append(line)
    if remaining > 0:
        lines.append(Text(f"… +{remaining} more", style="dim"))
    return Group(*lines)


def format_model(*, model: str | None, provider: str | None) -> str:
    """Format model and provider for display."""
    if model and provider:
        return f"{model} ({provider})"
    return model or "n/a"


def format_steps(*, steps_taken: int, max_steps: int | None) -> str:
    """Format step counters for display."""
    if max_steps is None:
        return str(steps_taken)
    return f"{steps_taken}/{max_steps}"


def format_reward(reward: float | None) -> str:
    """Format reward values for display."""
    if reward is None:
        return "-"
    return f"{reward:.1f}"


def format_duration(duration_seconds: float | None) -> str:
    """Format durations in seconds into a compact human-readable form."""
    if duration_seconds is None:
        return "n/a"
    total = round(duration_seconds)
    if total < 60:
        return f"{total}s"
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def module_label(module_path: str) -> str:
    """Return a short label for a dotted Python module path."""
    parts = [part for part in module_path.split(".") if part]
    return parts[-1] if parts else module_path


def parse_json_blob(raw_text: str | None) -> object | None:
    """Parse JSON from a raw text blob, returning None if parsing fails."""
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def compact(value: str | None, *, width: int) -> str:
    """Compact multi-line text into a single trimmed line."""
    text = (value or "").replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 3].rstrip() + "..."


def parallel_row_is_pass(result: object) -> bool:
    """Return True when the rollout passed verification."""
    if getattr(result, "error", None):
        return False
    return getattr(result, "verification_passed", None) is True


def parallel_row_is_failure(result: object) -> bool:
    """Return True when the rollout errored or failed verification."""
    if getattr(result, "error", None):
        return True
    return getattr(result, "verification_passed", None) is False


def parallel_row_is_unverified(result: object) -> bool:
    """Return True when the rollout completed without running verifiers."""
    if getattr(result, "error", None):
        return False
    return getattr(result, "verification_passed", None) is None


def coerce_str(value: object) -> str | None:
    """Coerce a value into a non-empty string, returning None otherwise."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def safe_float(value: object) -> float | None:
    """Parse a value into a float, returning None when parsing fails."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
