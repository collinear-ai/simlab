"""Reusable step-progress spinner for CLI output, powered by rich."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol
from typing import Self
from typing import runtime_checkable

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


class StepContext:
    """Handle passed into a ``step()`` block for emitting verbose detail."""

    def __init__(  # noqa: D107
        self,
        *,
        verbose: bool,
        console: Console,
        err_console: Console | None = None,
    ) -> None:
        self._verbose = verbose
        self._console = console
        self._err_console = err_console or Console(stderr=True)
        self._buffer: list[str] = []
        self.status: object | None = None

    def detail(self, message: str) -> None:
        """Emit a detail line (shown immediately in verbose mode, buffered otherwise)."""
        if self._verbose:
            self._console.print(f"    {message}", highlight=False)
        else:
            self._buffer.append(message)

    def flush_buffer(self) -> None:
        """Dump buffered detail messages (used on failure in non-verbose mode)."""
        for msg in self._buffer:
            self._err_console.print(f"    {msg}", highlight=False)

    def update(self, message: str) -> None:
        """Update the in-progress status message when running in a TTY."""
        status = self.status
        if status is None:
            return
        update = getattr(status, "update", None)
        if callable(update):
            update(message)


class StepProgress:
    """Display a step-by-step spinner for long-running CLI operations."""

    def __init__(  # noqa: D107
        self,
        *,
        verbose: bool = False,
        console: Console | None = None,
        err_console: Console | None = None,
    ) -> None:
        self._verbose = verbose
        self._console = console or Console()
        self._err_console = err_console or Console(stderr=True)

    @contextmanager
    def step(
        self,
        label: str,
        *,
        success_label: str | None = None,
    ) -> Generator[StepContext, None, None]:
        """Context manager that shows a spinner while the body executes."""
        ctx = StepContext(
            verbose=self._verbose,
            console=self._console,
            err_console=self._err_console,
        )
        status = None

        if self._console.is_terminal:
            status = self._console.status(f"  {label}...", spinner="dots")
            status.start()
            ctx.status = status

        try:
            yield ctx
        except (Exception, SystemExit):
            if status is not None:
                status.stop()
            self._render_result(label, success=False)
            ctx.flush_buffer()
            raise
        else:
            if status is not None:
                status.stop()
            self._render_result(success_label or label, success=True)

    def finish(
        self,
        total_time: float,
        endpoints: dict[str, str] | None = None,
    ) -> None:
        """Print final summary with elapsed time and optional endpoint table."""
        time_str = f"{int(total_time)}s"
        if self._console.is_terminal:
            self._console.print(f"\n  [green]✓[/green] Environment ready — {time_str}")
        else:
            self._console.print(f"\n\\[done] Environment ready — {time_str}", highlight=False)

        if endpoints:
            self._console.print("\n  Endpoints:")
            max_name = max(len(n) for n in endpoints) if endpoints else 0
            for name, url in endpoints.items():
                padding = " " * (max_name - len(name))
                self._console.print(f"    {name}:{padding}  {url}", highlight=False)

        self._console.print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_result(self, label: str, *, success: bool) -> None:
        """Write the final checkmark or cross for a step."""
        if self._console.is_terminal:
            if success:
                self._console.print(f"  [green]✓[/green] {label}")
            else:
                self._console.print(f"  [red]✗[/red] {label}")
        else:
            tag = "\\[done]" if success else "\\[FAIL]"
            self._console.print(f"{tag} {label}", highlight=False)


# ------------------------------------------------------------------
# ProgressReporter protocol for DaytonaRunner
# ------------------------------------------------------------------


@runtime_checkable
class ProgressReporter(Protocol):
    """Minimal reporting interface accepted by DaytonaRunner.up()."""

    def start_step(self, label: str, *, success_label: str | None = None) -> None:
        """Begin a named step."""
        ...

    def detail(self, message: str) -> None:
        """Emit a detail message within the current step."""
        ...

    def end_step(self, *, success: bool = True, error: str | None = None) -> None:
        """Finish the current step."""
        ...


class DefaultReporter:
    """Backwards-compatible reporter that uses click.echo (pre-existing behaviour)."""

    def start_step(self, label: str, *, success_label: str | None = None) -> None:
        """Begin a named step by printing via click.echo."""
        _ = success_label
        click.echo(f"{label}...")

    def detail(self, message: str) -> None:
        """Print a detail message via click.echo."""
        click.echo(f"  {message}")

    def end_step(self, *, success: bool = True, error: str | None = None) -> None:
        """No-op — DefaultReporter relies on caller printing results."""


class StepProgressReporter:
    """Adapter bridging StepProgress into the ProgressReporter protocol."""

    def __init__(self, progress: StepProgress) -> None:  # noqa: D107
        self._progress = progress
        self._ctx: StepContext | None = None
        self._exit: object = None

    def start_step(self, label: str, *, success_label: str | None = None) -> None:
        """Begin a step with a spinner."""
        cm = self._progress.step(label, success_label=success_label)
        self._ctx = cm.__enter__()
        self._exit = cm.__exit__

    def detail(self, message: str) -> None:
        """Forward detail to the active step context."""
        if self._ctx is not None:
            self._ctx.detail(message)

    def end_step(self, *, success: bool = True, error: str | None = None) -> None:
        """Complete the active step, rendering success or failure."""
        if self._exit is not None:
            if success:
                self._exit(None, None, None)  # type: ignore[operator]
            else:
                if error and self._ctx is not None:
                    for line in error.splitlines():
                        if line.strip():
                            self._ctx.detail(line)
                # Render failure without re-raising — the caller handles the error.
                with contextlib.suppress(RuntimeError):
                    self._exit(  # type: ignore[operator]
                        RuntimeError,
                        RuntimeError(error or "step failed"),
                        None,
                    )
            self._exit = None
            self._ctx = None


@dataclass(frozen=True)
class ParallelRolloutRow:
    """Snapshot of one rollout row in the parallel progress table."""

    rollout_idx: int
    task_name: str
    status: str
    steps_taken: int | None
    max_steps: int | None
    result: str | None


class ParallelRolloutProgress:
    """Live-updating table for parallel rollout progress."""

    def __init__(
        self,
        *,
        rollout_count: int,
        task_name: str,
        max_steps: int | None,
        console: Console | None = None,
    ) -> None:
        """Initialize the table with all rollouts queued."""
        self._console = console or Console()
        self._lock = threading.Lock()
        self._rollout_count = rollout_count
        self._rows: dict[int, ParallelRolloutRow] = {
            idx: ParallelRolloutRow(
                rollout_idx=idx,
                task_name=task_name,
                status="Queued",
                steps_taken=None,
                max_steps=max_steps,
                result=None,
            )
            for idx in range(rollout_count)
        }
        self._live: Live | None = None

    def __enter__(self) -> Self:
        """Start live rendering when stdout is a TTY."""
        if not self._console.is_terminal:
            return self
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=8,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop live rendering if it was started."""
        if self._live is None:
            return
        self._live.__exit__(exc_type, exc, tb)
        self._live = None

    def update(
        self,
        rollout_idx: int,
        *,
        status: str | None = None,
        steps_taken: int | None = None,
        result: str | None = None,
    ) -> None:
        """Update the status fields for a single rollout row."""
        with self._lock:
            current = self._rows.get(rollout_idx)
            if current is None:
                return
            self._rows[rollout_idx] = ParallelRolloutRow(
                rollout_idx=rollout_idx,
                task_name=current.task_name,
                status=status if status is not None else current.status,
                steps_taken=steps_taken if steps_taken is not None else current.steps_taken,
                max_steps=current.max_steps,
                result=result if result is not None else current.result,
            )

    def __rich_console__(self, console: Console, _options: object) -> Generator[Table, None, None]:
        """Render a Rich table for live updates."""
        _ = console
        yield self._build_table()

    def _build_table(self) -> Table:
        with self._lock:
            rows = [self._rows[i] for i in range(self._rollout_count)]

        table = Table(show_header=True, header_style="bold", padding=(0, 1))
        table.add_column("Rollout", no_wrap=True)
        table.add_column("Task", overflow="fold")
        table.add_column("Status", no_wrap=True)
        table.add_column("Steps", justify="right", no_wrap=True)
        table.add_column("Result", no_wrap=True)

        for row in rows:
            rollout = f"{row.rollout_idx + 1}/{self._rollout_count}"
            status = _format_parallel_status(row.status)
            steps = _format_parallel_steps(
                steps_taken=row.steps_taken,
                max_steps=row.max_steps,
            )
            result = _format_parallel_result(row.result)
            table.add_row(rollout, row.task_name, status, steps, result)
        return table


def _format_parallel_status(status: str) -> Text:
    normalized = status.strip().lower()
    if normalized == "queued":
        return Text(status, style="dim")
    if normalized == "starting":
        return Text(status, style="yellow")
    if normalized == "running":
        return Text(status, style="cyan")
    if normalized == "verifying":
        return Text(status, style="magenta")
    if normalized == "done":
        return Text(status, style="green")
    return Text(status)


def _format_parallel_steps(*, steps_taken: int | None, max_steps: int | None) -> str:
    if steps_taken is None:
        return "—"
    if max_steps is None:
        return str(steps_taken)
    return f"{steps_taken}/{max_steps}"


def _format_parallel_result(result: str | None) -> Text:
    if not result:
        return Text("")
    normalized = result.strip().upper()
    if normalized == "PASS":
        return Text(normalized, style="bold green")
    if normalized in {"FAIL", "ERR"}:
        return Text(normalized, style="bold red")
    return Text(normalized, style="bold yellow")
