"""Reusable step-progress spinner for CLI output, powered by rich."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from contextlib import contextmanager
from typing import Protocol
from typing import runtime_checkable

import click
from rich.console import Console


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
    def step(self, label: str) -> Generator[StepContext, None, None]:
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
            self._render_result(label, success=True)

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

    def start_step(self, label: str) -> None:
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

    def start_step(self, label: str) -> None:
        """Begin a named step by printing via click.echo."""
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

    def start_step(self, label: str) -> None:
        """Begin a step with a spinner."""
        cm = self._progress.step(label)
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
                # Render failure without re-raising — the caller handles the error.
                with contextlib.suppress(RuntimeError):
                    self._exit(  # type: ignore[operator]
                        RuntimeError,
                        RuntimeError(error or "step failed"),
                        None,
                    )
            self._exit = None
            self._ctx = None
