"""Tests for simlab.cli.progress — StepProgress spinner and reporter."""

from __future__ import annotations

import io

import pytest
from rich.console import Console
from simlab.cli.progress import DefaultReporter
from simlab.cli.progress import StepContext
from simlab.cli.progress import StepProgress
from simlab.cli.progress import StepProgressReporter


def _make_progress(
    *, verbose: bool = False, tty: bool = False
) -> tuple[StepProgress, io.StringIO, io.StringIO]:
    """Create a StepProgress with captured output streams."""
    out = io.StringIO()
    err = io.StringIO()
    console = Console(file=out, force_terminal=tty, highlight=False)
    err_console = Console(file=err, force_terminal=False, highlight=False)
    progress = StepProgress(verbose=verbose, console=console, err_console=err_console)
    return progress, out, err


def test_step_success_tty() -> None:
    """Successful step prints a green checkmark on TTY."""
    progress, out, _ = _make_progress(tty=True)

    with progress.step("Services started"):
        pass

    output = out.getvalue()
    assert "✓" in output
    assert "Services started" in output


def test_step_success_non_tty() -> None:
    """Successful step prints [done] tag when not a TTY."""
    progress, out, _ = _make_progress(tty=False)

    with progress.step("Services started"):
        pass

    assert "[done] Services started" in out.getvalue()


def test_step_success_uses_success_label() -> None:
    """Successful steps can render a different completion label."""
    progress, out, _ = _make_progress(tty=False)

    with progress.step("Building images", success_label="Images built"):
        pass

    output = out.getvalue()
    assert "[done] Images built" in output
    assert "[done] Building images" not in output


def _raise_with_details(progress: StepProgress) -> None:
    """Helper — raises inside a step with buffered details."""
    with progress.step("Services started") as ctx:
        ctx.detail("attempt 1")
        ctx.detail("attempt 2")
        raise RuntimeError("boom")


def test_step_failure_tty() -> None:
    """Failed step prints a red cross on TTY and dumps buffered details."""
    progress, out, err = _make_progress(tty=True)

    with pytest.raises(RuntimeError, match="boom"):
        _raise_with_details(progress)

    output = out.getvalue()
    assert "✗" in output
    assert "Services started" in output
    # Buffered details should be dumped to stderr on failure
    err_output = err.getvalue()
    assert "attempt 1" in err_output
    assert "attempt 2" in err_output


def test_step_failure_non_tty() -> None:
    """Failed step prints [FAIL] tag when not a TTY."""
    progress, out, _ = _make_progress(tty=False)

    with pytest.raises(RuntimeError), progress.step("Docker pull"):
        raise RuntimeError("fail")

    assert "[FAIL] Docker pull" in out.getvalue()


def test_step_failure_system_exit() -> None:
    """SystemExit is caught and renders failure state before re-raising."""
    progress, out, _ = _make_progress(tty=False)

    with pytest.raises(SystemExit), progress.step("Build images"):
        raise SystemExit(1)

    assert "[FAIL] Build images" in out.getvalue()


def test_verbose_mode_prints_details_immediately() -> None:
    """In verbose mode, detail() messages print immediately to stdout."""
    progress, out, _ = _make_progress(verbose=True, tty=False)

    with progress.step("Services started") as ctx:
        ctx.detail("pulling image A")
        ctx.detail("pulling image B")

    output = out.getvalue()
    assert "pulling image A" in output
    assert "pulling image B" in output


def test_non_verbose_buffers_details() -> None:
    """In non-verbose mode, detail() messages are NOT printed on success."""
    progress, out, err = _make_progress(verbose=False, tty=False)

    with progress.step("Services started") as ctx:
        ctx.detail("pulling image A")

    assert "pulling image A" not in out.getvalue()
    assert "pulling image A" not in err.getvalue()


def test_finish_with_endpoints() -> None:
    """finish() prints elapsed time and endpoint table."""
    progress, out, _ = _make_progress(tty=False)

    progress.finish(
        42.3,
        endpoints={
            "jira": "http://localhost:8000",
            "coding": "http://localhost:8001",
        },
    )

    output = out.getvalue()
    assert "Environment ready" in output
    assert "42s" in output
    assert "jira" in output
    assert "http://localhost:8000" in output
    assert "coding" in output


def test_finish_without_endpoints() -> None:
    """finish() works when no endpoints are provided."""
    progress, out, _ = _make_progress(tty=False)

    progress.finish(10.0)

    output = out.getvalue()
    assert "Environment ready" in output
    assert "10s" in output
    assert "Tool endpoints" not in output


def test_step_context_detail_verbose() -> None:
    """StepContext in verbose mode writes to console immediately."""
    out = io.StringIO()
    console = Console(file=out, force_terminal=False, highlight=False)
    ctx = StepContext(verbose=True, console=console)
    ctx.detail("hello")
    assert "hello" in out.getvalue()


def test_step_context_detail_buffered() -> None:
    """StepContext in non-verbose mode buffers messages."""
    out = io.StringIO()
    console = Console(file=out, force_terminal=False, highlight=False)
    ctx = StepContext(verbose=False, console=console)
    ctx.detail("buffered msg")
    assert len(ctx._buffer) == 1
    assert ctx._buffer[0] == "buffered msg"


def test_step_context_flush_buffer() -> None:
    """flush_buffer writes to err_console."""
    out = io.StringIO()
    err = io.StringIO()
    console = Console(file=out, force_terminal=False, highlight=False)
    err_console = Console(file=err, force_terminal=False, highlight=False)
    ctx = StepContext(verbose=False, console=console, err_console=err_console)
    ctx.detail("err detail")
    ctx.flush_buffer()
    assert "err detail" in err.getvalue()


def test_default_reporter_start_step(capsys: pytest.CaptureFixture[str]) -> None:
    """DefaultReporter.start_step() prints via click.echo for backwards compat."""
    rpt = DefaultReporter()
    rpt.start_step("Doing something")
    captured = capsys.readouterr()
    assert "Doing something..." in captured.out


def test_step_progress_reporter_success() -> None:
    """StepProgressReporter wraps StepProgress for the reporter protocol."""
    progress, out, _ = _make_progress(verbose=False, tty=False)
    rpt = StepProgressReporter(progress)

    rpt.start_step("Files uploaded")
    rpt.detail("uploaded docker-compose.yml")
    rpt.end_step(success=True)

    output = out.getvalue()
    assert "[done] Files uploaded" in output
    # Detail should be buffered (non-verbose), not printed on success
    assert "uploaded docker-compose.yml" not in output


def test_step_progress_reporter_success_uses_success_label() -> None:
    """Reporter callers can override the completion label for a step."""
    progress, out, _ = _make_progress(verbose=False, tty=False)
    rpt = StepProgressReporter(progress)

    rpt.start_step("Building images", success_label="Images built")
    rpt.end_step(success=True)

    output = out.getvalue()
    assert "[done] Images built" in output
    assert "[done] Building images" not in output


def test_step_progress_reporter_failure() -> None:
    """StepProgressReporter renders failure and dumps details."""
    progress, out, err = _make_progress(verbose=False, tty=False)
    rpt = StepProgressReporter(progress)

    rpt.start_step("Docker build")
    rpt.detail("build log line 1")
    rpt.end_step(success=False, error="build failed")

    assert "[FAIL] Docker build" in out.getvalue()
    assert "build log line 1" in err.getvalue()
    assert "build failed" in err.getvalue()
