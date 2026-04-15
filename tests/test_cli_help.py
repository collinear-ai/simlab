"""Smoke tests for the branded `simlab --help` output."""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from simlab.cli.main import COLLINEAR_DARK
from simlab.cli.main import COLLINEAR_ECRU
from simlab.cli.main import _pick_adaptive_body_color
from simlab.cli.main import cli

ALL_COMMANDS = (
    "quickstart",
    "auth",
    "templates",
    "env",
    "tools",
    "tasks-gen",
    "tasks",
    "eval",
    "runs",
    "feedback",
)

HIDDEN_GLOBAL_OPTIONS = (
    "--config-file",
    "--collinear-api-key",
    "--scenario-manager-api-url",
    "--daytona-api-key",
    "--environments-dir",
)


def test_help_shows_tagline_and_all_commands() -> None:
    """Plain --help renders the branded screen with every command listed."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "hill-climbing" in result.output
    assert "simlab quickstart" in result.output
    assert "https://docs.collinear.ai" in result.output

    for command in ALL_COMMANDS:
        assert command in result.output, f"missing command in help: {command}"


def test_help_hides_global_options_by_default() -> None:
    """Plain --help must not list the five global options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    for option in HIDDEN_GLOBAL_OPTIONS:
        assert option not in result.output, f"unexpectedly visible: {option}"


def test_help_verbose_reveals_global_options() -> None:
    """`--help --verbose` reveals the hidden global options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help", "--verbose"])

    assert result.exit_code == 0, result.output
    for option in HIDDEN_GLOBAL_OPTIONS:
        assert option in result.output, f"verbose help missing: {option}"


def test_help_verbose_reverse_order_also_works() -> None:
    """`--verbose --help` (flipped order) must also reveal global options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--verbose", "--help"])

    assert result.exit_code == 0, result.output
    for option in HIDDEN_GLOBAL_OPTIONS:
        assert option in result.output, f"verbose help missing: {option}"


def test_branded_help_mentions_plain_flag_for_ai_agents() -> None:
    """Branded help must surface the --plain flag so AI agents notice it."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "AI agents" in result.output
    assert "--plain" in result.output


def test_plain_help_is_unstyled_and_lists_everything() -> None:
    """--help --plain returns plain text with all commands and global options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help", "--plain"])

    assert result.exit_code == 0, result.output
    # No ASCII art, no box drawing.
    assert "███" not in result.output
    assert "━━━" not in result.output
    assert "┏" not in result.output
    # Tagline and usage line are present.
    assert "hill-climbing" in result.output
    assert "Usage: simlab" in result.output
    # Every command with its description.
    for command in ALL_COMMANDS:
        assert command in result.output, f"missing command: {command}"
    # Every global option (they are shown in plain mode, unlike branded mode).
    for option in HIDDEN_GLOBAL_OPTIONS:
        assert option in result.output, f"missing global option: {option}"


def test_plain_help_reverse_order_also_works() -> None:
    """--plain --help (flipped order) also renders plain output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--plain", "--help"])

    assert result.exit_code == 0, result.output
    assert "███" not in result.output
    assert "Usage: simlab" in result.output


def test_adaptive_palette_picks_dark_for_light_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """COLORFGBG reporting a light background → body text uses COLLINEAR_DARK."""
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert _pick_adaptive_body_color() == COLLINEAR_DARK


def test_adaptive_palette_picks_ecru_for_dark_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """COLORFGBG reporting a dark background → body text uses COLLINEAR_ECRU."""
    monkeypatch.setenv("COLORFGBG", "15;0")
    assert _pick_adaptive_body_color() == COLLINEAR_ECRU


def test_adaptive_palette_falls_back_when_colorfgbg_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No COLORFGBG → body text uses Rich's 'default' (terminal-native)."""
    monkeypatch.delenv("COLORFGBG", raising=False)
    assert _pick_adaptive_body_color() == "default"


def test_adaptive_palette_falls_back_when_colorfgbg_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unparseable COLORFGBG → body text uses Rich's 'default'."""
    monkeypatch.setenv("COLORFGBG", "nonsense;values")
    assert _pick_adaptive_body_color() == "default"
