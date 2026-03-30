from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from simlab import env_artifacts


class _InteractiveStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_detect_generation_drift_reports_changed_custom_tool(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    custom_tools_dir = env_dir / "custom-tools"
    custom_tools_dir.mkdir(parents=True)
    (env_dir / "env.yaml").write_text("name: test-env\ntools: []\n", encoding="utf-8")
    tool_file = custom_tools_dir / "harbor-main.yaml"
    tool_file.write_text("name: harbor-main\n", encoding="utf-8")

    env_artifacts.write_generation_state(env_dir)

    tool_file.write_text("name: harbor-main\ndescription: updated\n", encoding="utf-8")

    is_stale, reasons = env_artifacts.detect_generation_drift(env_dir)

    assert is_stale is True
    assert "Changed input: custom-tools/harbor-main.yaml" in reasons


def test_ensure_env_artifacts_current_regenerates_after_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[Path] = []

    def fake_regenerate(env_dir: Path) -> None:
        called.append(env_dir)

    monkeypatch.setattr(
        env_artifacts,
        "detect_generation_drift",
        lambda _env_dir: (True, ["Changed input: env.yaml"]),
    )
    monkeypatch.setattr(
        env_artifacts.click,
        "confirm",
        lambda _message, default=True: default,
    )
    monkeypatch.setattr(
        env_artifacts,
        "regenerate_env_artifacts",
        fake_regenerate,
    )
    monkeypatch.setattr(
        env_artifacts.sys,
        "stdin",
        SimpleNamespace(isatty=lambda: True),
    )
    interactive_stdout = _InteractiveStringIO()
    monkeypatch.setattr(
        env_artifacts.sys,
        "stdout",
        interactive_stdout,
    )

    env_artifacts.ensure_env_artifacts_current(tmp_path, action_label="env up")

    assert called == [tmp_path]


def test_ensure_env_artifacts_current_fails_noninteractive_when_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        env_artifacts,
        "detect_generation_drift",
        lambda _env_dir: (True, ["Changed input: custom-tools/harbor-main.yaml"]),
    )
    monkeypatch.setattr(
        env_artifacts.sys,
        "stdin",
        SimpleNamespace(isatty=lambda: False),
    )
    monkeypatch.setattr(
        env_artifacts.sys,
        "stdout",
        SimpleNamespace(isatty=lambda: False),
    )

    with pytest.raises(SystemExit) as exc_info:
        env_artifacts.ensure_env_artifacts_current(tmp_path, action_label="tasks run")

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Generated environment files are stale before tasks run." in captured.err
    assert f"simlab env init {tmp_path.name} --force" in captured.err
