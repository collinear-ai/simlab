from __future__ import annotations

from pathlib import Path

import pytest
import simlab.cli.autoresearch as autoresearch_module
from click.testing import CliRunner
from simlab.autoresearch.config import AutoresearchRunConfig
from simlab.cli.autoresearch import autoresearch


def test_autoresearch_run_quick_start_records_absolute_tasks_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    captured: dict[str, AutoresearchRunConfig] = {}

    def fake_run_autoresearch(*, cfg: AutoresearchRunConfig, **_kwargs: object) -> Path:
        captured["cfg"] = cfg
        return tmp_path / "output"

    monkeypatch.setattr(autoresearch_module, "run_autoresearch", fake_run_autoresearch)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        env_dir = Path("environments") / "env1"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "env.yaml").write_text("name: env1\n", encoding="utf-8")

        Path("bundle").mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            autoresearch,
            ["run", "--env", "env1", "--tasks-dir", "bundle", "--task", "t1"],
        )

    assert result.exit_code == 0, result.output
    cfg = captured.get("cfg")
    assert cfg is not None
    tasks_dir_value = Path(cfg.run.tasks_dir)
    assert tasks_dir_value.is_absolute()
    assert cfg.run.runtime == "local"
    assert cfg.run.rollout_count == 1
    assert cfg.run.max_parallel == 1
    assert cfg.budget.max_iterations == 1


def test_autoresearch_run_quick_start_normalizes_tasks_dir_tasks_subdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    captured: dict[str, AutoresearchRunConfig] = {}

    def fake_run_autoresearch(*, cfg: AutoresearchRunConfig, **_kwargs: object) -> Path:
        captured["cfg"] = cfg
        return tmp_path / "output"

    monkeypatch.setattr(autoresearch_module, "run_autoresearch", fake_run_autoresearch)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        env_dir = Path("environments") / "env1"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "env.yaml").write_text("name: env1\n", encoding="utf-8")

        (Path("bundle") / "tasks").mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            autoresearch,
            ["run", "--env", "env1", "--tasks-dir", "bundle/tasks", "--task", "t1"],
        )

    assert result.exit_code == 0, result.output
    cfg = captured.get("cfg")
    assert cfg is not None
    tasks_dir_value = Path(cfg.run.tasks_dir)
    assert tasks_dir_value.is_absolute()
    assert tasks_dir_value.name != "tasks"
