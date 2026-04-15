from __future__ import annotations

from pathlib import Path

import pytest
from simlab.cli.autoresearch import find_environments_dir_candidates


def test_find_environments_dir_candidates_discovers_prompt_autoresearch_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_name = "prompt-autoresearch-ops"
    env_root = tmp_path / "simlab_prompt_autoresearch" / "environments" / env_name
    env_root.mkdir(parents=True)
    (env_root / "env.yaml").write_text("name: test\n", encoding="utf-8")

    tasks_dir = tmp_path / "simlab_prompt_autoresearch" / "fixtures" / "task_bundle"
    tasks_dir.mkdir(parents=True)

    other_dir = tmp_path / "some_other_dir"
    other_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(other_dir)

    candidates = find_environments_dir_candidates(
        env_name=env_name,
        tasks_dir=tasks_dir,
        base_dir=None,
        max_depth=6,
    )
    assert candidates == [tmp_path / "simlab_prompt_autoresearch" / "environments"]


def test_find_environments_dir_candidates_expands_user_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    env_name = "prompt-autoresearch-ops"
    env_root = tmp_path / "simlab_prompt_autoresearch" / "environments" / env_name
    env_root.mkdir(parents=True)
    (env_root / "env.yaml").write_text("name: test\n", encoding="utf-8")

    (tmp_path / "simlab_prompt_autoresearch" / "fixtures" / "task_bundle").mkdir(parents=True)
    tasks_dir = Path("~/simlab_prompt_autoresearch/fixtures/task_bundle")

    other_dir = tmp_path / "some_other_dir"
    other_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(other_dir)

    candidates = find_environments_dir_candidates(
        env_name=env_name,
        tasks_dir=tasks_dir,
        base_dir=None,
        max_depth=6,
    )
    assert candidates == [tmp_path / "simlab_prompt_autoresearch" / "environments"]


def test_find_environments_dir_candidates_includes_base_dir_when_tasks_dir_is_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "env1"
    config_dir = tmp_path / "project"
    env_root = config_dir / "environments" / env_name
    env_root.mkdir(parents=True)
    (env_root / "env.yaml").write_text("name: env1\n", encoding="utf-8")

    external_bundle = tmp_path / "external" / "bundle"
    external_bundle.mkdir(parents=True, exist_ok=True)

    other_dir = tmp_path / "some_other_dir"
    other_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(other_dir)

    candidates = find_environments_dir_candidates(
        env_name=env_name,
        tasks_dir=external_bundle,
        base_dir=config_dir,
        max_depth=4,
    )
    assert candidates == [config_dir / "environments"]
