from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from simlab.autoresearch.config import AutoresearchRunConfig
from simlab.autoresearch.manager import run_autoresearch


def _write_env(tmp_path: Path) -> None:
    env_dir = tmp_path / "environments" / "env1"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: env1\nscenario_guidance_md: |\n  # Scenario Guidance\n  Baseline\n",
        encoding="utf-8",
    )


def _write_task(bundle_dir: Path, *, meta_task_id: str, file_stem: str | None = None) -> None:
    tasks_dir = bundle_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"task_id": meta_task_id}, "task": "Test task"}
    stem = file_stem or meta_task_id
    (tasks_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_run_autoresearch_allows_missing_proposer_key_when_max_iterations_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_env(tmp_path)

    tasks_dir = tmp_path / "bundle"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    _write_task(tasks_dir, meta_task_id="t1")

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": str(tasks_dir),
                "task_ids": ["t1"],
                "runtime": "local",
                "rollout_count": 1,
                "max_parallel": 1,
                "max_steps": 5,
                "agent_timeout_seconds": 10.0,
                "no_seed": True,
            },
            "agent": {"model": "gpt-4o-mini", "provider": "openai"},
            "proposer": {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "MISSING_PROPOSER_KEY",
            },
            "verifier": {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "MISSING_VERIFIER_KEY",
            },
            "objective": {"type": "pass_rate", "target": None},
            "budget": {"max_iterations": 0, "max_minutes": 1, "no_improvement_window": 0},
        }
    )

    def fake_run_and_eval_iteration(
        **_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return {"summary": {}}, {"iteration": 0, "objective_value": 0.0, "pass_rate": 0.0}

    monkeypatch.setattr(
        "simlab.autoresearch.manager._run_and_eval_iteration",
        fake_run_and_eval_iteration,
    )
    monkeypatch.setattr("simlab.autoresearch.manager.write_end_of_run_reports", lambda **_k: None)

    ctx = click.Context(click.Command("simlab"))
    run_dir = run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")
    assert (run_dir / "baseline" / "scenario_prompt.md").is_file()


def test_run_autoresearch_requires_proposer_key_when_entering_iteration_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_env(tmp_path)

    tasks_dir = tmp_path / "bundle"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    _write_task(tasks_dir, meta_task_id="t1")

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": str(tasks_dir),
                "task_ids": ["t1"],
                "runtime": "local",
                "rollout_count": 1,
                "max_parallel": 1,
                "max_steps": 5,
                "agent_timeout_seconds": 10.0,
                "no_seed": True,
            },
            "agent": {"model": "gpt-4o-mini", "provider": "openai"},
            "proposer": {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "MISSING_PROPOSER_KEY",
            },
            "verifier": {"model": "gpt-5.4", "provider": "openai"},
            "objective": {"type": "pass_rate", "target": 0.8},
            "budget": {"max_iterations": 1, "max_minutes": 1, "no_improvement_window": 0},
        }
    )

    def fake_run_and_eval_iteration(
        **_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return {"summary": {}}, {"iteration": 0, "objective_value": 0.0, "pass_rate": 0.0}

    monkeypatch.setattr(
        "simlab.autoresearch.manager._run_and_eval_iteration",
        fake_run_and_eval_iteration,
    )
    monkeypatch.setattr("simlab.autoresearch.manager.write_end_of_run_reports", lambda **_k: None)

    ctx = click.Context(click.Command("simlab"))
    with pytest.raises(click.ClickException, match="Missing proposer API key"):
        run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")


def test_run_autoresearch_requires_proposer_key_when_max_minutes_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_env(tmp_path)

    tasks_dir = tmp_path / "bundle"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    _write_task(tasks_dir, meta_task_id="t1")

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": str(tasks_dir),
                "task_ids": ["t1"],
                "runtime": "local",
                "rollout_count": 1,
                "max_parallel": 1,
                "max_steps": 5,
                "agent_timeout_seconds": 10.0,
                "no_seed": True,
            },
            "agent": {"model": "gpt-4o-mini", "provider": "openai"},
            "proposer": {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "MISSING_PROPOSER_KEY",
            },
            "verifier": {"model": "gpt-5.4", "provider": "openai"},
            "objective": {"type": "pass_rate", "target": 0.8},
            "budget": {"max_iterations": 1, "max_minutes": -1, "no_improvement_window": 0},
        }
    )

    def fake_run_and_eval_iteration(
        **_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return {"summary": {}}, {"iteration": 0, "objective_value": 0.0, "pass_rate": 0.0}

    monkeypatch.setattr(
        "simlab.autoresearch.manager._run_and_eval_iteration",
        fake_run_and_eval_iteration,
    )
    monkeypatch.setattr("simlab.autoresearch.manager.write_end_of_run_reports", lambda **_k: None)

    ctx = click.Context(click.Command("simlab"))
    with pytest.raises(click.ClickException, match="Missing proposer API key"):
        run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")


def test_run_autoresearch_allows_missing_verifier_key_when_no_rubric_files_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_env(tmp_path)

    tasks_dir = tmp_path / "bundle"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    _write_task(tasks_dir, meta_task_id="t1")

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": str(tasks_dir),
                "task_ids": ["t1"],
                "runtime": "local",
                "rollout_count": 1,
                "max_parallel": 1,
                "max_steps": 5,
                "agent_timeout_seconds": 10.0,
                "no_seed": True,
            },
            "agent": {"model": "gpt-4o-mini", "provider": "openai"},
            "proposer": {"model": "gpt-5.4", "provider": "openai"},
            "verifier": {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "MISSING_VERIFIER_KEY",
            },
            "objective": {"type": "pass_rate", "target": None},
            "budget": {"max_iterations": 0, "max_minutes": 1, "no_improvement_window": 0},
        }
    )

    def fake_run_and_eval_iteration(
        **_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return {"summary": {}}, {"iteration": 0, "objective_value": 0.0, "pass_rate": 0.0}

    monkeypatch.setattr(
        "simlab.autoresearch.manager._run_and_eval_iteration",
        fake_run_and_eval_iteration,
    )
    monkeypatch.setattr("simlab.autoresearch.manager.write_end_of_run_reports", lambda **_k: None)

    ctx = click.Context(click.Command("simlab"))
    run_dir = run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")
    assert (run_dir / "baseline" / "scenario_prompt.md").is_file()


def test_run_autoresearch_ignores_rubrics_for_other_tasks_when_using_shorthand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_env(tmp_path)

    tasks_dir = tmp_path / "bundle"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    _write_task(tasks_dir, meta_task_id="suite-1")
    _write_task(tasks_dir, meta_task_id="suite-10")

    (tasks_dir / "rubrics").mkdir(parents=True, exist_ok=True)
    (tasks_dir / "rubrics" / "suite-10.md").write_text("# Rubric\n", encoding="utf-8")

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": str(tasks_dir),
                "task_ids": ["1"],
                "runtime": "local",
                "rollout_count": 1,
                "max_parallel": 1,
                "max_steps": 5,
                "agent_timeout_seconds": 10.0,
                "no_seed": True,
            },
            "agent": {"model": "gpt-4o-mini", "provider": "openai"},
            "proposer": {"model": "gpt-5.4", "provider": "openai"},
            "verifier": {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "MISSING_VERIFIER_KEY",
            },
            "objective": {"type": "pass_rate", "target": None},
            "budget": {"max_iterations": 0, "max_minutes": 1, "no_improvement_window": 0},
        }
    )

    def fake_run_and_eval_iteration(
        **_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return {"summary": {}}, {"iteration": 0, "objective_value": 0.0, "pass_rate": 0.0}

    monkeypatch.setattr(
        "simlab.autoresearch.manager._run_and_eval_iteration",
        fake_run_and_eval_iteration,
    )
    monkeypatch.setattr("simlab.autoresearch.manager.write_end_of_run_reports", lambda **_k: None)

    ctx = click.Context(click.Command("simlab"))
    run_dir = run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")
    assert (run_dir / "baseline" / "scenario_prompt.md").is_file()


def test_run_autoresearch_requires_verifier_key_when_rubric_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_env(tmp_path)

    tasks_dir = tmp_path / "bundle"
    (tasks_dir / "rubrics").mkdir(parents=True, exist_ok=True)
    (tasks_dir / "rubrics" / "t1.md").write_text("# Rubric\n", encoding="utf-8")
    _write_task(tasks_dir, meta_task_id="t1")

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": str(tasks_dir),
                "task_ids": ["t1"],
                "runtime": "local",
                "rollout_count": 1,
                "max_parallel": 1,
                "max_steps": 5,
                "agent_timeout_seconds": 10.0,
                "no_seed": True,
            },
            "agent": {"model": "gpt-4o-mini", "provider": "openai"},
            "proposer": {"model": "gpt-5.4", "provider": "openai"},
            "verifier": {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "MISSING_VERIFIER_KEY",
            },
            "objective": {"type": "pass_rate", "target": None},
            "budget": {"max_iterations": 0, "max_minutes": 1, "no_improvement_window": 0},
        }
    )

    ctx = click.Context(click.Command("simlab"))
    with pytest.raises(click.ClickException, match="Missing verifier API key"):
        run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")


def test_run_autoresearch_skips_iteration_loop_when_baseline_hits_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_env(tmp_path)

    tasks_dir = tmp_path / "bundle"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    _write_task(tasks_dir, meta_task_id="t1")

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": str(tasks_dir),
                "task_ids": ["t1"],
                "runtime": "local",
                "rollout_count": 1,
                "max_parallel": 1,
                "max_steps": 5,
                "agent_timeout_seconds": 10.0,
                "no_seed": True,
            },
            "agent": {"model": "gpt-4o-mini", "provider": "openai"},
            "proposer": {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "MISSING_PROPOSER_KEY",
            },
            "verifier": {"model": "gpt-5.4", "provider": "openai"},
            "objective": {"type": "pass_rate", "target": 0.8},
            "budget": {"max_iterations": 3, "max_minutes": 1, "no_improvement_window": 0},
        }
    )

    def fake_run_and_eval_iteration(
        **_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return {"summary": {}}, {"iteration": 0, "objective_value": 1.0, "pass_rate": 1.0}

    monkeypatch.setattr(
        "simlab.autoresearch.manager._run_and_eval_iteration",
        fake_run_and_eval_iteration,
    )
    monkeypatch.setattr("simlab.autoresearch.manager.write_end_of_run_reports", lambda **_k: None)

    def fail_propose(**_kwargs: object) -> object:
        raise AssertionError("should not propose after baseline hits target")

    monkeypatch.setattr("simlab.autoresearch.manager.propose_next_change", fail_propose)

    ctx = click.Context(click.Command("simlab"))
    run_dir = run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")
    assert (run_dir / "baseline" / "scenario_prompt.md").is_file()


def test_run_autoresearch_requires_verifier_key_for_shorthand_task_id_with_rubric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_env(tmp_path)

    tasks_dir = tmp_path / "bundle"
    (tasks_dir / "rubrics").mkdir(parents=True, exist_ok=True)
    (tasks_dir / "rubrics" / "suite-foo.md").write_text("# Rubric\n", encoding="utf-8")
    _write_task(tasks_dir, meta_task_id="suite-foo")

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": str(tasks_dir),
                "task_ids": ["foo"],
                "runtime": "local",
                "rollout_count": 1,
                "max_parallel": 1,
                "max_steps": 5,
                "agent_timeout_seconds": 10.0,
                "no_seed": True,
            },
            "agent": {"model": "gpt-4o-mini", "provider": "openai"},
            "proposer": {"model": "gpt-5.4", "provider": "openai"},
            "verifier": {
                "model": "gpt-5.4",
                "provider": "openai",
                "api_key_env": "MISSING_VERIFIER_KEY",
            },
            "objective": {"type": "pass_rate", "target": None},
            "budget": {"max_iterations": 0, "max_minutes": 1, "no_improvement_window": 0},
        }
    )

    ctx = click.Context(click.Command("simlab"))
    with pytest.raises(click.ClickException, match="Missing verifier API key"):
        run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")


def test_run_autoresearch_rejects_reward_model_objective_without_rubrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_env(tmp_path)

    tasks_dir = tmp_path / "bundle"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    _write_task(tasks_dir, meta_task_id="t1")

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": str(tasks_dir),
                "task_ids": ["t1"],
                "runtime": "local",
                "rollout_count": 1,
                "max_parallel": 1,
                "max_steps": 5,
                "agent_timeout_seconds": 10.0,
                "no_seed": True,
            },
            "agent": {"model": "gpt-4o-mini", "provider": "openai"},
            "proposer": {"model": "gpt-5.4", "provider": "openai"},
            "verifier": {"model": "gpt-5.4", "provider": "openai"},
            "objective": {"type": "reward_model_score_mean", "target": None},
            "budget": {"max_iterations": 0, "max_minutes": 1, "no_improvement_window": 0},
        }
    )

    ctx = click.Context(click.Command("simlab"))
    with pytest.raises(click.ClickException, match="requires rubric files"):
        run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")
