from __future__ import annotations

from pathlib import Path

import click
import pytest
from simlab.autoresearch.config import AutoresearchRunConfig
from simlab.autoresearch.manager import run_autoresearch


def test_run_autoresearch_writes_crash_json_on_systemexit_after_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    env_dir = tmp_path / "environments" / "env1"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: env1\nscenario_guidance_md: |\n  # Scenario Guidance\n  Baseline\n",
        encoding="utf-8",
    )

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": "./tasks",
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
            "objective": {"type": "pass_rate", "target": None},
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

    def fake_propose_next_change(**_kwargs: object) -> dict[str, object]:
        raise SystemExit(7)

    monkeypatch.setattr("simlab.autoresearch.manager.propose_next_change", fake_propose_next_change)
    monkeypatch.setattr("simlab.autoresearch.manager.write_end_of_run_reports", lambda **_k: None)

    ctx = click.Context(click.Command("simlab"))
    with pytest.raises(SystemExit):
        run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")

    run_dirs = [p for p in (tmp_path / "output").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "crash.json").is_file()
    assert (run_dir / "best" / "scenario_prompt.md").is_file()


def test_run_autoresearch_retries_finalize_when_finalize_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    env_dir = tmp_path / "environments" / "env1"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: env1\nscenario_guidance_md: |\n  # Scenario Guidance\n  Baseline\n",
        encoding="utf-8",
    )

    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": "./tasks",
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

    call_count = {"count": 0}

    def fake_finalize_run(*, run_dir: Path, **_kwargs: object) -> None:
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise RuntimeError("boom")
        (run_dir / "finalize_retried.txt").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr("simlab.autoresearch.manager._finalize_run", fake_finalize_run)

    ctx = click.Context(click.Command("simlab"))
    with pytest.raises(RuntimeError):
        run_autoresearch(cfg=cfg, ctx=ctx, output_base_dir=tmp_path / "output")

    run_dirs = [p for p in (tmp_path / "output").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "crash.json").is_file()
    assert (run_dir / "finalize_retried.txt").is_file()
