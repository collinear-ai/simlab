from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from simlab.composer.engine import EnvConfig
from simlab.runtime import rollout_runner


class FakeArtifacts:
    def __init__(self) -> None:
        self.metadata: dict[str, object] = {}
        self.error: str | None = None
        self.messages: list[dict[str, object]] = []
        self.steps_taken = 0
        self.tool_calls: list[object] = []

    def dump(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")


class FakeEnvironment:
    def __init__(
        self,
        *,
        tool_servers: dict[str, str],
        mcp_clients: dict[str, object] | None = None,
    ) -> None:
        self.tool_servers = tool_servers
        self.mcp_clients = mcp_clients


class FakeCtx:
    def __init__(self) -> None:
        self.on_step_calls: list[int] = []

    def detail(self, _message: str) -> None:
        return

    def flush_buffer(self) -> None:
        return

    def update(self, _message: str) -> None:
        return

    def on_step(self, steps_taken: int) -> None:
        self.on_step_calls.append(steps_taken)


class FakeProgress:
    def __init__(self, ctx: FakeCtx) -> None:
        self._ctx = ctx

    @contextmanager
    def step(
        self,
        _label: str,
        *,
        success_label: str | None = None,
    ) -> Generator[FakeCtx, None, None]:
        _ = success_label
        yield self._ctx


def test_run_single_rollout_forwards_step_callbacks_to_ctx_on_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctx = FakeCtx()
    progress = FakeProgress(ctx)

    def fake_resolve_endpoints(**_kwargs: object):
        return {"email": "http://localhost:8040"}, False

    def fake_run_with_agent_contract(**kwargs: object):
        on_step = kwargs.get("on_step")
        artifacts = FakeArtifacts()
        if callable(on_step):
            on_step(1)
            artifacts.steps_taken = 1
        return artifacts

    monkeypatch.setattr(rollout_runner, "resolve_endpoints", fake_resolve_endpoints)
    monkeypatch.setattr(
        rollout_runner.harbor_urls,
        "rewrite_mcp_config_for_runtime",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(rollout_runner, "load_mcp_servers_from_env_dir", lambda _env_dir: None)
    monkeypatch.setattr(rollout_runner, "build_mcp_clients", lambda *a, **k: {})
    monkeypatch.setattr(rollout_runner, "needed_task_endpoints", lambda *a, **k: {})
    monkeypatch.setattr(rollout_runner, "require_reachable_endpoints", lambda *a, **k: None)
    monkeypatch.setattr(rollout_runner, "require_mcp_tools_available", lambda *a, **k: None)
    monkeypatch.setattr(rollout_runner.NpcChatSession, "from_task_data", lambda *a, **k: None)
    monkeypatch.setattr(rollout_runner, "load_skills_markdown", lambda *a, **k: "")
    monkeypatch.setattr(rollout_runner, "build_skills_guidance_section", lambda *_a, **_k: "")
    monkeypatch.setattr(rollout_runner, "build_services_available_section", lambda *a, **k: "")
    monkeypatch.setattr(
        rollout_runner,
        "get_agent_runtime_helpers",
        lambda: (FakeEnvironment, fake_run_with_agent_contract),
    )

    config = EnvConfig(name="env", tools=["email"])
    task_data = {
        "meta": {"task_id": "task-1", "display_name": "Task 1"},
        "task": "Do the task.",
        "tool_servers": [{"name": "email-env", "tool_server_url": "http://legacy:8040"}],
        "seed_emails": [],
        "seed_calendar_events": [],
        "seed_group_channels": [],
        "npcs": [],
        "verifiers": [],
    }

    global_cfg = SimpleNamespace(
        daytona_api_key="",
        verifier_model="",
        verifier_provider="",
        verifier_base_url="",
        verifier_api_key="",
    )

    outcome = rollout_runner.run_single_rollout(
        env_dir=tmp_path,
        config=config,
        config_path=str(tmp_path / "env.yaml"),
        global_cfg=global_cfg,
        task_data=task_data,
        task_id="task-1",
        profiles={},
        meta={"task_id": "task-1"},
        display="Task 1",
        model="test-model",
        provider="openai",
        api_key="test-key",
        base_url=None,
        max_steps=3,
        agent_import_path=None,
        agent_timeout_seconds=1.0,
        no_seed=True,
        verbose=False,
        daytona=False,
        tasks_dir=str(tmp_path / "bundle"),
        bundle_dir=None,
        backend_id=None,
        base_url_api="https://api.example.com",
        scenario_manager_api_key=None,
        managed_env=False,
        keep_alive=False,
        skip_env_setup=True,
        rollout_format=rollout_runner.ROLLOUT_FORMAT_DEFAULT,
        progress=progress,
        output_root=tmp_path / "output",
    )

    assert outcome.exit_code == 0
    assert ctx.on_step_calls == [1]
