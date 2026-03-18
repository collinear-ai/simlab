from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from simlab.cli import tasks as tasks_cli


def test_effective_tool_servers_merges_environment_endpoints() -> None:
    rewritten_task = {
        "tool_servers": [
            {"name": "frappe-hrms-env", "tool_server_url": "http://localhost:8030"},
            {"name": "chronos-server", "tool_server_url": "http://localhost:8050"},
        ]
    }
    endpoints = {
        "frappe-hrms": "http://localhost:8030",
        "calendar": "http://localhost:8050",
        "email": "http://localhost:8040",
        "rocketchat": "http://localhost:8060",
    }
    merged = tasks_cli._effective_tool_servers(rewritten_task, endpoints)
    assert "frappe-hrms-env" in merged
    assert "chronos-server" in merged
    assert merged["email-env"] == "http://localhost:8040"
    assert merged["rocketchat-env"] == "http://localhost:8060"


def test_rewrite_tool_server_urls_uses_service_name_mapping() -> None:
    task_data = {
        "tool_servers": [
            {
                "name": "coding-env",
                "tool_server_url": (
                    "http://k8s-default-rlgymser-c1fb54ba33-7ea806a8397f28db."
                    "elb.us-east-1.amazonaws.com:8020"
                ),
            },
            {"name": "google-workspace-tool-server", "tool_server_url": "http://legacy:8090"},
            {"name": "sec-edgar-env", "tool_server_url": "http://legacy:8070"},
            {"name": "twelve-data-env", "tool_server_url": "http://legacy:8080"},
        ]
    }
    endpoints = {
        "coding": "http://localhost:18020",
        "google-workspace": "http://localhost:18090",
        "sec-edgar": "http://localhost:18070",
        "twelve-data": "http://localhost:18080",
    }
    rewritten = tasks_cli._rewrite_tool_server_urls(task_data, endpoints)
    servers = {s["name"]: s["tool_server_url"] for s in rewritten["tool_servers"]}
    assert servers["coding-env"] == "http://localhost:18020"
    assert servers["google-workspace-tool-server"] == "http://localhost:18090"
    assert servers["sec-edgar-env"] == "http://localhost:18070"
    assert servers["twelve-data-env"] == "http://localhost:18080"


def test_seed_task_data_retries_failed_calls(monkeypatch) -> None:  # noqa: ANN001
    attempts = {"count": 0}

    def fake_query(url: str, tool_name: str, params: dict):
        _ = url, tool_name, params
        attempts["count"] += 1
        if attempts["count"] < 3:
            return None
        return {"ok": True}

    monkeypatch.setattr(tasks_cli, "query_tool_server", fake_query)

    task = {
        "seed_emails": [
            {
                "from_profile_id": "marcus_johnson",
                "to_addr": "hr@weaverenterprises.com",
                "subject": "Candidate Screening Request",
                "body_text": "hello",
            }
        ],
        "seed_calendar_events": [],
    }
    profiles = {"marcus_johnson": {"email": "marcus@example.com"}}
    endpoints = {"email": "http://localhost:8040"}
    ok, fail = tasks_cli._seed_task_data(task, profiles, endpoints)
    assert ok == 1
    assert fail == 0
    assert attempts["count"] == 3


def test_provision_task_calendar_users_only_runs_calendar_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = {
        "apps": ["calendar"],
        "npcs": [{"id": "patricia_lee"}],
        "seed_calendar_events": [{"account": "patricia_lee"}],
    }
    config = tasks_cli.EnvConfig(
        name="simlab-env",
        tools=["calendar", "frappe-hrms"],
        overrides={},
    )
    config_path = tmp_path / "simlab-env.yaml"
    config_path.write_text("name: simlab-env\n")
    expected_config_path = config_path

    calls: list[tuple[str, list[str], dict[str, str]]] = []

    def fake_services(
        _config: tasks_cli.EnvConfig,
        profile: str,
        config_path: Path,
        tool_names: list[str] | None = None,
    ) -> list[str]:
        assert config_path == expected_config_path
        assert tool_names == ["calendar"]
        return ["baikal-preseed"] if profile == "preseed" else ["baikal-seed"]

    def fake_run(
        _compose_dir: Path,
        svc_names: list[str],
        profile: str,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        calls.append((profile, svc_names, env_overrides or {}))

    monkeypatch.setattr("simlab.cli.env._get_profiled_service_names", fake_services)
    monkeypatch.setattr("simlab.cli.env._run_profiled_services_local", fake_run)

    tasks_cli._provision_task_calendar_users(
        task,
        config=config,
        config_path=str(config_path),
        using_daytona=False,
    )

    assert calls == [
        ("preseed", ["baikal-preseed"], {"CALDAV_USERS": "patricia_lee"}),
        ("seed", ["baikal-seed"], {"CALDAV_USERS": "patricia_lee"}),
    ]


def test_resolve_endpoints_does_not_auto_fallback_to_daytona(monkeypatch) -> None:  # noqa: ANN001
    config = tasks_cli.EnvConfig(name="x", tools=["email"], overrides={})

    monkeypatch.setattr(
        tasks_cli, "get_tool_endpoints", lambda _cfg: {"email": "http://localhost:8040"}
    )
    called = {"daytona": False}

    def fake_daytona(_path: str, daytona_api_key: str | None = None) -> dict[str, str]:
        _ = daytona_api_key
        called["daytona"] = True
        return {"email": "https://8040-x.daytonaproxy.net"}

    monkeypatch.setattr(tasks_cli, "_get_daytona_endpoints", fake_daytona)

    endpoints, using_daytona = tasks_cli._resolve_endpoints(
        config_path="my-env.yaml",
        config=config,
        daytona_requested=False,
        daytona_api_key=None,
    )
    assert using_daytona is False
    assert endpoints == {"email": "http://localhost:8040"}
    assert called["daytona"] is False


def test_require_reachable_endpoints_raises_when_none_reachable(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(tasks_cli, "_reachable_endpoints", lambda _e: {"email": False})
    try:
        tasks_cli._require_reachable_endpoints(
            endpoints={"email": "https://8040-x.daytonaproxy.net"},
            action="task run",
            using_daytona=True,
        )
        assert False, "Expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 1


def test_require_reachable_endpoints_local_error_mentions_local_debugging(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(tasks_cli, "_reachable_endpoints", lambda _e: {"coding": False})

    with pytest.raises(SystemExit) as exc_info:
        tasks_cli._require_reachable_endpoints(
            endpoints={"coding": "http://localhost:8020"},
            action="task run",
            using_daytona=False,
        )

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Local tool servers are not reachable." in captured.err
    assert "docker compose ps" in captured.err


def test_require_reachable_endpoints_local_error_hints_daytona_when_state_exists(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(tasks_cli, "_reachable_endpoints", lambda _e: {"coding": False})
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    config_path = env_dir / "env.yaml"
    config_path.write_text("name: env\n", encoding="utf-8")
    (env_dir / "daytona-state.json").write_text('{"sandbox_id":"sbx-1"}', encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        tasks_cli._require_reachable_endpoints(
            endpoints={"coding": "http://localhost:8020"},
            action="task run",
            using_daytona=False,
            config_path=str(config_path),
        )

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Local tool servers are not reachable." in captured.err
    assert "rerun with --daytona" in captured.err
    assert "daytona-state.json" in captured.err


def test_require_reachable_endpoints_passes_when_any_reachable(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(tasks_cli, "_reachable_endpoints", lambda _e: {"email": True})
    tasks_cli._require_reachable_endpoints(
        endpoints={"email": "http://localhost:8040"},
        action="task run",
        using_daytona=False,
    )


def test_build_services_available_section_local_defaults(tmp_path: Path) -> None:
    config = tasks_cli.EnvConfig(
        name="x",
        tools=["frappe-hrms", "rocketchat", "email", "calendar"],
        overrides={},
    )
    config_path = tmp_path / "my-env.yaml"
    config_path.write_text("name: x\n")
    section = tasks_cli._build_services_available_section(
        config,
        daytona=False,
        config_path=str(config_path),
        endpoints={},
    )
    assert section.startswith("Services available to you as websites:")
    assert "Frappe HRMS: http://localhost:8000" in section
    assert "Rocket.Chat: http://localhost:3000" in section
    assert "MailHog: http://localhost:8025" in section
    assert "Baikal Calendar: http://localhost:80" in section


def test_build_services_available_section_uses_compose_ports(tmp_path: Path) -> None:
    """Compose dir is parent of env.yaml (env-centric layout)."""
    config = tasks_cli.EnvConfig(
        name="my-env",
        tools=["frappe-hrms", "rocketchat", "email", "calendar"],
        overrides={},
    )
    env_dir = tmp_path / "my-env"
    env_dir.mkdir(parents=True, exist_ok=True)
    config_path = env_dir / "env.yaml"
    config_path.write_text("name: my-env\n")
    (env_dir / "docker-compose.yml").write_text(
        "\n".join(
            [
                "services:",
                "  frappe-hrms:",
                "    ports: ['18000:8000']",
                "  rocketchat:",
                "    ports: ['13000:3000']",
                "  mailhog:",
                "    ports: ['18025:8025']",
                "  baikal:",
                "    ports: ['10080:80']",
            ]
        )
    )
    section = tasks_cli._build_services_available_section(
        config,
        daytona=False,
        config_path=str(config_path),
        endpoints={},
    )
    assert "Frappe HRMS: http://localhost:18000" in section
    assert "Rocket.Chat: http://localhost:13000" in section
    assert "MailHog: http://localhost:18025" in section
    assert "Baikal Calendar: http://localhost:10080" in section


def test_build_services_available_section_daytona_endpoints(tmp_path: Path) -> None:
    config = tasks_cli.EnvConfig(name="x", tools=["frappe-hrms", "email"], overrides={})
    config_path = tmp_path / "my-env.yaml"
    config_path.write_text("name: x\n")
    section = tasks_cli._build_services_available_section(
        config,
        daytona=True,
        config_path=str(config_path),
        endpoints={
            "frappe-hrms": "https://8030-example.daytonaproxy.net",
            "email": "https://8040-example.daytonaproxy.net",
        },
    )
    assert section.startswith("Tool endpoints available to you:")
    assert "frappe-hrms: https://8030-example.daytonaproxy.net" in section
    assert "email: https://8040-example.daytonaproxy.net" in section


def test_get_daytona_endpoints_checks_status_inactive(monkeypatch) -> None:  # noqa: ANN001
    tmp = Path.cwd() / "tmp_test_daytona_inactive"
    tmp.mkdir(exist_ok=True)
    cfg = tmp / "env.yaml"
    cfg.write_text("name: x\ntools: [email]\noverrides: {}\n")
    (tmp / "daytona-state.json").write_text('{"sandbox_id":"sbx-1"}')

    class FakeSandbox:
        status = "stopped"

    class FakeDaytona:
        def get(self, sandbox_id: str):
            _ = sandbox_id
            return FakeSandbox()

    monkeypatch.setattr(tasks_cli.click, "confirm", lambda *_a, **_k: False)
    monkeypatch.setattr(tasks_cli, "_get_daytona_client", lambda *_args, **_kwargs: FakeDaytona())
    monkeypatch.setattr(
        tasks_cli,
        "ToolRegistry",
        lambda: SimpleNamespace(
            load_all=lambda: None,
            get_tool=lambda _n: SimpleNamespace(tool_server_port=8040, tool_server_url=None),
        ),
    )
    try:
        tasks_cli._get_daytona_endpoints(str(cfg))
        assert False, "Expected SystemExit for inactive sandbox"
    except SystemExit as exc:
        assert exc.code == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_get_daytona_endpoints_prompts_and_resumes(monkeypatch) -> None:  # noqa: ANN001
    tmp = Path.cwd() / "tmp_test_daytona_resume"
    tmp.mkdir(exist_ok=True)
    cfg = tmp / "env.yaml"
    cfg.write_text("name: x\ntools: [email]\noverrides: {}\n")
    (tmp / "daytona-state.json").write_text('{"sandbox_id":"sbx-1"}')

    class FakePreview:
        def __init__(self, url: str) -> None:
            self.url = url

    class FakeSandbox:
        def __init__(self, status: str) -> None:
            self.status = status

        def resume(self) -> None:
            self.status = "running"

        def get_preview_link(self, port: int):
            return FakePreview(f"https://{port}-x.daytonaproxy.net")

    class FakeDaytona:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.sandbox = FakeSandbox("stopped")

        def get(self, sandbox_id: str):
            _ = sandbox_id
            return self.sandbox

    class FakeRegistry:
        def load_all(self) -> None:
            return None

        def get_tool(self, name: str):
            if name == "email":
                return SimpleNamespace(tool_server_port=8040, tool_server_url=None)
            return None

    monkeypatch.setattr(tasks_cli.click, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(tasks_cli.time, "sleep", lambda _s: None)
    monkeypatch.setattr(tasks_cli, "_get_daytona_client", lambda *_args, **_kwargs: FakeDaytona())
    monkeypatch.setattr(tasks_cli, "ToolRegistry", FakeRegistry)
    endpoints = tasks_cli._get_daytona_endpoints(str(cfg))
    assert endpoints["email"].startswith("https://")
    shutil.rmtree(tmp, ignore_errors=True)


def test_get_daytona_endpoints_status_then_urls(monkeypatch) -> None:  # noqa: ANN001
    tmp = Path.cwd() / "tmp_test_daytona_active"
    tmp.mkdir(exist_ok=True)
    cfg = tmp / "env.yaml"
    cfg.write_text("name: x\ntools: [email]\noverrides: {}\n")
    (tmp / "daytona-state.json").write_text('{"sandbox_id":"sbx-1"}')

    class FakePreview:
        def __init__(self, url: str) -> None:
            self.url = url

    class FakeSandbox:
        status = "running"

        def get_preview_link(self, port: int):
            return FakePreview(f"https://{port}-x.daytonaproxy.net")

    class FakeDaytona:
        def get(self, sandbox_id: str):
            _ = sandbox_id
            return FakeSandbox()

    class FakeRegistry:
        def load_all(self) -> None:
            return None

        def get_tool(self, name: str):
            if name == "email":
                return SimpleNamespace(tool_server_port=8040, tool_server_url=None)
            return None

    monkeypatch.setattr(tasks_cli, "_get_daytona_client", lambda *_args, **_kwargs: FakeDaytona())
    monkeypatch.setattr(tasks_cli, "ToolRegistry", FakeRegistry)
    endpoints = tasks_cli._get_daytona_endpoints(str(cfg))
    assert endpoints["email"].startswith("https://")
    shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_agent_runtime_settings_uses_global_config_defaults() -> None:
    global_cfg = SimpleNamespace(
        agent_model="gpt-5",
        agent_provider="openai",
        agent_api_key="cfg-key",
        agent_base_url="https://llm.example.com/v1",
    )
    resolved = tasks_cli._resolve_agent_runtime_settings(
        global_cfg,
        model=None,
        provider=None,
        api_key=None,
        base_url=None,
    )
    assert resolved == ("gpt-5", "openai", "cfg-key", "https://llm.example.com/v1")


def test_resolve_agent_runtime_settings_uses_openai_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")
    global_cfg = SimpleNamespace(
        agent_model="gpt-5",
        agent_provider="openai",
        agent_api_key=None,
        agent_base_url="https://llm.example.com/v1",
    )

    resolved = tasks_cli._resolve_agent_runtime_settings(
        global_cfg,
        model=None,
        provider=None,
        api_key=None,
        base_url=None,
    )

    assert resolved == ("gpt-5", "openai", "openai-env-key", "https://llm.example.com/v1")


def test_apply_verifier_env_overrides_sets_and_restores(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("SIMLAB_VERIFIER_MODEL", "old-model")
    monkeypatch.delenv("SIMLAB_VERIFIER_API_KEY", raising=False)
    global_cfg = SimpleNamespace(
        verifier_model="new-model",
        verifier_provider="openai",
        verifier_base_url="https://judge.example.com/v1",
        verifier_api_key="judge-key",
    )

    original, applied = tasks_cli._apply_verifier_env_overrides(global_cfg)
    try:
        assert os.environ["SIMLAB_VERIFIER_MODEL"] == "new-model"
        assert os.environ["SIMLAB_VERIFIER_PROVIDER"] == "openai"
        assert os.environ["SIMLAB_VERIFIER_BASE_URL"] == "https://judge.example.com/v1"
        assert os.environ["SIMLAB_VERIFIER_API_KEY"] == "judge-key"
    finally:
        tasks_cli._restore_env(original, applied)

    assert os.environ["SIMLAB_VERIFIER_MODEL"] == "old-model"
    assert "SIMLAB_VERIFIER_API_KEY" not in os.environ
