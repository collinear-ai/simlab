from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from simlab.cli import tasks as tasks_cli
from simlab.runtime.daytona_runner import DaytonaNotFoundError
from simlab.runtime.env_lifecycle import ensure_daytona_sandbox_ready
from simlab.runtime.env_lifecycle import ensure_env_started_daytona
from simlab.runtime.env_lifecycle import ensure_env_started_local
from simlab.runtime.env_lifecycle import env_down_daytona
from simlab.runtime.env_lifecycle import env_down_local
from simlab.runtime.env_lifecycle import env_has_local_services
from simlab.runtime.env_lifecycle import is_env_running_daytona
from simlab.runtime.env_lifecycle import is_env_running_local
from simlab.runtime.env_lifecycle import run_env_seed_daytona
from simlab.runtime.env_lifecycle import run_env_seed_local


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
        "crm": "http://localhost:8045",
        "email": "http://localhost:8040",
        "erp": "http://localhost:8100",
        "rocketchat": "http://localhost:8060",
    }
    merged = tasks_cli._effective_tool_servers(rewritten_task, endpoints)
    assert "frappe-hrms-env" in merged
    assert "chronos-server" in merged
    assert merged["crm-env"] == "http://localhost:8045"
    assert merged["email-env"] == "http://localhost:8040"
    assert merged["erp-env"] == "http://localhost:8100"
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
            {"name": "crm-env", "tool_server_url": "http://legacy:8045"},
            {"name": "google-workspace-tool-server", "tool_server_url": "http://legacy:8090"},
            {"name": "sec-edgar-env", "tool_server_url": "http://legacy:8070"},
            {"name": "twelve-data-env", "tool_server_url": "http://legacy:8080"},
        ]
    }
    endpoints = {
        "coding": "http://localhost:18020",
        "crm": "http://localhost:18045",
        "google-workspace": "http://localhost:18090",
        "sec-edgar": "http://localhost:18070",
        "twelve-data": "http://localhost:18080",
    }
    rewritten = tasks_cli._rewrite_tool_server_urls(task_data, endpoints)
    servers = {s["name"]: s["tool_server_url"] for s in rewritten["tool_servers"]}
    assert servers["coding-env"] == "http://localhost:18020"
    assert servers["crm-env"] == "http://localhost:18045"
    assert servers["google-workspace-tool-server"] == "http://localhost:18090"
    assert servers["sec-edgar-env"] == "http://localhost:18070"
    assert servers["twelve-data-env"] == "http://localhost:18080"


def test_rewrite_tool_server_urls_maps_erp_service_name() -> None:
    task_data = {
        "tool_servers": [
            {"name": "erp-env", "tool_server_url": "http://erp-env:8100"},
            {"name": "email-env", "tool_server_url": "http://email-env:8040"},
        ]
    }
    endpoints = {
        "erp": "https://erp.preview",
        "email": "https://email.preview",
    }

    rewritten = tasks_cli._rewrite_tool_server_urls(task_data, endpoints)
    servers = {s["name"]: s["tool_server_url"] for s in rewritten["tool_servers"]}

    assert servers["erp-env"] == "https://erp.preview"
    assert servers["email-env"] == "https://email.preview"


def test_needed_task_endpoints_only_include_seed_tools_for_seeding() -> None:
    task = {
        "apps": ["calendar"],
        "tool_servers": [
            {"name": "coding-env", "tool_server_url": "http://legacy:8020"},
            {"name": "crm-env", "tool_server_url": "http://legacy:8045"},
        ],
        "seed_emails": [{"subject": "hello"}],
        "seed_calendar_events": [],
        "npcs": [{"id": "patricia_lee"}],
    }
    endpoints = {
        "calendar": "http://localhost:8050",
        "coding": "http://localhost:8020",
        "crm": "http://localhost:8045",
        "email": "http://localhost:8040",
    }

    assert tasks_cli.needed_task_endpoints(
        task,
        endpoints,
        include_tool_servers=False,
    ) == {
        "calendar": "http://localhost:8050",
        "email": "http://localhost:8040",
    }


def test_needed_task_endpoints_include_task_servers_for_run() -> None:
    task = {
        "apps": ["calendar"],
        "tool_servers": [
            {"name": "coding-env", "tool_server_url": "http://legacy:8020"},
            {"name": "crm-env", "tool_server_url": "http://legacy:8045"},
        ],
        "seed_emails": [{"subject": "hello"}],
        "seed_calendar_events": [],
        "npcs": [{"id": "patricia_lee"}],
    }
    endpoints = {
        "calendar": "http://localhost:8050",
        "coding": "http://localhost:8020",
        "crm": "http://localhost:8045",
        "email": "http://localhost:8040",
    }

    assert tasks_cli.needed_task_endpoints(
        task,
        endpoints,
        include_tool_servers=True,
    ) == {
        "calendar": "http://localhost:8050",
        "coding": "http://localhost:8020",
        "crm": "http://localhost:8045",
        "email": "http://localhost:8040",
    }


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


def test_seed_task_data_uses_literal_email_sender(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, str] = {}

    def fake_query(url: str, tool_name: str, params: dict):
        _ = url, tool_name
        captured["from_email"] = params["from_email"]
        return {"ok": True}

    monkeypatch.setattr(tasks_cli, "query_tool_server", fake_query)

    task = {
        "seed_emails": [
            {
                "from_profile_id": "procurement@brightpath.io",
                "to_addr": "agent@weaverenterprises.com",
                "subject": "Brightpath Solutions purchase request",
                "body_text": "hello",
            }
        ],
        "seed_calendar_events": [],
    }
    ok, fail = tasks_cli._seed_task_data(task, {}, {"email": "http://localhost:8040"})

    assert ok == 1
    assert fail == 0
    assert captured["from_email"] == "procurement@brightpath.io"


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

    monkeypatch.setattr("simlab.runtime.env_lifecycle._get_profiled_service_names", fake_services)
    monkeypatch.setattr("simlab.runtime.env_lifecycle._run_profiled_services_local", fake_run)

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


def test_run_env_seed_services_runs_seed_profile_for_local_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tasks_cli.EnvConfig(
        name="simlab-env",
        tools=["crm"],
        overrides={},
    )
    config_path = tmp_path / "simlab-env.yaml"
    config_path.write_text("name: simlab-env\n")
    expected_config_path = config_path

    calls: list[tuple[list[str], str, dict[str, str] | None]] = []

    def fake_services(
        _config: tasks_cli.EnvConfig,
        profile: str,
        config_path: Path,
        tool_names: list[str] | None = None,
    ) -> list[str]:
        assert config_path == expected_config_path
        assert profile == "seed"
        assert tool_names is None
        return ["crm-seed"]

    def fake_run(
        _compose_dir: Path,
        svc_names: list[str],
        profile: str,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        calls.append((svc_names, profile, env_overrides))

    monkeypatch.setattr("simlab.runtime.env_lifecycle._get_profiled_service_names", fake_services)
    monkeypatch.setattr("simlab.runtime.env_lifecycle._run_profiled_services_local", fake_run)

    tasks_cli.run_env_seed_services(
        config,
        str(config_path),
        using_daytona=False,
    )

    assert calls == [(["crm-seed"], "seed", None)]


def test_resolve_endpoints_does_not_auto_fallback_to_daytona(monkeypatch) -> None:  # noqa: ANN001
    config = tasks_cli.EnvConfig(name="x", tools=["email"], overrides={})

    monkeypatch.setattr(
        tasks_cli,
        "get_tool_endpoints",
        lambda _cfg, config_path=None: {"email": "http://localhost:8040"},
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


def test_require_reachable_endpoints_gateway_only_raises_when_gateway_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tasks_cli,
        "_reachable_endpoints",
        lambda _e: {tasks_cli.ComposeEngine.MCP_GATEWAY_SERVICE_NAME: False},
    )
    monkeypatch.setattr(tasks_cli, "_endpoint_is_reachable", lambda _u: False)

    with pytest.raises(SystemExit) as exc_info:
        tasks_cli._require_reachable_endpoints(
            endpoints={
                tasks_cli.ComposeEngine.MCP_GATEWAY_SERVICE_NAME: "http://localhost:8081/mcp"
            },
            action="task run",
            using_daytona=False,
        )

    assert exc_info.value.code == 1


def test_require_reachable_endpoints_gateway_only_passes_when_gateway_responds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tasks_cli,
        "_reachable_endpoints",
        lambda _e: {tasks_cli.ComposeEngine.MCP_GATEWAY_SERVICE_NAME: False},
    )
    monkeypatch.setattr(tasks_cli, "_endpoint_is_reachable", lambda _u: True)

    tasks_cli._require_reachable_endpoints(
        endpoints={tasks_cli.ComposeEngine.MCP_GATEWAY_SERVICE_NAME: "http://localhost:8081/mcp"},
        action="task run",
        using_daytona=False,
    )


def test_require_reachable_endpoints_raises_when_only_gateway_responds_in_mixed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tasks_cli,
        "_reachable_endpoints",
        lambda _e: {
            "email": False,
            tasks_cli.ComposeEngine.MCP_GATEWAY_SERVICE_NAME: False,
        },
    )
    monkeypatch.setattr(tasks_cli, "_endpoint_is_reachable", lambda _u: True)

    with pytest.raises(SystemExit) as exc_info:
        tasks_cli._require_reachable_endpoints(
            endpoints={
                "email": "http://localhost:8040",
                tasks_cli.ComposeEngine.MCP_GATEWAY_SERVICE_NAME: "http://localhost:8081/mcp",
            },
            action="task run",
            using_daytona=False,
        )

    assert exc_info.value.code == 1


def test_get_tool_endpoints_uses_mapped_gateway_port(tmp_path: Path) -> None:
    env_dir = tmp_path / "gateway-env"
    env_dir.mkdir()
    config_path = env_dir / "env.yaml"
    config_path.write_text("name: gateway-env\ntools: []\n", encoding="utf-8")
    (env_dir / "docker-compose.yml").write_text(
        "services:\n  mcp-gateway:\n    ports:\n      - 8081:8080\n",
        encoding="utf-8",
    )
    (env_dir / "mcp-servers.json").write_text(
        '{"mcpServers": {"weather": {"command": "uvx", "args": ["mcp-weather"]}}}',
        encoding="utf-8",
    )

    endpoints = tasks_cli.get_tool_endpoints(
        tasks_cli.EnvConfig(name="gateway-env", tools=[], overrides={}),
        config_path=config_path,
    )

    assert (
        endpoints[tasks_cli.ComposeEngine.MCP_GATEWAY_SERVICE_NAME] == "http://localhost:8081/mcp"
    )


def test_build_mcp_clients_preserves_command_server_identity() -> None:
    clients = tasks_cli._build_mcp_clients(
        {
            "mcpServers": {
                "weather": {"command": "uvx", "args": ["mcp-weather"]},
                "notion": {"command": "npx", "args": ["-y", "@notionhq/notion-mcp-server"]},
            }
        },
        {tasks_cli.ComposeEngine.MCP_GATEWAY_SERVICE_NAME: "http://localhost:8081/mcp"},
    )

    assert set(clients) == {"weather", "notion"}
    assert clients["weather"]._url == "http://localhost:8081/mcp"
    assert clients["weather"]._tool_prefix == "weather_"
    assert clients["notion"]._tool_prefix == "notion_"


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


def test_get_daytona_endpoints_without_state_uses_non_sandbox_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp = Path.cwd() / "tmp_test_daytona_external_only"
    tmp.mkdir(exist_ok=True)
    cfg = tmp / "env.yaml"
    cfg.write_text("name: x\ntools: []\noverrides: {}\n")
    monkeypatch.setattr(
        tasks_cli,
        "get_tool_endpoints",
        lambda _config, config_path=None: {"external-tool": "https://email.example.com"},
    )

    endpoints = tasks_cli._get_daytona_endpoints(str(cfg))

    assert endpoints == {"external-tool": "https://email.example.com"}
    shutil.rmtree(tmp, ignore_errors=True)


def test_get_daytona_endpoints_without_state_for_url_only_mcp_returns_empty(monkeypatch) -> None:  # noqa: ANN001
    tmp = Path.cwd() / "tmp_test_daytona_url_only_mcp"
    tmp.mkdir(exist_ok=True)
    cfg = tmp / "env.yaml"
    cfg.write_text("name: x\ntools: []\noverrides: {}\n")
    (tmp / "mcp-servers.json").write_text(
        '{"mcpServers": {"weather": {"url": "https://mcp.weather.example.com"}}}',
        encoding="utf-8",
    )

    class FakeRegistry:
        def load_all(self) -> None:
            return None

        def get_tool(self, name: str):
            _ = name

    monkeypatch.setattr(tasks_cli, "ToolRegistry", FakeRegistry)

    endpoints = tasks_cli._get_daytona_endpoints(str(cfg))

    assert endpoints == {}
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

    restart_calls: list[list[str]] = []

    class FakeRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def restart_sandbox_services(
            self,
            sandbox: object,
            *,
            preseed_svc_names: list[str] | None = None,
            log_prefix: str = "",
        ) -> None:
            _ = sandbox, log_prefix
            restart_calls.append(list(preseed_svc_names or []))

    monkeypatch.setattr(tasks_cli.click, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(tasks_cli.time, "sleep", lambda _s: None)
    monkeypatch.setattr(tasks_cli, "_get_daytona_client", lambda *_args, **_kwargs: FakeDaytona())
    monkeypatch.setattr(tasks_cli, "get_daytona_runner_class", lambda: FakeRunner)
    monkeypatch.setattr(
        tasks_cli,
        "get_env_runtime_helpers",
        lambda: (
            lambda _config, profile, _config_path, tool_names=None: ["email-preseed"]
            if profile == "preseed" and tool_names is None
            else [],
            lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(tasks_cli, "ToolRegistry", FakeRegistry)
    endpoints = tasks_cli._get_daytona_endpoints(str(cfg))
    assert endpoints["email"].startswith("https://")
    assert restart_calls == [["email-preseed"]]
    shutil.rmtree(tmp, ignore_errors=True)


def test_get_daytona_endpoints_resume_restart_failure_mentions_env_up(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp = Path.cwd() / "tmp_test_daytona_restart_failure"
    tmp.mkdir(exist_ok=True)
    cfg = tmp / "env.yaml"
    cfg.write_text("name: x\ntools: [email]\noverrides: {}\n")
    (tmp / "daytona-state.json").write_text('{"sandbox_id":"sbx-1"}')

    class FakeSandbox:
        def __init__(self) -> None:
            self.status = "stopped"

        def resume(self) -> None:
            self.status = "running"

        def get_preview_link(self, port: int):
            _ = port
            return SimpleNamespace(url="https://8040-x.daytonaproxy.net")

    class FakeDaytona:
        def __init__(self) -> None:
            self.sandbox = FakeSandbox()

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

    class FakeRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def restart_sandbox_services(
            self,
            sandbox: object,
            *,
            preseed_svc_names: list[str] | None = None,
            log_prefix: str = "",
        ) -> None:
            _ = sandbox, preseed_svc_names, log_prefix
            raise RuntimeError("docker compose up failed")

    monkeypatch.setattr(tasks_cli.click, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(tasks_cli.time, "sleep", lambda _s: None)
    monkeypatch.setattr(tasks_cli, "_get_daytona_client", lambda *_args, **_kwargs: FakeDaytona())
    monkeypatch.setattr(tasks_cli, "get_daytona_runner_class", lambda: FakeRunner)
    monkeypatch.setattr(
        tasks_cli,
        "get_env_runtime_helpers",
        lambda: (
            lambda _config, profile, _config_path, tool_names=None: ["email-preseed"]
            if profile == "preseed" and tool_names is None
            else [],
            lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(tasks_cli, "ToolRegistry", FakeRegistry)

    with pytest.raises(SystemExit) as exc_info:
        tasks_cli._get_daytona_endpoints(str(cfg))

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Unable to restart services in the resumed Daytona sandbox." in captured.err
    assert "simlab env up x --daytona" in captured.err
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


def test_get_daytona_endpoints_uses_mapped_gateway_port(monkeypatch) -> None:  # noqa: ANN001
    tmp = Path.cwd() / "tmp_test_daytona_gateway_port"
    tmp.mkdir(exist_ok=True)
    cfg = tmp / "env.yaml"
    cfg.write_text("name: x\ntools: [twelve-data]\noverrides: {}\n")
    (tmp / "daytona-state.json").write_text('{"sandbox_id":"sbx-1"}')
    (tmp / "docker-compose.yml").write_text(
        "services:\n  mcp-gateway:\n    ports:\n      - 8081:8080\n",
        encoding="utf-8",
    )
    (tmp / "mcp-servers.json").write_text(
        '{"mcpServers": {"weather": {"command": "uvx", "args": ["mcp-weather"]}}}',
        encoding="utf-8",
    )

    class FakePreview:
        def __init__(self, url: str) -> None:
            self.url = url

    class FakeSandbox:
        status = "running"

        def __init__(self) -> None:
            self.preview_ports: list[int] = []

        def get_preview_link(self, port: int):
            self.preview_ports.append(port)
            return FakePreview(f"https://{port}-x.daytonaproxy.net")

    fake_sandbox = FakeSandbox()

    class FakeDaytona:
        def get(self, sandbox_id: str):
            _ = sandbox_id
            return fake_sandbox

    class FakeRegistry:
        def load_all(self) -> None:
            return None

        def get_tool(self, name: str):
            if name == "twelve-data":
                return SimpleNamespace(tool_server_port=8080, tool_server_url=None)
            return None

    monkeypatch.setattr(tasks_cli, "_get_daytona_client", lambda *_args, **_kwargs: FakeDaytona())
    monkeypatch.setattr(tasks_cli, "ToolRegistry", FakeRegistry)

    endpoints = tasks_cli._get_daytona_endpoints(str(cfg))

    assert fake_sandbox.preview_ports == [8080, 8081]
    assert endpoints["twelve-data"] == "https://8080-x.daytonaproxy.net"
    assert endpoints[tasks_cli.ComposeEngine.MCP_GATEWAY_SERVICE_NAME] == (
        "https://8081-x.daytonaproxy.net/mcp"
    )
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


# ---------------------------------------------------------------------------
# _provision_task_group_channels
# ---------------------------------------------------------------------------


def test_provision_task_group_channels_runs_rocketchat_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify group channel provisioning runs rocketchat seed with correct env overrides."""
    task = {
        "npcs": [{"id": "diana_walsh"}, {"id": "james_foster"}],
        "seed_group_channels": [
            {
                "channel_name": "billing-support",
                "member_profile_ids": ["diana_walsh", "james_foster", "mike_williams"],
                "messages": [],
            }
        ],
    }
    profiles = {
        "diana_walsh": {"first_name": "Diana", "last_name": "Walsh", "email": "diana@example.com"},
        "james_foster": {
            "first_name": "James",
            "last_name": "Foster",
            "email": "james@example.com",
        },
    }
    config = tasks_cli.EnvConfig(
        name="simlab-env",
        tools=["rocketchat", "frappe-hrms"],
        overrides={},
    )
    config_path = tmp_path / "simlab-env.yaml"
    config_path.write_text("name: simlab-env\n")

    calls: list[tuple[str, list[str], dict[str, str]]] = []

    def fake_services(
        _config: tasks_cli.EnvConfig,
        profile: str,  # noqa: ARG001
        config_path: Path,  # noqa: ARG001
        tool_names: list[str] | None = None,
    ) -> list[str]:
        assert tool_names == ["rocketchat"]
        return ["rocketchat-seed"]

    def fake_run(
        _compose_dir: Path,
        svc_names: list[str],
        profile: str,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        calls.append((profile, svc_names, env_overrides or {}))

    monkeypatch.setattr("simlab.runtime.env_lifecycle._get_profiled_service_names", fake_services)
    monkeypatch.setattr("simlab.runtime.env_lifecycle._run_profiled_services_local", fake_run)

    tasks_cli._provision_task_group_channels(
        task,
        profiles,
        config=config,
        config_path=str(config_path),
        using_daytona=False,
    )

    assert len(calls) == 1
    profile, svc_names, env_overrides = calls[0]
    assert profile == "seed"
    assert svc_names == ["rocketchat-seed"]

    import json  # noqa: PLC0415

    npc_configs = json.loads(env_overrides["ROCKETCHAT_NPC_CONFIGS"])
    assert "diana_walsh" in npc_configs
    assert "james_foster" in npc_configs
    assert "mike_williams" in npc_configs  # from member_profile_ids
    assert npc_configs["diana_walsh"]["username"] == "diana_walsh"
    assert npc_configs["diana_walsh"]["email"] == "diana@example.com"
    assert npc_configs["diana_walsh"]["name"] == "Diana Walsh"
    # mike_williams not in profiles — should get fallback display name
    assert npc_configs["mike_williams"]["name"] == "Mike Williams"

    # Verify group channels were passed through
    group_channels = json.loads(env_overrides["ROCKETCHAT_SEED_GROUP_CHANNELS"])
    assert len(group_channels) == 1
    assert group_channels[0]["channel_name"] == "billing-support"


def test_provision_task_group_channels_skips_when_no_channels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No seed_group_channels means no seed container is run."""
    task = {"npcs": [{"id": "alice"}], "seed_group_channels": []}
    config = tasks_cli.EnvConfig(name="x", tools=["rocketchat"], overrides={})
    config_path = tmp_path / "env.yaml"
    config_path.write_text("name: x\n")

    called = {"yes": False}

    def fake_run(*_args: object, **_kwargs: object) -> None:
        called["yes"] = True

    monkeypatch.setattr("simlab.runtime.env_lifecycle._run_profiled_services_local", fake_run)

    tasks_cli._provision_task_group_channels(
        task, {}, config=config, config_path=str(config_path), using_daytona=False
    )
    assert not called["yes"]


def test_provision_task_group_channels_skips_when_no_rocketchat_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If rocketchat is not in config.tools, provisioning is skipped."""
    task = {
        "npcs": [],
        "seed_group_channels": [
            {"channel_name": "general", "member_profile_ids": ["bob"], "messages": []}
        ],
    }
    config = tasks_cli.EnvConfig(name="x", tools=["email"], overrides={})
    config_path = tmp_path / "env.yaml"
    config_path.write_text("name: x\n")

    called = {"yes": False}

    def fake_run(*_args: object, **_kwargs: object) -> None:
        called["yes"] = True

    monkeypatch.setattr("simlab.runtime.env_lifecycle._run_profiled_services_local", fake_run)

    tasks_cli._provision_task_group_channels(
        task, {}, config=config, config_path=str(config_path), using_daytona=False
    )
    assert not called["yes"]


def test_api_task_to_local_includes_seed_group_channels() -> None:
    """Verify _api_task_to_local copies seed_group_channels from API response."""
    api_task = {
        "task_id": "t1",
        "name": "Test",
        "description": "Do something",
        "tool_servers": [],
        "seed_emails": [],
        "seed_calendar_events": [],
        "seed_group_channels": [
            {"channel_name": "ops", "member_profile_ids": ["alice"], "messages": []}
        ],
        "npc_profiles": [],
        "verifier_modules": [],
    }
    task_dict, _profiles = tasks_cli._api_task_to_local(api_task)
    assert len(task_dict["seed_group_channels"]) == 1
    assert task_dict["seed_group_channels"][0]["channel_name"] == "ops"


def test_seed_command_waits_only_on_seed_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    env_dir = tmp_path / "environments" / "my-env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: my-env\ntools: [email, calendar, coding, crm]\n",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "bundle"
    tasks_dir = bundle_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / "task-1.json").write_text(
        json.dumps(
            {
                "meta": {
                    "task_id": "task-1",
                    "display_name": "Task 1",
                    "difficulty": "easy",
                    "category": "workflow",
                },
                "task": "Do the task.",
                "apps": ["calendar"],
                "tool_servers": [{"name": "coding-env", "tool_server_url": "http://legacy:8020"}],
                "seed_emails": [
                    {
                        "from_profile_id": "ops@example.com",
                        "to_addr": "agent@example.com",
                        "subject": "Hello",
                        "body_text": "Body",
                    }
                ],
                "seed_calendar_events": [],
                "seed_group_channels": [],
                "npcs": [{"id": "patricia_lee"}],
                "verifiers": [],
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_require(
        *,
        endpoints: dict[str, str],
        action: str,
        using_daytona: bool,
        config_path: str | None = None,
        wait: bool = False,
    ) -> None:
        captured["endpoints"] = endpoints
        captured["action"] = action
        captured["using_daytona"] = using_daytona
        captured["config_path"] = config_path
        captured["wait"] = wait

    monkeypatch.setattr(
        tasks_cli,
        "get_global_config_from_ctx",
        lambda _ctx: SimpleNamespace(daytona_api_key="daytona-key"),
    )
    monkeypatch.setattr(
        tasks_cli,
        "_resolve_endpoints",
        lambda **_kwargs: (
            {
                "email": "https://email.preview",
                "calendar": "https://calendar.preview",
                "coding": "https://coding.preview",
                "crm": "https://crm.preview",
            },
            True,
        ),
    )
    monkeypatch.setattr(tasks_cli, "_require_reachable_endpoints", fake_require)
    monkeypatch.setattr(tasks_cli, "_provision_task_group_channels", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "_provision_task_calendar_users", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "_ensure_task_calendar_accounts", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "_seed_task_data", lambda *args, **kwargs: (1, 0))

    runner = CliRunner()
    result = runner.invoke(
        tasks_cli.tasks,
        [
            "seed",
            "--env",
            "my-env",
            "--task",
            "task-1",
            "--tasks-dir",
            str(bundle_dir),
            "--daytona",
        ],
        env={
            "SIMLAB_DISABLE_TELEMETRY": "1",
            "SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments"),
        },
    )

    assert result.exit_code == 0, result.output
    assert captured["endpoints"] == {
        "email": "https://email.preview",
        "calendar": "https://calendar.preview",
    }
    assert captured["action"] == "task seeding"
    assert captured["using_daytona"] is True
    assert captured["wait"] is True


def test_run_command_waits_only_on_task_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    env_dir = tmp_path / "environments" / "my-env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.yaml").write_text(
        "name: my-env\ntools: [email, calendar, coding, crm]\n",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "bundle"
    tasks_dir = bundle_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / "task-1.json").write_text(
        json.dumps(
            {
                "meta": {
                    "task_id": "task-1",
                    "display_name": "Task 1",
                    "difficulty": "easy",
                    "category": "workflow",
                },
                "task": "Do the task.",
                "apps": ["calendar"],
                "tool_servers": [{"name": "coding-env", "tool_server_url": "http://legacy:8020"}],
                "seed_emails": [
                    {
                        "from_profile_id": "ops@example.com",
                        "to_addr": "agent@example.com",
                        "subject": "Hello",
                        "body_text": "Body",
                    }
                ],
                "seed_calendar_events": [],
                "seed_group_channels": [],
                "npcs": [{"id": "patricia_lee"}],
                "verifiers": [],
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

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

    def fake_require(
        *,
        endpoints: dict[str, str],
        action: str,
        using_daytona: bool,
        config_path: str | None = None,
        wait: bool = False,
    ) -> None:
        captured["endpoints"] = endpoints
        captured["action"] = action
        captured["using_daytona"] = using_daytona
        captured["config_path"] = config_path
        captured["wait"] = wait

    monkeypatch.setattr(
        tasks_cli,
        "get_global_config_from_ctx",
        lambda _ctx: SimpleNamespace(
            daytona_api_key="daytona-key",
            verifier_model="",
            verifier_provider="",
            verifier_base_url="",
            verifier_api_key="",
        ),
    )
    monkeypatch.setattr(
        tasks_cli,
        "resolve_scenario_manager_api_url",
        lambda *args, **kwargs: "https://api.example.com",
    )
    monkeypatch.setattr(tasks_cli, "resolve_collinear_api_key", lambda *args, **kwargs: "ck-test")
    monkeypatch.setattr(
        tasks_cli,
        "_resolve_agent_runtime_settings",
        lambda *args, **kwargs: ("test-model", "openai", None, None),
    )
    monkeypatch.setattr(
        tasks_cli,
        "_resolve_endpoints",
        lambda **_kwargs: (
            {
                "email": "https://email.preview",
                "calendar": "https://calendar.preview",
                "coding": "https://coding.preview",
                "crm": "https://crm.preview",
            },
            True,
        ),
    )
    monkeypatch.setattr(tasks_cli, "_require_reachable_endpoints", fake_require)
    monkeypatch.setattr(tasks_cli, "load_mcp_servers_from_env_dir", lambda _env_dir: None)
    monkeypatch.setattr(tasks_cli, "_build_mcp_clients", lambda *args, **kwargs: {})
    monkeypatch.setattr(tasks_cli, "_require_mcp_tools_available", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "run_env_seed_services", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "_provision_task_group_channels", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "_provision_task_calendar_users", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "_ensure_task_calendar_accounts", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks_cli, "_seed_task_data", lambda *args, **kwargs: (1, 0))
    monkeypatch.setattr(
        tasks_cli,
        "get_agent_runtime_helpers",
        lambda: (
            FakeEnvironment,
            lambda **kwargs: FakeArtifacts(),
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        tasks_cli.tasks,
        [
            "run",
            "--env",
            "my-env",
            "--task",
            "task-1",
            "--tasks-dir",
            str(bundle_dir),
            "--daytona",
            "--agent-import-path",
            "custom.agent:Agent",
            "--agent-model",
            "test-model",
        ],
        env={
            "SIMLAB_DISABLE_TELEMETRY": "1",
            "SIMLAB_ENVIRONMENTS_DIR": str(tmp_path / "environments"),
        },
    )

    assert result.exit_code == 0, result.output
    assert captured["endpoints"] == {
        "email": "https://email.preview",
        "calendar": "https://calendar.preview",
        "coding": "https://coding.preview",
    }
    assert captured["action"] == "task run"
    assert captured["using_daytona"] is True
    assert captured["wait"] is True


def test_env_has_local_services_false_when_no_compose(tmp_path: Path) -> None:
    assert env_has_local_services(tmp_path) is False


def test_env_has_local_services_false_for_empty_compose(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    assert env_has_local_services(tmp_path) is False


def test_env_has_local_services_true_when_services_defined(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yml").write_text("services:\n  web:\n    image: nginx\n")
    assert env_has_local_services(tmp_path) is True


def test_is_env_running_local_false_when_no_compose(tmp_path: Path) -> None:
    assert is_env_running_local(tmp_path) is False


def test_is_env_running_daytona_false_when_no_state_file(tmp_path: Path) -> None:
    assert is_env_running_daytona(tmp_path) is False


def test_is_env_running_daytona_false_when_bad_json(tmp_path: Path) -> None:
    (tmp_path / "daytona-state.json").write_text("not json")
    assert is_env_running_daytona(tmp_path) is False


def test_is_env_running_daytona_false_when_no_sandbox_id(tmp_path: Path) -> None:
    (tmp_path / "daytona-state.json").write_text("{}")
    assert is_env_running_daytona(tmp_path) is False


def test_lifecycle_imports_available() -> None:
    """Verify all lifecycle functions are callable."""
    for fn in (
        env_has_local_services,
        is_env_running_local,
        is_env_running_daytona,
        ensure_daytona_sandbox_ready,
        ensure_env_started_local,
        ensure_env_started_daytona,
        run_env_seed_local,
        run_env_seed_daytona,
        env_down_local,
        env_down_daytona,
    ):
        assert callable(fn)


# ---------------------------------------------------------------------------
# is_env_running_local — partial stack detection (P2)
# ---------------------------------------------------------------------------


def test_is_env_running_local_false_when_partial_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If only a subset of expected services are running, return False."""
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n"
        "  web:\n    image: nginx\n"
        "  db:\n    image: postgres\n"
        "  seed:\n    image: seed\n    profiles: [seed]\n"
    )
    # 2 non-profile services expected, but only 1 running container ID returned
    monkeypatch.setattr(
        "simlab.runtime.env_lifecycle.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(stdout="abc123\n", returncode=0),
    )
    assert is_env_running_local(tmp_path) is False


def test_is_env_running_local_true_when_all_services_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Return True when all non-profile services are running."""
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n"
        "  web:\n    image: nginx\n"
        "  db:\n    image: postgres\n"
        "  seed:\n    image: seed\n    profiles: [seed]\n"
    )
    # 2 non-profile services expected, 2 running container IDs returned
    monkeypatch.setattr(
        "simlab.runtime.env_lifecycle.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(stdout="abc123\ndef456\n", returncode=0),
    )
    assert is_env_running_local(tmp_path) is True


def test_is_env_running_local_false_when_no_containers_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Return False when compose exists but zero containers are running."""
    (tmp_path / "docker-compose.yml").write_text("services:\n  web:\n    image: nginx\n")
    monkeypatch.setattr(
        "simlab.runtime.env_lifecycle.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(stdout="", returncode=0),
    )
    assert is_env_running_local(tmp_path) is False


# ---------------------------------------------------------------------------
# ensure_daytona_sandbox_ready — resume / cleanup (P1)
# ---------------------------------------------------------------------------


def test_ensure_daytona_sandbox_ready_false_when_no_state(tmp_path: Path) -> None:
    assert ensure_daytona_sandbox_ready(tmp_path) is False


def test_ensure_daytona_sandbox_ready_true_when_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Active sandbox → return True without side effects."""
    (tmp_path / "daytona-state.json").write_text('{"sandbox_id": "sbx-1"}')
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    class FakeSandbox:
        status = "running"

    class FakeClient:
        def get(self, _sandbox_id: str) -> FakeSandbox:
            return FakeSandbox()

    fake_daytona_mod = SimpleNamespace(
        Daytona=lambda _cfg: FakeClient(),
        DaytonaConfig=lambda api_key: None,
    )
    monkeypatch.setattr(
        "simlab.runtime.env_lifecycle.import_module", lambda _name: fake_daytona_mod
    )
    assert ensure_daytona_sandbox_ready(tmp_path) is True
    # State file should still exist (no cleanup)
    assert (tmp_path / "daytona-state.json").exists()


def test_ensure_daytona_sandbox_ready_resumes_paused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Paused sandbox → resume it, return True."""
    (tmp_path / "daytona-state.json").write_text('{"sandbox_id": "sbx-1"}')
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    class FakeSandbox:
        def __init__(self) -> None:
            self.status = "paused"

        def resume(self) -> None:
            self.status = "running"

    sandbox = FakeSandbox()

    class FakeClient:
        def get(self, _sandbox_id: str) -> FakeSandbox:
            return sandbox

    fake_daytona_mod = SimpleNamespace(
        Daytona=lambda _cfg: FakeClient(),
        DaytonaConfig=lambda api_key: None,
    )
    monkeypatch.setattr(
        "simlab.runtime.env_lifecycle.import_module", lambda _name: fake_daytona_mod
    )
    monkeypatch.setattr("simlab.runtime.env_lifecycle.time.sleep", lambda _s: None)
    assert ensure_daytona_sandbox_ready(tmp_path) is True
    assert (tmp_path / "daytona-state.json").exists()


def test_ensure_daytona_sandbox_ready_cleans_up_on_failed_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Paused sandbox that cannot be resumed → delete it, remove state, return False."""
    (tmp_path / "daytona-state.json").write_text('{"sandbox_id": "sbx-1"}')
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")
    deleted = {"called": False}

    class FakeSandbox:
        status = "paused"
        # No resume method → resume will fail

    class FakeClient:
        def get(self, _sandbox_id: str) -> FakeSandbox:
            return FakeSandbox()

        def delete(self, _sandbox: FakeSandbox) -> None:
            deleted["called"] = True

    fake_daytona_mod = SimpleNamespace(
        Daytona=lambda _cfg: FakeClient(),
        DaytonaConfig=lambda api_key: None,
    )
    monkeypatch.setattr(
        "simlab.runtime.env_lifecycle.import_module", lambda _name: fake_daytona_mod
    )
    assert ensure_daytona_sandbox_ready(tmp_path) is False
    # Stale sandbox should have been deleted and state file removed
    assert deleted["called"]
    assert not (tmp_path / "daytona-state.json").exists()


def test_is_env_running_daytona_is_pure_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """is_env_running_daytona returns False for paused sandbox without side effects."""
    (tmp_path / "daytona-state.json").write_text('{"sandbox_id": "sbx-1"}')
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    class FakeSandbox:
        status = "paused"

    class FakeClient:
        def get(self, _sandbox_id: str) -> FakeSandbox:
            return FakeSandbox()

    fake_daytona_mod = SimpleNamespace(
        Daytona=lambda _cfg: FakeClient(),
        DaytonaConfig=lambda api_key: None,
    )
    monkeypatch.setattr(
        "simlab.runtime.env_lifecycle.import_module", lambda _name: fake_daytona_mod
    )
    # Pure check: returns False but does NOT delete or modify state
    assert is_env_running_daytona(tmp_path) is False
    assert (tmp_path / "daytona-state.json").exists()


def test_ensure_daytona_sandbox_ready_preserves_state_on_failed_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If sandbox delete fails, keep daytona-state.json for manual cleanup."""
    (tmp_path / "daytona-state.json").write_text('{"sandbox_id": "sbx-1"}')
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    class FakeSandbox:
        status = "paused"
        # No resume method → resume will fail

    class FakeClient:
        def get(self, _sandbox_id: str) -> FakeSandbox:
            return FakeSandbox()

        def delete(self, _sandbox: FakeSandbox) -> None:
            msg = "API timeout"
            raise TimeoutError(msg)

    fake_daytona_mod = SimpleNamespace(
        Daytona=lambda _cfg: FakeClient(),
        DaytonaConfig=lambda api_key: None,
    )
    monkeypatch.setattr(
        "simlab.runtime.env_lifecycle.import_module", lambda _name: fake_daytona_mod
    )
    assert ensure_daytona_sandbox_ready(tmp_path) is False
    # State file must be preserved because delete failed — needed for manual cleanup
    assert (tmp_path / "daytona-state.json").exists()


def test_get_daytona_endpoints_rejects_resume_when_disallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """allow_resume=False should fail fast on a non-active sandbox."""
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    config_path = env_dir / "env.yaml"
    config_path.write_text("name: x\ntools: [email]\noverrides: {}\n")
    (env_dir / "daytona-state.json").write_text('{"sandbox_id":"sbx-1"}')

    class FakeSandbox:
        status = "paused"

    class FakeDaytona:
        def get(self, _sandbox_id: str) -> FakeSandbox:
            return FakeSandbox()

    monkeypatch.setattr(tasks_cli, "_get_daytona_client", lambda *_args, **_kwargs: FakeDaytona())
    monkeypatch.setattr(
        tasks_cli,
        "ToolRegistry",
        lambda: SimpleNamespace(
            load_all=lambda: None,
            get_tool=lambda _n: SimpleNamespace(tool_server_port=8040, tool_server_url=None),
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        tasks_cli._get_daytona_endpoints(str(config_path), allow_resume=False)

    assert exc_info.value.code == 1


def test_is_env_running_local_false_when_docker_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Return False (not crash) when Docker CLI is not installed."""
    (tmp_path / "docker-compose.yml").write_text("services:\n  web:\n    image: nginx\n")

    def raise_oserror(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr("simlab.runtime.env_lifecycle.subprocess.run", raise_oserror)
    assert is_env_running_local(tmp_path) is False


def test_ensure_daytona_sandbox_ready_cleans_state_when_sandbox_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sandbox already deleted out-of-band → remove stale state file, return False."""
    (tmp_path / "daytona-state.json").write_text('{"sandbox_id": "sbx-gone"}')
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    class FakeClient:
        def get(self, _sandbox_id: str) -> None:
            raise DaytonaNotFoundError("sandbox not found")

    fake_daytona_mod = SimpleNamespace(
        Daytona=lambda _cfg: FakeClient(),
        DaytonaConfig=lambda api_key: None,
    )
    monkeypatch.setattr(
        "simlab.runtime.env_lifecycle.import_module", lambda _name: fake_daytona_mod
    )
    assert ensure_daytona_sandbox_ready(tmp_path) is False
    # State file should be removed — sandbox is gone, safe to create a new one
    assert not (tmp_path / "daytona-state.json").exists()


def test_ensure_daytona_sandbox_ready_preserves_state_when_lookup_is_uncertain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Transient lookup failure should keep state file to avoid losing pointer."""
    (tmp_path / "daytona-state.json").write_text('{"sandbox_id": "sbx-1"}')
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    class FakeClient:
        def get(self, _sandbox_id: str) -> None:
            raise TimeoutError("daytona api timeout")

    fake_daytona_mod = SimpleNamespace(
        Daytona=lambda _cfg: FakeClient(),
        DaytonaConfig=lambda api_key: None,
    )
    monkeypatch.setattr(
        "simlab.runtime.env_lifecycle.import_module", lambda _name: fake_daytona_mod
    )
    assert ensure_daytona_sandbox_ready(tmp_path) is False
    assert (tmp_path / "daytona-state.json").exists()


def test_ensure_daytona_sandbox_ready_cleans_state_when_corrupt(tmp_path: Path) -> None:
    """Corrupt state file → remove it, return False."""
    (tmp_path / "daytona-state.json").write_text("not json")
    assert ensure_daytona_sandbox_ready(tmp_path) is False
    assert not (tmp_path / "daytona-state.json").exists()
