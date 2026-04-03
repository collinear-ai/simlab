from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from simlab.catalog.registry import BuildDefinition
from simlab.catalog.registry import EnvVarRequirement
from simlab.catalog.registry import ServiceDefinition
from simlab.catalog.registry import ToolDefinition
from simlab.catalog.registry import ToolRegistry
from simlab.composer.engine import DEFAULT_IMAGE_REGISTRY
from simlab.composer.engine import CodingConfig
from simlab.composer.engine import CodingMount
from simlab.composer.engine import ComposeEngine
from simlab.composer.engine import ComposeOutput
from simlab.composer.engine import EnvConfig
from simlab.composer.engine import write_output


def test_env_config_defaults_to_ghcr_registry() -> None:
    config = EnvConfig()
    assert config.registry == DEFAULT_IMAGE_REGISTRY


def test_rewrite_image_preserves_collinear_path_under_registry() -> None:
    rewritten = ComposeEngine._rewrite_image(
        "collinear/email-env:latest",
        DEFAULT_IMAGE_REGISTRY,
    )
    assert rewritten == "ghcr.io/collinear-ai/collinear/email-env:latest"


def test_compose_uses_default_registry_for_collinear_images() -> None:
    registry = ToolRegistry()
    registry._tools["email"] = ToolDefinition(
        name="email",
        display_name="Email",
        description="Email tool",
        category="communication",
        tool_server_port=8040,
        services={
            "email-env": ServiceDefinition(
                image="collinear/email-env:latest",
                ports=["8040"],
            )
        },
    )

    output = ComposeEngine(registry).compose(EnvConfig(tools=["email"]))

    assert "image: ghcr.io/collinear-ai/collinear/email-env:latest" in output.compose_yaml


def test_compose_preserves_list_environment_passthrough_entries() -> None:
    registry = ToolRegistry()
    registry._tools["email"] = ToolDefinition(
        name="email",
        display_name="Email",
        description="Email tool",
        category="communication",
        tool_server_port=8040,
        services={
            "email-env": ServiceDefinition(
                image="collinear/email-env:latest",
                ports=["8040"],
                environment=["OPENAI_API_KEY", "STATIC=value"],
            )
        },
    )

    output = ComposeEngine(registry).compose(
        EnvConfig(
            tools=["email"],
            overrides={"email": {"STATIC": "overridden"}},
        )
    )

    compose = yaml.safe_load(output.compose_yaml)
    env = compose["services"]["email-env"]["environment"]

    assert env == ["OPENAI_API_KEY", "STATIC=overridden"]
    assert "OPENAI_API_KEY=" in output.env_file


def test_compose_injects_frappe_seed_scenario_for_versioned_template_id() -> None:
    registry = ToolRegistry()
    registry.load_all()

    output = ComposeEngine(registry).compose(
        EnvConfig(
            tools=["frappe-hrms"],
            template="hr_people_management:1.0.0",
        )
    )

    compose = yaml.safe_load(output.compose_yaml)
    services = compose["services"]

    assert services["frappe-hrms-env"]["environment"]["FRAPPE_SEED_SCENARIO"] == (
        "hr_people_management"
    )
    assert services["frappe-hrms-seed"]["environment"]["FRAPPE_SEED_SCENARIO"] == (
        "hr_people_management"
    )


def test_compose_pins_latest_images_from_versioned_template_id() -> None:
    registry = ToolRegistry()
    registry._tools["email"] = ToolDefinition(
        name="email",
        display_name="Email",
        description="Email tool",
        category="communication",
        tool_server_port=8040,
        services={
            "email-env": ServiceDefinition(
                image="collinear/email-env:latest",
                ports=["8040"],
            )
        },
    )

    output = ComposeEngine(registry).compose(
        EnvConfig(
            tools=["email"],
            template="hr:1.2.3",
        )
    )

    assert "image: ghcr.io/collinear-ai/collinear/email-env:1.2.3" in output.compose_yaml


def test_compose_does_not_pin_images_for_unversioned_template_id() -> None:
    registry = ToolRegistry()
    registry._tools["email"] = ToolDefinition(
        name="email",
        display_name="Email",
        description="Email tool",
        category="communication",
        tool_server_port=8040,
        services={
            "email-env": ServiceDefinition(
                image="collinear/email-env:latest",
                ports=["8040"],
            )
        },
    )

    output = ComposeEngine(registry).compose(
        EnvConfig(
            tools=["email"],
            template="crm_sales",
        )
    )

    assert "image: ghcr.io/collinear-ai/collinear/email-env:latest" in output.compose_yaml


def test_compose_injects_frappe_seed_scenario_for_unversioned_template_id() -> None:
    registry = ToolRegistry()
    registry.load_all()

    output = ComposeEngine(registry).compose(
        EnvConfig(
            tools=["frappe-hrms"],
            template="hr_people_management",
        )
    )

    compose = yaml.safe_load(output.compose_yaml)
    services = compose["services"]

    assert services["frappe-hrms-env"]["environment"]["FRAPPE_SEED_SCENARIO"] == (
        "hr_people_management"
    )
    assert services["frappe-hrms-seed"]["environment"]["FRAPPE_SEED_SCENARIO"] == (
        "hr_people_management"
    )


def test_compose_loads_erp_catalog_tool() -> None:
    registry = ToolRegistry()
    registry.load_all()

    output = ComposeEngine(registry).compose(EnvConfig(tools=["erp"]))
    compose = yaml.safe_load(output.compose_yaml)
    services = compose["services"]

    assert services["erp-env"]["image"] == "ghcr.io/collinear-ai/collinear/erp-env:latest"
    assert services["erp-env"]["environment"]["ERP_FIXED_NOW"] == "2026-03-18T12:00:00+00:00"
    assert services["erp-env"]["environment"]["ERP_SEED_DATA_PATH"] == (
        "/app/src/toolsets/erp/seed_data/erp_seed_data.json"
    )
    assert services["erp-seed"]["image"] == "ghcr.io/collinear-ai/collinear/erp-env:latest"
    assert services["erp-seed"]["depends_on"]["erp-env"]["condition"] == "service_healthy"
    reset_command = services["erp-seed"]["command"][-1]
    assert "http://erp-env:8100/reset" in reset_command
    assert "method='POST'" in reset_command


def test_compose_loads_project_management_catalog_tool() -> None:
    registry = ToolRegistry()
    registry.load_all()

    output = ComposeEngine(registry).compose(EnvConfig(tools=["project-management"]))
    compose = yaml.safe_load(output.compose_yaml)
    services = compose["services"]

    assert services["project-management-env"]["image"] == (
        "ghcr.io/collinear-ai/collinear/project-management-env:latest"
    )
    assert services["project-management-env"]["environment"]["PROJECT_MANAGEMENT_FIXED_NOW"] == (
        "2026-03-20T12:00:00+00:00"
    )
    assert (
        services["project-management-env"]["environment"]["PROJECT_MANAGEMENT_SEED_DATA_PATH"]
        == "/app/src/toolsets/project_management/project_management_seed_data.json"
    )
    assert services["project-management-seed"]["image"] == (
        "ghcr.io/collinear-ai/collinear/project-management-env:latest"
    )
    assert (
        services["project-management-seed"]["depends_on"]["project-management-env"]["condition"]
        == "service_healthy"
    )
    reset_command = services["project-management-seed"]["command"][-1]
    assert "http://project-management-env:8110/reset" in reset_command
    assert "method='POST'" in reset_command


def test_compose_allows_frappe_seed_scenario_override() -> None:
    registry = ToolRegistry()
    registry.load_all()

    output = ComposeEngine(registry).compose(
        EnvConfig(
            tools=["frappe-hrms"],
            template="hr_people_management:1.0.0",
            overrides={"frappe-hrms": {"FRAPPE_SEED_SCENARIO": "custom_scenario"}},
        )
    )

    compose = yaml.safe_load(output.compose_yaml)
    services = compose["services"]

    assert services["frappe-hrms-env"]["environment"]["FRAPPE_SEED_SCENARIO"] == ("custom_scenario")
    assert services["frappe-hrms-seed"]["environment"]["FRAPPE_SEED_SCENARIO"] == (
        "custom_scenario"
    )


def test_compose_uses_inline_scenario_guidance_md() -> None:
    registry = ToolRegistry()
    registry._tools["email"] = ToolDefinition(
        name="email",
        display_name="Email",
        description="Email tool",
        category="communication",
        tool_server_port=8040,
        services={
            "email-env": ServiceDefinition(
                image="collinear/email-env:latest",
                ports=["8040"],
            )
        },
    )

    config = EnvConfig(
        tools=["email"],
        scenario_guidance_md="# Scenario Guidance\nUse the environment carefully.\n",
    )

    output = ComposeEngine(registry).compose(config)

    assert output.scenario_guidance_md == "# Scenario Guidance\nUse the environment carefully."


def test_rocketchat_seed_is_profiled_not_default_service() -> None:
    registry = ToolRegistry()
    registry.load_all()

    output = ComposeEngine(registry).compose(EnvConfig(tools=["rocketchat"]))
    compose = yaml.safe_load(output.compose_yaml)
    services = compose["services"]

    assert services["rocketchat-seed"]["profiles"] == ["seed"]
    assert "profiles" not in services["rocketchat"]


def test_compose_writes_scenario_guidance_bundle_file(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry._tools["email"] = ToolDefinition(
        name="email",
        display_name="Email",
        description="Email tool",
        category="communication",
        tool_server_port=8040,
        services={
            "email-env": ServiceDefinition(
                image="collinear/email-env:latest",
                ports=["8040"],
            )
        },
    )

    guidance = tmp_path / "skills.md"
    guidance.write_text("# Scenario Guidance\nUse the environment carefully.\n", encoding="utf-8")

    config = EnvConfig(
        tools=["email"],
        scenario_guidance_path=str(guidance),
    )

    output = ComposeEngine(registry).compose(config, config_dir=tmp_path)

    assert output.scenario_guidance_md == "# Scenario Guidance\nUse the environment carefully."


def test_compose_bundles_coding_assets(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.load_all()

    setup_script = tmp_path / "install-stateless-tools.sh"
    setup_script.write_text("#!/usr/bin/env bash\n")
    skill_dir = tmp_path / "pdf-xlsx-reporting"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# PDF/XLSX Reporting\nUse CLI tools.\n")
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "headcount.csv").write_text("department,count\neng,10\n")
    scenario_guidance = tmp_path / "skills.md"
    scenario_guidance.write_text("# Bundle Guidance\nUse coding skills first.\n")

    config = EnvConfig(
        tools=["coding"],
        scenario_guidance_path=str(scenario_guidance),
        coding=CodingConfig(
            setup_scripts=[str(setup_script)],
            skills=[str(skill_dir)],
            mounts=[
                CodingMount(
                    source=str(fixture_dir),
                    target="/workspace/fixtures",
                    read_only=True,
                )
            ],
            env={"GOOGLE_WORKSPACE_CLI_KEYRING_BACKEND": "file"},
        ),
    )

    output = ComposeEngine(registry).compose(config, config_dir=tmp_path)
    out_dir = tmp_path / "generated"
    write_output(output, out_dir)
    compose = yaml.safe_load(output.compose_yaml)
    openhands = compose["services"]["openhands-agent-server"]

    assert (
        f"{setup_script.as_posix()}:/app/setup/01-install-stateless-tools.sh:ro"
        in openhands["volumes"]
    )
    assert (
        f"{skill_dir.as_posix()}:/workspace/.openhands/skills/pdf-xlsx-reporting:ro"
        in openhands["volumes"]
    )
    assert f"{fixture_dir.as_posix()}:/workspace/fixtures:ro" in openhands["volumes"]
    assert openhands["environment"]["GOOGLE_WORKSPACE_CLI_KEYRING_BACKEND"] == "file"
    assert not output.bundled_assets


def test_write_output_moves_existing_env_to_backup(tmp_path: Path) -> None:
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    previous_env = "\n".join(["ACCUWEATHER_API_KEY=real-secret", "MCP_REGION=eu", ""])
    (out_dir / ".env").write_text(previous_env, encoding="utf-8")
    output = ComposeOutput(
        compose_yaml="services: {}\n",
        env_file="\n".join(
            [
                "# Fill in required environment variables",
                "",
                "ACCUWEATHER_API_KEY=demo-key",
                "MCP_REGION=us",
                "",
            ]
        ),
        tool_endpoints={},
    )

    write_output(output, out_dir)

    assert (out_dir / ".env.bak").read_text(encoding="utf-8") == previous_env
    assert (out_dir / ".env").read_text(encoding="utf-8") == "\n".join(
        [
            "# Fill in required environment variables",
            "",
            "ACCUWEATHER_API_KEY=demo-key",
            "MCP_REGION=us",
            "",
        ]
    )


def test_compose_preserves_command_server_env_defaults(tmp_path: Path) -> None:
    registry = ToolRegistry()
    env_dir = tmp_path / "mcp-env"
    env_dir.mkdir()
    (env_dir / "mcp-servers.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "weather": {
                        "command": "uvx",
                        "args": ["mcp-weather"],
                        "env": {
                            "ACCUWEATHER_API_KEY": "demo-key",
                            "MCP_REGION": "us",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    output = ComposeEngine(registry).compose(EnvConfig(tools=[]), env_dir=env_dir)
    compose = yaml.safe_load(output.compose_yaml)
    gateway_env = compose["services"]["mcp-gateway"]["environment"]
    gateway_config = json.loads(output.mcp_gateway_config_json or "{}")

    assert gateway_env == {
        "CONFIG_PATH": "/config/mcp-gateway-config.json",
        "MCP_GATEWAY_CONFIG": output.mcp_gateway_config_json,
        "MCP_GATEWAY_PORT": "8080",
        "SIMLAB_MCP_WEATHER__ACCUWEATHER_API_KEY": "${ACCUWEATHER_API_KEY}",
        "SIMLAB_MCP_WEATHER__MCP_REGION": "${MCP_REGION}",
    }
    assert gateway_config["servers"] == [
        {
            "name": "weather",
            "transport": "stdio",
            "command": "uvx mcp-weather",
            "env": {
                "ACCUWEATHER_API_KEY": "demo-key",
                "MCP_REGION": "us",
            },
        }
    ]
    assert "ACCUWEATHER_API_KEY=demo-key" in output.env_file
    assert "MCP_REGION=us" in output.env_file


def test_compose_uses_scoped_gateway_env_for_shared_command_server_keys(tmp_path: Path) -> None:
    registry = ToolRegistry()
    env_dir = tmp_path / "mcp-env"
    env_dir.mkdir()
    (env_dir / "mcp-servers.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "weather": {
                        "command": "uvx",
                        "args": ["mcp-weather"],
                        "env": {"API_KEY": "w-key"},
                    },
                    "docs": {"command": "uvx", "args": ["mcp-docs"], "env": {"API_KEY": "d-key"}},
                }
            }
        ),
        encoding="utf-8",
    )

    output = ComposeEngine(registry).compose(EnvConfig(tools=[]), env_dir=env_dir)
    compose = yaml.safe_load(output.compose_yaml)
    gateway_env = compose["services"]["mcp-gateway"]["environment"]
    env_lines = set(output.env_file.splitlines())

    assert gateway_env["SIMLAB_MCP_WEATHER__API_KEY"] == "${SIMLAB_MCP_WEATHER__API_KEY}"
    assert gateway_env["SIMLAB_MCP_DOCS__API_KEY"] == "${SIMLAB_MCP_DOCS__API_KEY}"
    assert "SIMLAB_MCP_WEATHER__API_KEY=w-key" in env_lines
    assert "SIMLAB_MCP_DOCS__API_KEY=d-key" in env_lines
    assert "API_KEY=w-key" not in env_lines
    assert "API_KEY=d-key" not in env_lines


def test_compose_uses_resolved_gateway_config_mount_for_symlinked_env_dir(tmp_path: Path) -> None:
    registry = ToolRegistry()
    real_env_dir = tmp_path / "real-env"
    real_env_dir.mkdir()
    symlink_env_dir = tmp_path / "linked-env"
    symlink_env_dir.symlink_to(real_env_dir, target_is_directory=True)
    (real_env_dir / "mcp-servers.json").write_text(
        json.dumps({"mcpServers": {"weather": {"command": "uvx", "args": ["mcp-weather"]}}}),
        encoding="utf-8",
    )

    output = ComposeEngine(registry).compose(EnvConfig(tools=[]), env_dir=symlink_env_dir)
    compose = yaml.safe_load(output.compose_yaml)
    gateway_volumes = compose["services"]["mcp-gateway"]["volumes"]

    mount_path = (real_env_dir / "mcp-gateway-config.json").as_posix()
    assert gateway_volumes == [f"{mount_path}:/config/mcp-gateway-config.json:ro"]


def test_compose_resolves_gateway_port_conflicts(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry._tools["market-data"] = ToolDefinition(
        name="market-data",
        display_name="Market Data",
        description="Market data tool",
        category="finance",
        tool_server_port=8080,
        services={
            "market-data-env": ServiceDefinition(
                image="collinear/market-data-env:latest",
                ports=["8080"],
            )
        },
    )
    env_dir = tmp_path / "mcp-env"
    env_dir.mkdir()
    (env_dir / "mcp-servers.json").write_text(
        json.dumps({"mcpServers": {"weather": {"command": "uvx", "args": ["mcp-weather"]}}}),
        encoding="utf-8",
    )

    output = ComposeEngine(registry).compose(EnvConfig(tools=["market-data"]), env_dir=env_dir)
    compose = yaml.safe_load(output.compose_yaml)

    assert compose["services"]["market-data-env"]["ports"] == ["8080:8080"]
    assert compose["services"]["mcp-gateway"]["ports"] == ["8081:8080"]
    assert output.mcp_gateway_port == 8081
    assert output.tool_endpoints["mcp-gateway"] == "http://localhost:8081/mcp"


def test_generate_env_file_includes_descriptions() -> None:
    """Env var descriptions from tool YAMLs appear as comments in the .env file."""
    registry = ToolRegistry()
    registry._tools["gws"] = ToolDefinition(
        name="gws",
        display_name="Google Workspace",
        description="Google tools",
        category="productivity",
        tool_server_port=8090,
        required_env_vars=[
            EnvVarRequirement(name="CREDENTIALS_CONFIG", description="Base64-encoded OAuth creds"),
            EnvVarRequirement(name="CREDENTIALS_EMAIL", description="Google account email"),
        ],
        services={
            "gws-env": ServiceDefinition(
                image="collinear/gws-env:latest",
                ports=["8090"],
                environment={
                    "CREDENTIALS_CONFIG": "${CREDENTIALS_CONFIG}",
                    "CREDENTIALS_EMAIL": "${CREDENTIALS_EMAIL}",
                },
            )
        },
    )

    output = ComposeEngine(registry).compose(EnvConfig(tools=["gws"]))

    assert "# Base64-encoded OAuth creds\nCREDENTIALS_CONFIG=" in output.env_file
    assert "# Google account email\nCREDENTIALS_EMAIL=" in output.env_file


def test_generate_env_file_omits_comment_for_undescribed_vars() -> None:
    """Env vars without descriptions are emitted without preceding comments."""
    registry = ToolRegistry()
    registry._tools["simple"] = ToolDefinition(
        name="simple",
        display_name="Simple",
        description="Simple tool",
        category="test",
        tool_server_port=9000,
        required_env_vars=[
            EnvVarRequirement(name="API_KEY"),
        ],
        services={
            "simple-env": ServiceDefinition(
                image="collinear/simple:latest",
                ports=["9000"],
                environment={"API_KEY": "${API_KEY}"},
            )
        },
    )

    output = ComposeEngine(registry).compose(EnvConfig(tools=["simple"]))

    assert "API_KEY=" in output.env_file
    lines = output.env_file.splitlines()
    api_key_idx = next(i for i, line in enumerate(lines) if line.startswith("API_KEY="))
    # The line before the variable should be blank, not a description comment
    assert lines[api_key_idx - 1] == ""


def test_google_workspace_env_vars_have_descriptions() -> None:
    """Verify google-workspace.yaml loads descriptions into EnvVarRequirement objects."""
    registry = ToolRegistry()
    registry.load_all()

    tool = registry.get_tool("google-workspace")
    assert tool is not None
    descriptions = {req.name: req.description for req in tool.required_env_vars}
    assert descriptions["CREDENTIALS_CONFIG"]
    assert descriptions["CREDENTIALS_EMAIL"]
    assert descriptions["DRIVE_NAMESPACE_ROOT_FOLDER_ID"]


def test_tool_definition_assigns_image_for_build_only_service() -> None:
    tool = ToolDefinition(
        name="harbor-main",
        display_name="Harbor Main",
        description="Harbor task runtime",
        category="custom",
        tool_server_port=9000,
        services={
            "harbor-main-env": ServiceDefinition(
                build=BuildDefinition(
                    context="./gateway",
                    dockerfile="Dockerfile",
                ),
                ports=["9000"],
            )
        },
    )

    assert tool.services["harbor-main-env"].build == BuildDefinition(
        context="./gateway",
        dockerfile="Dockerfile",
    )
    assert tool.services["harbor-main-env"].image == "harbor-main-env:latest"


def test_registry_load_all_rejects_env_local_duplicate_tool_name(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    custom_tools_dir = env_dir / "custom-tools"
    custom_tools_dir.mkdir(parents=True)
    (custom_tools_dir / "email.yaml").write_text(
        "\n".join(
            [
                "name: email",
                "display_name: Shadow Email",
                "description: Duplicate tool name",
                "category: custom",
                "tool_server_url: http://localhost:9000",
                "",
            ]
        ),
        encoding="utf-8",
    )

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Duplicate tool definition for 'email'"):
        registry.load_all(env_dir=env_dir)
