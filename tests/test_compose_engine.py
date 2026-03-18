from __future__ import annotations

from pathlib import Path

import yaml
from simlab.catalog.registry import ServiceDefinition
from simlab.catalog.registry import ToolDefinition
from simlab.catalog.registry import ToolRegistry
from simlab.composer.engine import DEFAULT_IMAGE_REGISTRY
from simlab.composer.engine import CodingConfig
from simlab.composer.engine import CodingMount
from simlab.composer.engine import ComposeEngine
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


def test_compose_injects_frappe_seed_scenario_for_template() -> None:
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


def test_compose_allows_frappe_seed_scenario_override() -> None:
    registry = ToolRegistry()
    registry.load_all()

    output = ComposeEngine(registry).compose(
        EnvConfig(
            tools=["frappe-hrms"],
            template="hr_people_management",
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
        "./install-stateless-tools.sh:/app/setup/01-install-stateless-tools.sh:ro"
        in openhands["volumes"]
    )
    assert (
        "./pdf-xlsx-reporting:/workspace/.openhands/skills/pdf-xlsx-reporting:ro"
        in openhands["volumes"]
    )
    assert "./fixtures:/workspace/fixtures:ro" in openhands["volumes"]
    assert openhands["environment"]["GOOGLE_WORKSPACE_CLI_KEYRING_BACKEND"] == "file"
    assert not output.bundled_assets
