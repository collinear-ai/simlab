"""Composition engine — generates docker-compose.yml from selected tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel
from pydantic import Field

from simlab.catalog.registry import ServiceDefinition
from simlab.catalog.registry import ToolDefinition
from simlab.catalog.registry import ToolRegistry
from simlab.composer.npc_config import inject_npc_env_vars

DEFAULT_IMAGE_REGISTRY = "ghcr.io/collinear-ai"


class EnvConfig(BaseModel):
    """User's environment configuration (written to simlab-env.yaml)."""

    name: str = "simlab-env"
    tools: list[str] = Field(default_factory=list)
    overrides: dict[str, dict[str, str]] = Field(default_factory=dict)
    scenario_guidance_md: str | None = None
    registry: str | None = DEFAULT_IMAGE_REGISTRY
    scenario_manager_api_url: str | None = None
    template: str | None = None


class ComposeOutput(BaseModel):
    """Output of the composition engine."""

    compose_yaml: str
    env_file: str
    tool_endpoints: dict[str, str]
    seed_services: list[str] = Field(default_factory=list)
    has_build_contexts: bool = False
    scenario_guidance_md: str | None = None


class ComposeEngine:
    """Generates docker-compose.yml and supporting files from tool selections."""

    NETWORK_NAME: ClassVar[str] = "simlab"
    FRAPPE_SCENARIO_SERVICE_NAMES: ClassVar[set[str]] = {"frappe-hrms-env", "frappe-hrms-seed"}

    def __init__(self, registry: ToolRegistry) -> None:
        """Store the tool registry for lookups."""
        self._registry = registry

    def compose(self, config: EnvConfig) -> ComposeOutput:
        """Build compose output for the given config."""
        tools = self._registry.get_tools(config.tools)

        services: dict[str, dict] = {}
        volumes: dict[str, dict] = {}
        claimed_ports: dict[int, str] = {}
        env_vars: dict[str, str] = {}
        tool_endpoints: dict[str, str] = {}
        preseed_service_names: list[str] = []
        seed_service_names: list[str] = []

        for tool in tools:
            if tool.is_external:
                # External tools have no Docker services — just record the URL
                tool_endpoints[tool.name] = f"{tool.tool_server_url}/tools"
                continue
            self._collect_services(tool, config, services, claimed_ports)
            self._collect_volumes(tool, volumes)
            self._collect_env_vars(tool, env_vars)
            tool_endpoints[tool.name] = f"http://localhost:{tool.tool_server_port}/tools"
            self._collect_profiled_services(
                tool_name=tool.name,
                service_defs=tool.preseed_services,
                config=config,
                services=services,
                profiled_service_names=preseed_service_names,
                profile_name="preseed",
            )
            self._collect_profiled_services(
                tool_name=tool.name,
                service_defs=tool.seed_services,
                config=config,
                services=services,
                profiled_service_names=seed_service_names,
                profile_name="seed",
            )

        # Inject NPC configs if rocketchat is present
        if "rocketchat" in config.tools:
            inject_npc_env_vars(services)

        compose = {
            "services": services,
            "networks": {self.NETWORK_NAME: {"driver": "bridge"}},
        }
        if volumes:
            compose["volumes"] = volumes

        compose_yaml = yaml.dump(compose, default_flow_style=False, sort_keys=False, width=10000)
        env_file = self._generate_env_file(env_vars)
        scenario_guidance_md = self._resolve_scenario_guidance(config)

        # Detect whether any service has a build context
        has_builds = any("build" in svc for svc in services.values())

        return ComposeOutput(
            compose_yaml=compose_yaml,
            env_file=env_file,
            tool_endpoints=tool_endpoints,
            seed_services=seed_service_names,
            has_build_contexts=has_builds,
            scenario_guidance_md=scenario_guidance_md,
        )

    @staticmethod
    def _rewrite_image(image: str, registry: str) -> str:
        """Rewrite collinear/* images to use a remote registry.

        ``collinear/email-env:latest`` → ``ghcr.io/collinear-ai/collinear/email-env:latest``

        Public images (not prefixed with ``collinear/``) are returned unchanged.
        """
        if not image.startswith("collinear/"):
            return image
        return f"{registry}/{image}"

    def _collect_services(
        self,
        tool: ToolDefinition,
        config: EnvConfig,
        services: dict[str, dict],
        claimed_ports: dict[int, str],
    ) -> None:
        overrides = config.overrides.get(tool.name, {})

        # Determine which ports should be host-mapped
        host_ports: set[str] = {str(tool.tool_server_port)} if tool.tool_server_port else set()
        for ep in tool.exposed_ports:
            host_ports.add(str(ep.port))

        for svc_name, svc_def in tool.services.items():
            image = svc_def.image
            if config.registry:
                image = self._rewrite_image(image, config.registry)

            svc: dict = {
                "image": image,
                "networks": [self.NETWORK_NAME],
            }

            # Emit build context so docker compose can build from source
            if svc_def.build:
                svc["build"] = svc_def.build

            # Port mappings: only tool_server and exposed_ports get host bindings
            ports = []
            for port_str in svc_def.ports:
                port_num = int(port_str.strip('"'))
                if str(port_num) in host_ports:
                    actual_port = self._resolve_port(port_num, svc_name, claimed_ports)
                    ports.append(f"{actual_port}:{port_num}")
                # Internal-only ports don't need explicit mapping on compose network
            if ports:
                svc["ports"] = ports

            # Environment
            env = dict(svc_def.environment)
            env.update(self._get_service_env(tool.name, svc_name, config))
            env.update(overrides)
            if env:
                svc["environment"] = env

            # Depends on (supports both list and dict-with-conditions format)
            if svc_def.depends_on:
                if isinstance(svc_def.depends_on, dict):
                    svc["depends_on"] = dict(svc_def.depends_on)
                else:
                    svc["depends_on"] = list(svc_def.depends_on)

            # Volumes
            if svc_def.volumes:
                svc["volumes"] = list(svc_def.volumes)

            # Command
            if svc_def.command:
                svc["command"] = svc_def.command

            # Healthcheck
            if svc_def.healthcheck:
                svc["healthcheck"] = dict(svc_def.healthcheck)

            services[svc_name] = svc

    def _collect_profiled_services(
        self,
        tool_name: str,
        service_defs: dict[str, ServiceDefinition],
        config: EnvConfig,
        services: dict[str, dict],
        profiled_service_names: list[str],
        profile_name: str,
    ) -> None:
        """Add profiled services so they only run on demand."""
        overrides = config.overrides.get(tool_name, {})
        for svc_name, svc_def in service_defs.items():
            image = svc_def.image
            if config.registry:
                image = self._rewrite_image(image, config.registry)

            svc: dict = {
                "image": image,
                "networks": [self.NETWORK_NAME],
                "profiles": [profile_name],
            }

            if svc_def.build:
                svc["build"] = svc_def.build

            env = dict(svc_def.environment)
            env.update(self._get_service_env(tool_name, svc_name, config))
            env.update(overrides)
            if env:
                svc["environment"] = env

            if svc_def.depends_on:
                if isinstance(svc_def.depends_on, dict):
                    svc["depends_on"] = dict(svc_def.depends_on)
                else:
                    svc["depends_on"] = list(svc_def.depends_on)

            if svc_def.volumes:
                svc["volumes"] = list(svc_def.volumes)

            if svc_def.command:
                svc["command"] = svc_def.command

            if svc_def.healthcheck:
                svc["healthcheck"] = dict(svc_def.healthcheck)

            services[svc_name] = svc
            profiled_service_names.append(svc_name)

    def _resolve_scenario_guidance(self, config: EnvConfig) -> str | None:
        """Resolve scenario guidance text from inline config."""
        if config.scenario_guidance_md:
            content = config.scenario_guidance_md.strip()
            return content or None
        return None

    def _collect_volumes(self, tool: ToolDefinition, volumes: dict[str, dict]) -> None:
        for vol_name, vol_def in tool.volumes.items():
            vol: dict = {}
            if vol_def.driver:
                vol["driver"] = vol_def.driver
            volumes[vol_name] = vol

    def _collect_env_vars(self, tool: ToolDefinition, env_vars: dict[str, str]) -> None:
        for var in tool.required_env_vars:
            env_vars[var] = ""
        # Scan service environments for ${VAR} references
        for svc_def in tool.services.values():
            for val in svc_def.environment.values():
                for match in re.findall(r"\$\{(\w+)\}", val):
                    if match not in env_vars:
                        env_vars[match] = ""

    def _get_service_env(self, tool_name: str, svc_name: str, config: EnvConfig) -> dict[str, str]:
        """Return compose-time env vars derived from environment config."""
        if (
            tool_name == "frappe-hrms"
            and config.template
            and svc_name in self.FRAPPE_SCENARIO_SERVICE_NAMES
        ):
            return {"FRAPPE_SEED_SCENARIO": config.template}
        return {}

    def _resolve_port(self, port: int, svc_name: str, claimed_ports: dict[int, str]) -> int:
        """Resolve port conflicts by incrementing."""
        actual = port
        while actual in claimed_ports and claimed_ports[actual] != svc_name:
            actual += 1
        claimed_ports[actual] = svc_name
        return actual

    def _generate_env_file(self, env_vars: dict[str, str]) -> str:
        if not env_vars:
            return "# No environment variables required\n"
        lines = ["# Fill in required environment variables", ""]
        for var, default in sorted(env_vars.items()):
            lines.append(f"{var}={default}")
        lines.append("")
        return "\n".join(lines)


def write_output(output: ComposeOutput, output_dir: Path) -> None:
    """Write composition output files to a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "docker-compose.yml").write_text(output.compose_yaml)
    (output_dir / ".env").write_text(output.env_file)
