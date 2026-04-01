"""Composition engine — generates docker-compose.yml from selected tools."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel
from pydantic import Field

from simlab.catalog.registry import BuildDefinition
from simlab.catalog.registry import ServiceDefinition
from simlab.catalog.registry import ToolDefinition
from simlab.catalog.registry import ToolRegistry
from simlab.composer.npc_config import NPC_CREDENTIALS_FILENAME
from simlab.composer.npc_config import NPC_INTERACTION_CONFIGS_FILENAME
from simlab.composer.npc_config import inject_npc_env_vars
from simlab.mcp_config import build_gateway_config_arena_format
from simlab.mcp_config import build_gateway_container_env
from simlab.mcp_config import get_mcp_command_servers
from simlab.mcp_config import load_mcp_servers_from_env_dir

DEFAULT_IMAGE_REGISTRY = "ghcr.io/collinear-ai"


def parse_template_id(template_id: str) -> tuple[str, str | None]:
    """Parse a template id into (template, version).

    Accepts either an unversioned template id like ``crm_sales`` or a versioned
    id like ``crm_sales:0.2.0``.
    """
    raw = template_id.strip()
    if not raw:
        raise ValueError("Template id cannot be empty")

    if ":" not in raw:
        return raw, None

    template, version = raw.split(":", 1)
    template = template.strip()
    version = version.strip()
    if not template or not version:
        raise ValueError(f"Invalid template id: {template_id!r}")
    if ":" in version:
        raise ValueError(f"Invalid template id version: {template_id!r}")
    return template, version


def pin_image_tag_for_template(image: str, template_id: str | None) -> str:
    """Pin collinear/*:latest images to the versioned template tag."""
    if not template_id:
        return image
    _, template_version = parse_template_id(template_id)
    if template_version is None:
        return image

    value = (image or "").strip()
    if not value or "@" in value or ":" not in value:
        return image
    if not (value.startswith("collinear/") or "/collinear/" in value):
        return image

    name, current_tag = value.rsplit(":", 1)
    if current_tag != "latest":
        return image
    return f"{name}:{template_version}"


class CodingMount(BaseModel):
    """A local file or directory mounted into the coding runtime."""

    source: str
    target: str
    read_only: bool = True


class CodingConfig(BaseModel):
    """Additional customization for the coding toolset."""

    setup_scripts: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    mounts: list[CodingMount] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class BundledAsset(BaseModel):
    """A local file or directory copied into the generated compose bundle."""

    source: str
    destination: str
    is_directory: bool = False


class EnvConfig(BaseModel):
    """User's environment configuration (written to simlab-env.yaml)."""

    name: str = "simlab-env"
    tools: list[str] = Field(default_factory=list)
    overrides: dict[str, dict[str, str]] = Field(default_factory=dict)
    coding: CodingConfig | None = None
    scenario_guidance_path: str | None = None
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
    bundled_assets: list[BundledAsset] = Field(default_factory=list)
    scenario_guidance_md: str | None = None
    mcp_gateway_config_json: str | None = None
    mcp_gateway_port: int | None = None
    npc_credentials_json: str | None = None
    npc_interaction_configs_json: str | None = None


class ComposeEngine:
    """Generates docker-compose.yml and supporting files from tool selections."""

    NETWORK_NAME: ClassVar[str] = "simlab"
    FRAPPE_SCENARIO_SERVICE_NAMES: ClassVar[set[str]] = {"frappe-hrms-env", "frappe-hrms-seed"}

    def __init__(self, registry: ToolRegistry) -> None:
        """Store the tool registry for lookups."""
        self._registry = registry

    MCP_GATEWAY_SERVICE_NAME: ClassVar[str] = "mcp-gateway"
    MCP_GATEWAY_PORT: ClassVar[int] = 8080
    MCP_GATEWAY_CONFIG_FILENAME: ClassVar[str] = "mcp-gateway-config.json"

    def compose(
        self,
        config: EnvConfig,
        config_dir: Path | None = None,
        env_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> ComposeOutput:
        """Build compose output for the given config."""
        resolved_config_dir = config_dir or Path.cwd()
        tools = self._registry.get_tools(config.tools)

        services: dict[str, dict] = {}
        volumes: dict[str, dict] = {}
        claimed_ports: dict[int, str] = {}
        env_vars: dict[str, str] = {}
        env_var_descriptions: dict[str, str] = {}
        tool_endpoints: dict[str, str] = {}
        preseed_service_names: list[str] = []
        seed_service_names: list[str] = []
        bundled_assets: list[BundledAsset] = []

        for tool in tools:
            if tool.is_external:
                # External tools have no Docker services — just record the URL
                tool_endpoints[tool.name] = f"{tool.tool_server_url}/tools"
                continue
            self._collect_services(tool, config, services, claimed_ports)
            self._collect_volumes(tool, volumes)
            self._collect_env_vars(tool, env_vars, env_var_descriptions)
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
        npc_credentials_json: str | None = None
        npc_interaction_configs_json: str | None = None
        if "rocketchat" in config.tools:
            resolved_output_dir = (output_dir or Path.cwd()).resolve().as_posix()
            npc_credentials_json, npc_interaction_configs_json = inject_npc_env_vars(
                services, resolved_output_dir
            )

        if "coding" in config.tools and config.coding is not None:
            bundled_assets.extend(
                self._apply_coding_config(
                    services=services,
                    config=config,
                    config_dir=resolved_config_dir,
                )
            )
        mcp_gateway_config_json: str | None = None
        mcp_gateway_port: int | None = None
        if env_dir is not None:
            mcp_config = load_mcp_servers_from_env_dir(env_dir)
            if mcp_config is not None:
                command_servers = get_mcp_command_servers(mcp_config)
                if command_servers:
                    resolved_env_dir = env_dir.resolve()
                    gateway_config = build_gateway_config_arena_format(command_servers)
                    mcp_gateway_config_json = json.dumps(gateway_config, indent=2)
                    mcp_gateway_port = self._resolve_port(
                        self.MCP_GATEWAY_PORT,
                        self.MCP_GATEWAY_SERVICE_NAME,
                        claimed_ports,
                    )
                    # Collect env var names from command servers for gateway container
                    gateway_env: dict[str, str] = {
                        "CONFIG_PATH": f"/config/{self.MCP_GATEWAY_CONFIG_FILENAME}",
                        "MCP_GATEWAY_CONFIG": mcp_gateway_config_json,
                        "MCP_GATEWAY_PORT": str(self.MCP_GATEWAY_PORT),
                    }
                    gateway_overrides, gateway_defaults = build_gateway_container_env(
                        command_servers
                    )
                    gateway_env.update(gateway_overrides)
                    env_vars.update(gateway_defaults)
                    gateway_config_source = (
                        resolved_env_dir / self.MCP_GATEWAY_CONFIG_FILENAME
                    ).as_posix()
                    services[self.MCP_GATEWAY_SERVICE_NAME] = {
                        "build": {"context": "./gateway", "dockerfile": "Dockerfile"},
                        "ports": [f"{mcp_gateway_port}:{self.MCP_GATEWAY_PORT}"],
                        "environment": gateway_env,
                        "volumes": [
                            f"{gateway_config_source}:/config/{self.MCP_GATEWAY_CONFIG_FILENAME}:ro"
                        ],
                        "networks": [self.NETWORK_NAME],
                    }
                    tool_endpoints[self.MCP_GATEWAY_SERVICE_NAME] = (
                        f"http://localhost:{mcp_gateway_port}/mcp"
                    )

        compose = {
            "services": services,
            "networks": {self.NETWORK_NAME: {"driver": "bridge"}},
        }
        if volumes:
            compose["volumes"] = volumes

        compose_yaml = yaml.dump(compose, default_flow_style=False, sort_keys=False, width=10000)
        env_file = self._generate_env_file(env_vars, env_var_descriptions)
        scenario_guidance_md = self._resolve_scenario_guidance(config, resolved_config_dir)

        # Detect whether any service has a build context
        has_builds = any("build" in svc for svc in services.values())

        return ComposeOutput(
            compose_yaml=compose_yaml,
            env_file=env_file,
            tool_endpoints=tool_endpoints,
            seed_services=seed_service_names,
            has_build_contexts=has_builds,
            bundled_assets=bundled_assets,
            scenario_guidance_md=scenario_guidance_md,
            mcp_gateway_config_json=mcp_gateway_config_json,
            mcp_gateway_port=mcp_gateway_port,
            npc_credentials_json=npc_credentials_json,
            npc_interaction_configs_json=npc_interaction_configs_json,
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
            image = pin_image_tag_for_template(image, config.template)

            svc: dict = {
                "image": image,
                "networks": [self.NETWORK_NAME],
            }

            # Emit build context so docker compose can build from source
            if svc_def.build:
                svc["build"] = self._serialize_build(svc_def.build)

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
            image = pin_image_tag_for_template(image, config.template)

            svc: dict = {
                "image": image,
                "networks": [self.NETWORK_NAME],
                "profiles": [profile_name],
            }

            if svc_def.build:
                svc["build"] = self._serialize_build(svc_def.build)

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

    def _apply_coding_config(
        self,
        *,
        services: dict[str, dict],
        config: EnvConfig,
        config_dir: Path,
    ) -> list[BundledAsset]:
        """Augment the coding runtime with startup hooks, skills, and mounts."""
        coding_cfg = config.coding
        if coding_cfg is None:
            return []

        agent_service = services.get("openhands-agent-server")
        if not isinstance(agent_service, dict):
            return []

        volumes = list(agent_service.get("volumes", []))
        environment = dict(agent_service.get("environment", {}))

        if coding_cfg.setup_scripts:
            for idx, raw_path in enumerate(coding_cfg.setup_scripts, start=1):
                source_path = self._resolve_local_path(raw_path, config_dir)
                if source_path.is_dir():
                    raise ValueError(f"Setup script must be a file, not a directory: {source_path}")
                volumes.append(
                    f"{self._compose_source_for(source_path, config_dir)}:"
                    f"/app/setup/{idx:02d}-{source_path.name}:ro"
                )

        if coding_cfg.skills:
            for raw_path in coding_cfg.skills:
                source_path = self._resolve_local_path(raw_path, config_dir)
                volumes.append(
                    f"{self._compose_source_for(source_path, config_dir)}:"
                    f"{self._skill_mount_target_for(source_path)}:ro"
                )

        if coding_cfg.mounts:
            for mount in coding_cfg.mounts:
                source_path = self._resolve_local_path(mount.source, config_dir)
                suffix = ":ro" if mount.read_only else ""
                volumes.append(
                    f"{self._compose_source_for(source_path, config_dir)}:{mount.target}{suffix}"
                )

        if coding_cfg.env:
            environment.update(coding_cfg.env)

        if volumes:
            agent_service["volumes"] = volumes
        if environment:
            agent_service["environment"] = environment

        return []

    @staticmethod
    def _resolve_local_path(raw_path: str, config_dir: Path) -> Path:
        """Resolve a config-authored path relative to the env config directory."""
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = config_dir / path
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Referenced coding asset does not exist: {resolved}")
        return resolved

    @staticmethod
    def _skill_mount_target_for(source_path: Path) -> str:
        """Map a skill file or directory into the OpenHands project skills path."""
        base = Path("/workspace/.openhands/skills")
        if source_path.name == "skills":
            return base.as_posix()
        if source_path.is_dir():
            return (base / source_path.name).as_posix()
        if source_path.name == "SKILL.md" and source_path.parent.name:
            return (base / source_path.parent.name / source_path.name).as_posix()
        return (base / source_path.name).as_posix()

    @staticmethod
    def _compose_source_for(source_path: Path, config_dir: Path) -> str:
        """Return the host-side compose volume source for a resolved local path."""
        _ = config_dir
        return source_path.as_posix()

    @classmethod
    def get_external_coding_asset_paths(cls, config: EnvConfig, config_dir: Path) -> list[Path]:
        """Return coding asset paths that live outside the env config directory."""
        coding_cfg = config.coding
        if coding_cfg is None:
            return []

        external_paths: list[Path] = []
        raw_paths = [
            *coding_cfg.setup_scripts,
            *coding_cfg.skills,
            *(mount.source for mount in coding_cfg.mounts),
        ]
        for raw_path in raw_paths:
            resolved = cls._resolve_local_path(raw_path, config_dir)
            try:
                resolved.relative_to(config_dir)
            except ValueError:
                external_paths.append(resolved)
        return external_paths

    def _resolve_scenario_guidance(self, config: EnvConfig, config_dir: Path) -> str | None:
        """Resolve scenario guidance text from inline config or a local file."""
        if config.scenario_guidance_md:
            content = config.scenario_guidance_md.strip()
            return content or None
        if config.scenario_guidance_path:
            source_path = self._resolve_local_path(config.scenario_guidance_path, config_dir)
            if source_path.is_dir():
                raise ValueError(
                    f"Scenario guidance path must be a file, not a directory: {source_path}"
                )
            content = source_path.read_text(encoding="utf-8").strip()
            return content or None
        return None

    def _collect_volumes(self, tool: ToolDefinition, volumes: dict[str, dict]) -> None:
        for vol_name, vol_def in tool.volumes.items():
            vol: dict = {}
            if vol_def.driver:
                vol["driver"] = vol_def.driver
            volumes[vol_name] = vol

    @staticmethod
    def _serialize_build(build: str | BuildDefinition) -> str | dict[str, str]:
        """Convert a typed build definition into docker-compose YAML data."""
        if isinstance(build, str):
            return build
        return build.model_dump(exclude_none=True)

    def _collect_env_vars(
        self,
        tool: ToolDefinition,
        env_vars: dict[str, str],
        env_var_descriptions: dict[str, str],
    ) -> None:
        for req in tool.required_env_vars:
            env_vars[req.name] = ""
            if req.description:
                env_var_descriptions[req.name] = req.description
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
            template, _ = parse_template_id(config.template)
            return {"FRAPPE_SEED_SCENARIO": template}
        return {}

    def _resolve_port(self, port: int, svc_name: str, claimed_ports: dict[int, str]) -> int:
        """Resolve port conflicts by incrementing."""
        actual = port
        while actual in claimed_ports and claimed_ports[actual] != svc_name:
            actual += 1
        claimed_ports[actual] = svc_name
        return actual

    def _generate_env_file(
        self,
        env_vars: dict[str, str],
        env_var_descriptions: dict[str, str] | None = None,
    ) -> str:
        if not env_vars:
            return "# No environment variables required\n"
        descriptions = env_var_descriptions or {}
        lines = ["# Fill in required environment variables", ""]
        for var, default in sorted(env_vars.items()):
            desc = descriptions.get(var)
            if desc:
                lines.append(f"# {desc}")
            lines.append(f"{var}={default}")
        lines.append("")
        return "\n".join(lines)


def _get_gateway_source_dir() -> Path:
    """Return the path to the gateway directory in the simlab package."""
    return Path(__file__).resolve().parent.parent / "gateway"


def get_mcp_gateway_host_port(env_dir: Path) -> int:
    """Return the mapped host port for the MCP gateway from docker-compose.yml."""
    compose_file = env_dir / "docker-compose.yml"
    if not compose_file.is_file():
        return ComposeEngine.MCP_GATEWAY_PORT
    compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    services = compose.get("services") if isinstance(compose, dict) else None
    gateway = (
        services.get(ComposeEngine.MCP_GATEWAY_SERVICE_NAME) if isinstance(services, dict) else None
    )
    ports = gateway.get("ports") if isinstance(gateway, dict) else None
    if not isinstance(ports, list):
        return ComposeEngine.MCP_GATEWAY_PORT
    for entry in ports:
        if isinstance(entry, str):
            parts = [part.strip() for part in entry.split(":")]
            if len(parts) >= 2 and parts[-1] == str(ComposeEngine.MCP_GATEWAY_PORT):
                try:
                    return int(parts[-2])
                except ValueError:
                    continue
    return ComposeEngine.MCP_GATEWAY_PORT


def write_output(output: ComposeOutput, output_dir: Path) -> None:
    """Write composition output files to a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "docker-compose.yml").write_text(output.compose_yaml)
    env_path = output_dir / ".env"
    if env_path.exists():
        env_path.replace(output_dir / ".env.bak")
    env_path.write_text(output.env_file)
    for asset in output.bundled_assets:
        src = Path(asset.source)
        dest = output_dir / asset.destination
        if asset.is_directory:
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    if output.npc_credentials_json is not None:
        (output_dir / NPC_CREDENTIALS_FILENAME).write_text(
            output.npc_credentials_json, encoding="utf-8"
        )
    if output.npc_interaction_configs_json is not None:
        (output_dir / NPC_INTERACTION_CONFIGS_FILENAME).write_text(
            output.npc_interaction_configs_json, encoding="utf-8"
        )
    if output.mcp_gateway_config_json is not None:
        (output_dir / ComposeEngine.MCP_GATEWAY_CONFIG_FILENAME).write_text(
            output.mcp_gateway_config_json, encoding="utf-8"
        )
        gateway_dest = output_dir / "gateway"
        gateway_dest.mkdir(parents=True, exist_ok=True)
        gateway_src = _get_gateway_source_dir()
        for name in ("Dockerfile", "requirements.txt", "run_gateway.py"):
            src_file = gateway_src / name
            if src_file.is_file():
                (gateway_dest / name).write_text(src_file.read_text(encoding="utf-8"))
