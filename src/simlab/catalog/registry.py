"""Tool catalog registry — loads tool definitions from YAML files."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class ExposedPort(BaseModel):
    """A port exposed to the host for web UI access."""

    port: int
    description: str = ""


class EnvVarRequirement(BaseModel):
    """A required environment variable with an optional user-facing description."""

    model_config = {"extra": "forbid"}

    name: str
    description: str = ""


class BuildDefinition(BaseModel):
    """Supported docker-compose build configuration for catalog services."""

    model_config = {"extra": "forbid"}

    context: str
    dockerfile: str | None = None


class ServiceDefinition(BaseModel):
    """A single docker-compose service."""

    image: str = ""
    build: str | BuildDefinition | None = None
    ports: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    depends_on: list[str] | dict[str, Any] = Field(default_factory=list)
    volumes: list[str] = Field(default_factory=list)
    command: list[str] | str | None = None
    healthcheck: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_image_or_build(self) -> ServiceDefinition:
        if not self.image and not self.build:
            raise ValueError("Service must specify either 'image' or 'build'")
        return self


class VolumeDefinition(BaseModel):
    """A named docker volume."""

    driver: str | None = None


class ToolDefinition(BaseModel):
    """Complete definition of a tool server and its dependencies."""

    name: str
    display_name: str
    description: str
    category: str
    tool_server_port: int | None = None
    tool_server_url: str | None = None
    exposed_ports: list[ExposedPort] = Field(default_factory=list)
    required_env_vars: list[EnvVarRequirement] = Field(default_factory=list)
    services: dict[str, ServiceDefinition] = Field(default_factory=dict)
    preseed_services: dict[str, ServiceDefinition] = Field(default_factory=dict)
    seed_services: dict[str, ServiceDefinition] = Field(default_factory=dict)
    volumes: dict[str, VolumeDefinition] = Field(default_factory=dict)

    @property
    def is_external(self) -> bool:
        """True when the tool points at an external URL with no Docker services."""
        return bool(self.tool_server_url and not self.services)

    @model_validator(mode="after")
    def _check_server_config(self) -> ToolDefinition:
        has_docker = bool(self.tool_server_port and self.services)
        has_url = bool(self.tool_server_url)
        if not has_docker and not has_url:
            raise ValueError(
                f"Tool '{self.name}' must define either "
                "(tool_server_port + services) or tool_server_url"
            )
        if has_docker and has_url and not self.services:
            raise ValueError(
                f"Tool '{self.name}' has tool_server_url — "
                "remove services/tool_server_port or remove tool_server_url"
            )
        # Auto-generate image tag for services that only have a build context
        for service_group in (self.services, self.preseed_services, self.seed_services):
            for svc_name, svc_def in service_group.items():
                if svc_def.build and not svc_def.image:
                    svc_def.image = f"{svc_name}:latest"
        return self


def _parse_env_var_requirements(raw: list[str | dict[str, str]]) -> list[EnvVarRequirement]:
    """Parse mixed-format required_env_vars: plain strings or {name, description} dicts."""
    result: list[EnvVarRequirement] = []
    for item in raw:
        if isinstance(item, str):
            result.append(EnvVarRequirement(name=item))
        elif isinstance(item, dict):
            result.append(EnvVarRequirement(**item))
        else:
            raise TypeError(f"required_env_vars item must be a string or dict, got {type(item)}")
    return result


def _parse_tool_yaml(data: dict[str, Any]) -> ToolDefinition:
    """Parse a raw YAML dict into a ToolDefinition."""
    services = {}
    for svc_name, svc_data in data.get("services", {}).items():
        services[svc_name] = ServiceDefinition(**svc_data)

    preseed_services = {}
    for svc_name, svc_data in data.get("preseed_services", {}).items():
        preseed_services[svc_name] = ServiceDefinition(**svc_data)

    seed_services = {}
    for svc_name, svc_data in data.get("seed_services", {}).items():
        seed_services[svc_name] = ServiceDefinition(**svc_data)

    volumes = {}
    for vol_name, vol_data in data.get("volumes", {}).items():
        vol_spec = {} if vol_data is None else vol_data
        volumes[vol_name] = VolumeDefinition(**vol_spec)

    return ToolDefinition(
        name=data["name"],
        display_name=data["display_name"],
        description=data["description"],
        category=data["category"],
        tool_server_port=data.get("tool_server_port"),
        tool_server_url=data.get("tool_server_url"),
        exposed_ports=[ExposedPort(**ep) for ep in data.get("exposed_ports", [])],
        required_env_vars=_parse_env_var_requirements(data.get("required_env_vars", [])),
        services=services,
        preseed_services=preseed_services,
        seed_services=seed_services,
        volumes=volumes,
    )


class ToolRegistry:
    """Registry of all available tool servers."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, ToolDefinition] = {}

    def _register_tool(self, tool: ToolDefinition, *, source: str) -> None:
        """Register one tool, rejecting duplicates."""
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool definition for '{tool.name}' from {source}")
        self._tools[tool.name] = tool

    def load_all(self, *, env_dir: Path | None = None) -> None:
        """Load all tool YAML files from the catalog/tools package directory."""
        self._tools = {}
        tools_pkg = resources.files("simlab.catalog") / "tools"
        for item in tools_pkg.iterdir():
            if hasattr(item, "name") and item.name.endswith(".yaml"):
                text = item.read_text(encoding="utf-8")
                data = yaml.safe_load(text)
                tool = _parse_tool_yaml(data)
                self._register_tool(tool, source=f"built-in catalog file {item.name}")
        if env_dir is not None:
            self.load_from_directory(env_dir / "custom-tools")

    def load_from_directory(self, path: Path) -> None:
        """Load tool YAMLs from an arbitrary directory (for testing)."""
        if not path.is_dir():
            return
        for yaml_file in sorted(path.glob("*.yaml")):
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            tool = _parse_tool_yaml(data)
            self._register_tool(tool, source=str(yaml_file))

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Return the tool definition for the given name, or None."""
        return self._tools.get(name)

    def get_tools(self, names: list[str]) -> list[ToolDefinition]:
        """Return tool definitions for the given names; raises KeyError if unknown."""
        tools = []
        for name in names:
            tool = self._tools.get(name)
            if tool is None:
                raise KeyError(f"Unknown tool: {name}")
            tools.append(tool)
        return tools

    def list_all(self) -> list[ToolDefinition]:
        """Return all tools sorted by category and name."""
        return sorted(self._tools.values(), key=lambda t: (t.category, t.name))

    def list_by_category(self) -> dict[str, list[ToolDefinition]]:
        """Return tools grouped by category."""
        categories: dict[str, list[ToolDefinition]] = {}
        for tool in self.list_all():
            categories.setdefault(tool.category, []).append(tool)
        return categories

    @property
    def tool_names(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())
