"""Compile a Harbor task into a generated SimLab env and task bundle."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import tomllib
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias
from typing import cast
from urllib.parse import urlsplit

import yaml

from simlab.env_artifacts import regenerate_env_artifacts

_HARBOR_TOOL_NAME = "harbor-main"
_HARBOR_AGENT_SERVICE = "harbor-openhands-agent-server"
_HARBOR_CODING_SERVICE = "harbor-coding-env"
_HARBOR_WORKSPACE_VOLUME = "harbor-main-workspace"
_DEFAULT_WORKDIR = "/workspace"
_OPENHANDS_VENV_DIR = "/opt/openhands-venv"
_OPENHANDS_SEED_DIR = "/opt/harbor-seed/workdir"
_UV_DOCKER_IMAGE = "ghcr.io/astral-sh/uv:0.10.9"
_OPENHANDS_VERSION = "1.12.1"
_OPENHANDS_AGENT_SERVER_VERSION = "1.11.2"

HarborScalar: TypeAlias = str | int | float | bool | None
HarborValue: TypeAlias = HarborScalar | Sequence["HarborValue"] | Mapping[str, "HarborValue"]
HarborObject: TypeAlias = dict[str, HarborValue]
HarborServices: TypeAlias = dict[str, HarborObject]
HarborEnvironment: TypeAlias = dict[str, str] | list[str]

_OPENHANDS_ENTRYPOINT = """#!/usr/bin/env bash
set -euo pipefail

workspace_dir="${OPENHANDS_WORKSPACE_DIR:-/workspace}"
seed_dir="${OPENHANDS_SEED_DIR:-/opt/harbor-seed/workdir}"
setup_dir="${OPENHANDS_SETUP_DIR:-/app/setup}"

if [[ -d "${seed_dir}" ]]; then
  mkdir -p "${workspace_dir}"
  cp -an "${seed_dir}/." "${workspace_dir}/"
fi

if [[ -d "${setup_dir}" ]]; then
  mapfile -t setup_scripts < <(find "${setup_dir}" -maxdepth 1 -type f -name '*.sh' | sort)
  for script_path in "${setup_scripts[@]}"; do
    echo "[openhands-agent-server] running setup script: ${script_path}"
    bash "${script_path}"
  done
fi

exec python -m openhands.agent_server --host 0.0.0.0 --port 8000
"""

_OPENHANDS_HEALTHCHECK = """#!/usr/bin/env python3
import os
import sys
from urllib.request import Request
from urllib.request import urlopen

session_key = os.getenv("SESSION_API_KEY", "dev-session-key")
request = Request("http://localhost:8000/health", headers={"X-Session-API-Key": session_key})

try:
    with urlopen(request, timeout=3) as response:  # noqa: S310
        sys.exit(0 if response.getcode() == 200 else 1)
except Exception:
    sys.exit(1)
"""


@dataclass(frozen=True)
class HarborMcpServer:
    """One Harbor MCP server entry from ``task.toml``."""

    name: str
    transport: str | None
    url: str

    @property
    def service_host(self) -> str | None:
        """Return the backing Docker service hostname when the URL targets one."""
        host = urlsplit(self.url).hostname
        if not host or host in {"localhost", "127.0.0.1"}:
            return None
        return host

    @property
    def service_port(self) -> int | None:
        """Return the backing Docker service port when present in the URL."""
        return urlsplit(self.url).port


@dataclass(frozen=True)
class HarborTaskSpec:
    """Validated Harbor task inputs relevant to the SimLab adapter."""

    task_dir: Path
    task_id: str
    display_name: str
    instruction_text: str
    difficulty: str
    category: str
    workdir: str
    dockerfile_text: str | None
    docker_image: str | None
    agent_timeout_seconds: float | None
    verifier_timeout_seconds: float | None
    solution_env: dict[str, str]
    verifier_env: dict[str, str]
    compose_services: HarborServices
    main_service_overrides: HarborObject
    mcp_servers: list[HarborMcpServer]


@dataclass(frozen=True)
class HarborPreparedRun:
    """Paths for a generated temporary Harbor run."""

    env_dir: Path
    bundle_dir: Path
    task_id: str
    agent_timeout_seconds: float | None


def parse_harbor_task(task_dir: Path, *, task_id: str | None = None) -> HarborTaskSpec:
    """Parse and validate a single Harbor task directory."""
    resolved_task_dir = task_dir.resolve()
    task_toml = resolved_task_dir / "task.toml"
    instruction_md = resolved_task_dir / "instruction.md"
    tests_dir = resolved_task_dir / "tests"
    test_script = tests_dir / "test.sh"
    environment_dir = resolved_task_dir / "environment"

    if not task_toml.is_file():
        raise ValueError(f"Harbor task is missing task.toml: {task_toml}")
    if not instruction_md.is_file():
        raise ValueError(f"Harbor task is missing instruction.md: {instruction_md}")
    if not tests_dir.is_dir():
        raise ValueError(f"Harbor task is missing tests/: {tests_dir}")
    if not test_script.is_file():
        raise ValueError(f"Harbor task is missing tests/test.sh: {test_script}")

    raw = _require_table(tomllib.loads(task_toml.read_text(encoding="utf-8")), label="task.toml")
    metadata = _table_or_empty(raw.get("metadata"), label="metadata")
    agent_cfg = _table_or_empty(raw.get("agent"), label="agent")
    environment_cfg = _table_or_empty(raw.get("environment"), label="environment")
    verifier_cfg = _table_or_empty(raw.get("verifier"), label="verifier")
    solution_cfg = _table_or_empty(raw.get("solution"), label="solution")

    dockerfile_path = environment_dir / "Dockerfile"
    dockerfile_text = (
        dockerfile_path.read_text(encoding="utf-8") if dockerfile_path.is_file() else None
    )
    docker_image = _string_or_none(environment_cfg.get("docker_image"))
    if dockerfile_text is None and docker_image is None:
        raise ValueError(
            "Harbor task must define environment/Dockerfile or environment.docker_image"
        )

    compose_services, main_service_overrides = _load_harbor_compose_overrides(environment_dir)
    instruction_text = instruction_md.read_text(encoding="utf-8").strip()
    resolved_task_id = _normalize_task_id(task_id or resolved_task_dir.name)

    mcp_servers: list[HarborMcpServer] = []
    for entry in _list_or_none(environment_cfg.get("mcp_servers")) or []:
        if not isinstance(entry, dict):
            raise TypeError("environment.mcp_servers entries must be tables")
        entry_table = _require_table(entry, label="environment.mcp_servers entries")
        name = _string_or_none(entry_table.get("name"))
        url = _string_or_none(entry_table.get("url"))
        if not name or not url:
            raise ValueError("environment.mcp_servers entries must include name and url")
        mcp_servers.append(
            HarborMcpServer(
                name=name,
                transport=_string_or_none(entry_table.get("transport")),
                url=url,
            )
        )

    return HarborTaskSpec(
        task_dir=resolved_task_dir,
        task_id=resolved_task_id,
        display_name=_display_name_for_task(metadata, resolved_task_dir),
        instruction_text=instruction_text,
        difficulty=_string_or_none(metadata.get("difficulty")) or "medium",
        category=_string_or_none(metadata.get("category")) or "harbor",
        workdir=_infer_workdir(dockerfile_text) if dockerfile_text else _DEFAULT_WORKDIR,
        dockerfile_text=dockerfile_text,
        docker_image=docker_image,
        agent_timeout_seconds=_timeout_seconds_or_none(
            agent_cfg.get("timeout_sec"),
            label="agent.timeout_sec",
        ),
        verifier_timeout_seconds=_timeout_seconds_or_none(
            verifier_cfg.get("timeout_sec"),
            label="verifier.timeout_sec",
        ),
        solution_env=_normalize_env_mapping(
            _table_or_empty(solution_cfg.get("env"), label="solution.env")
        ),
        verifier_env=_normalize_env_mapping(
            _table_or_empty(verifier_cfg.get("env"), label="verifier.env")
        ),
        compose_services=compose_services,
        main_service_overrides=main_service_overrides,
        mcp_servers=mcp_servers,
    )


def prepare_harbor_run(
    task_dir: Path,
    *,
    workspace_root: Path,
    task_id: str | None = None,
) -> HarborPreparedRun:
    """Generate a temporary env and local task bundle for one Harbor task."""
    spec = parse_harbor_task(task_dir, task_id=task_id)
    env_dir = workspace_root / "env"
    bundle_dir = workspace_root / "task-bundle"
    harbor_env_dir = env_dir / "harbor-environment"
    harbor_tests_dir = env_dir / "harbor-tests"
    custom_tools_dir = env_dir / "custom-tools"

    env_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    custom_tools_dir.mkdir(parents=True, exist_ok=True)
    harbor_env_dir.mkdir(parents=True, exist_ok=True)
    if (spec.task_dir / "environment").is_dir():
        shutil.copytree(spec.task_dir / "environment", harbor_env_dir, dirs_exist_ok=True)
    shutil.copytree(spec.task_dir / "tests", harbor_tests_dir, dirs_exist_ok=True)

    support_dir = harbor_env_dir / ".simlab"
    support_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(spec.task_dir / "tests", support_dir / "tests", dirs_exist_ok=True)
    (support_dir / "openhands-entrypoint.sh").write_text(_OPENHANDS_ENTRYPOINT, encoding="utf-8")
    (support_dir / "openhands-healthcheck.py").write_text(_OPENHANDS_HEALTHCHECK, encoding="utf-8")
    (harbor_env_dir / "Dockerfile.simlab-wrapper").write_text(
        _build_wrapper_dockerfile(spec),
        encoding="utf-8",
    )

    (env_dir / "env.yaml").write_text(
        yaml.safe_dump(
            {
                "name": f"harbor-{spec.task_id}",
                "tools": [_HARBOR_TOOL_NAME],
                "overrides": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (custom_tools_dir / f"{_HARBOR_TOOL_NAME}.yaml").write_text(
        yaml.safe_dump(_build_harbor_tool_definition(spec), sort_keys=False),
        encoding="utf-8",
    )
    if spec.mcp_servers:
        (env_dir / "mcp-servers.json").write_text(
            json.dumps(
                {"mcpServers": {server.name: {"url": server.url} for server in spec.mcp_servers}},
                indent=2,
            ),
            encoding="utf-8",
        )

    regenerate_env_artifacts(env_dir)

    tasks_dir = bundle_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f"{spec.task_id}.json").write_text(
        json.dumps(_build_task_payload(spec), indent=2),
        encoding="utf-8",
    )

    return HarborPreparedRun(
        env_dir=env_dir,
        bundle_dir=bundle_dir,
        task_id=spec.task_id,
        agent_timeout_seconds=spec.agent_timeout_seconds,
    )


def _build_task_payload(spec: HarborTaskSpec) -> HarborObject:
    payload: HarborObject = {
        "meta": {
            "task_id": spec.task_id,
            "display_name": spec.display_name,
            "difficulty": spec.difficulty,
            "category": spec.category,
            "source": "harbor",
        },
        "task": spec.instruction_text,
        "apps": [],
        "tool_servers": [
            {
                "name": _HARBOR_TOOL_NAME,
                "tool_server_url": "http://harbor-main:8020",
            }
        ],
        "seed_emails": [],
        "seed_calendar_events": [],
        "seed_group_channels": [],
        "npcs": [],
        "verifiers": [],
        "harbor_verifier": {
            "kind": "test_sh",
            "service": _HARBOR_AGENT_SERVICE,
            "script_path": "/tests/test.sh",
            "workdir": spec.workdir,
            "env": spec.verifier_env,
        },
    }
    if spec.verifier_timeout_seconds is not None:
        harbor_verifier = payload["harbor_verifier"]
        if isinstance(harbor_verifier, dict):
            harbor_verifier["timeout_sec"] = spec.verifier_timeout_seconds
    return payload


def _build_harbor_tool_definition(spec: HarborTaskSpec) -> HarborObject:
    services: HarborServices = {}
    main_env = {"SESSION_API_KEY": "dev-session-key"}
    main_env.update(spec.solution_env)
    raw_main_environment = spec.main_service_overrides.get("environment")
    main_overrides_env = _normalize_service_environment(
        raw_main_environment if isinstance(raw_main_environment, (dict, list)) else None
    )
    merged_main_env = _merge_service_environment(main_overrides_env, main_env)

    main_volumes = [f"{_HARBOR_WORKSPACE_VOLUME}:{spec.workdir}"]
    main_volumes.extend(
        _rewrite_volume_mounts(_list_or_none(spec.main_service_overrides.get("volumes")))
    )

    main_service: HarborObject = {
        "build": {
            "context": "./harbor-environment",
            "dockerfile": "Dockerfile.simlab-wrapper",
        },
        "ports": ["8000"],
        "environment": merged_main_env,
        "volumes": _dedupe_strings(main_volumes),
    }
    depends_on = spec.main_service_overrides.get("depends_on")
    if isinstance(depends_on, dict):
        main_service["depends_on"] = depends_on
    services[_HARBOR_AGENT_SERVICE] = main_service

    services[_HARBOR_CODING_SERVICE] = {
        "image": "collinear/coding-env:latest",
        "ports": ["8020"],
        "environment": {
            "OPENHANDS_AGENT_SERVER_URL": f"http://{_HARBOR_AGENT_SERVICE}:8000",
            "OPENHANDS_SESSION_API_KEY": "dev-session-key",
            "OPENHANDS_WORKING_DIR": spec.workdir,
        },
        "depends_on": {
            _HARBOR_AGENT_SERVICE: {
                "condition": "service_healthy",
            }
        },
    }

    for service_name, service in spec.compose_services.items():
        services[service_name] = _convert_harbor_service(service)

    exposed_ports: list[HarborObject] = [
        {
            "port": port,
            "description": f"Harbor sidecar port {port}",
        }
        for port in sorted(_collect_exposed_ports(spec))
    ]

    return {
        "name": _HARBOR_TOOL_NAME,
        "display_name": spec.display_name,
        "description": f"Generated Harbor runtime for {spec.task_id}",
        "category": "development",
        "tool_server_port": 8020,
        "exposed_ports": exposed_ports,
        "services": services,
        "volumes": {
            _HARBOR_WORKSPACE_VOLUME: {},
        },
    }


def _build_wrapper_dockerfile(spec: HarborTaskSpec) -> str:
    base = spec.dockerfile_text or f"FROM {spec.docker_image}\n"
    docker_workdir = json.dumps(spec.workdir)
    docker_seed_dir = json.dumps(_OPENHANDS_SEED_DIR)
    quoted_workdir = shlex.quote(spec.workdir)
    quoted_seed_dir = shlex.quote(_OPENHANDS_SEED_DIR)
    return (
        f"{base.rstrip()}\n\n"
        "USER root\n"
        "RUN if command -v apt-get >/dev/null 2>&1; then \\\n"
        "      apt-get update -qq >/dev/null && \\\n"
        "      DEBIAN_FRONTEND=noninteractive DEBCONF_NOWARNINGS=yes \\\n"
        "      apt-get install -y -qq -o Dpkg::Use-Pty=0 \\\n"
        "        bash \\\n"
        "        ca-certificates \\\n"
        "        curl \\\n"
        "        git \\\n"
        "        nodejs \\\n"
        "        npm \\\n"
        "        python3 \\\n"
        "        python3-pip >/dev/null && \\\n"
        "      rm -rf /var/lib/apt/lists/*; \\\n"
        "    elif command -v apk >/dev/null 2>&1; then \\\n"
        "      apk add --no-cache \\\n"
        "        bash \\\n"
        "        ca-certificates \\\n"
        "        curl \\\n"
        "        git \\\n"
        "        nodejs \\\n"
        "        npm \\\n"
        "        python3 \\\n"
        "        py3-pip >/dev/null; \\\n"
        "    fi\n"
        "RUN if ! command -v bash >/dev/null 2>&1; then \\\n"
        '      echo "bash is required in the Harbor runtime wrapper" >&2; exit 1; \\\n'
        "    fi\n"
        "RUN if ! command -v python3 >/dev/null 2>&1; then \\\n"
        '      echo "python3 is required in the Harbor runtime wrapper" >&2; exit 1; \\\n'
        "    fi\n"
        f"COPY --from={_UV_DOCKER_IMAGE} /uv /uvx /bin/\n"
        f"RUN uv venv --python python3 {_OPENHANDS_VENV_DIR}\n"
        f'ENV PATH="{_OPENHANDS_VENV_DIR}/bin:${{PATH}}"\n'
        f"RUN uv pip install --quiet --python {_OPENHANDS_VENV_DIR}/bin/python "
        f"openhands=={_OPENHANDS_VERSION} "
        f"openhands-agent-server=={_OPENHANDS_AGENT_SERVER_VERSION}\n"
        "RUN mkdir -p /root/.local/bin && \\\n"
        "    ln -sf /bin/uv /root/.local/bin/uv && \\\n"
        "    ln -sf /bin/uvx /root/.local/bin/uvx && \\\n"
        "    printf 'export PATH=\"/root/.local/bin:${PATH}\"\\n' > /root/.local/bin/env\n"
        "COPY .simlab/openhands-healthcheck.py /healthcheck.py\n"
        "COPY .simlab/openhands-entrypoint.sh /entrypoint.sh\n"
        "COPY .simlab/tests /tests\n"
        f"RUN mkdir -p {quoted_seed_dir} && \\\n"
        f"    if [ -d {quoted_workdir} ]; then cp -a {quoted_workdir}/. {quoted_seed_dir}/; fi\n"
        "RUN chmod +x /entrypoint.sh && mkdir -p /tests /logs/verifier\n"
        "EXPOSE 8000\n"
        "ENV SESSION_API_KEY=dev-session-key\n"
        f"ENV OPENHANDS_WORKSPACE_DIR={docker_workdir}\n"
        f"ENV OPENHANDS_SEED_DIR={docker_seed_dir}\n"
        f"WORKDIR {docker_workdir}\n"
        "HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=12 "
        "CMD python3 /healthcheck.py\n"
        'ENTRYPOINT ["/entrypoint.sh"]\n'
    )


def _load_harbor_compose_overrides(
    environment_dir: Path,
) -> tuple[HarborServices, HarborObject]:
    compose_path = environment_dir / "docker-compose.yaml"
    if not compose_path.is_file():
        compose_path = environment_dir / "docker-compose.yml"
    if not compose_path.is_file():
        return {}, {}

    data = _require_table(
        cast(object, yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}),
        label=f"Harbor docker-compose override {compose_path}",
    )
    services = data.get("services")
    if not isinstance(services, dict):
        raise TypeError(f"Harbor docker-compose override must define services: {compose_path}")

    copied: HarborServices = {
        str(service_name): _require_table(service, label=f"services.{service_name}")
        for service_name, service in services.items()
    }
    main_service = copied.pop("main", {})
    if main_service and not isinstance(main_service, dict):
        raise ValueError("services.main in Harbor docker-compose override must be an object")

    return copied, main_service if isinstance(main_service, dict) else {}


def _convert_harbor_service(service: HarborObject) -> HarborObject:
    converted: HarborObject = {}
    image = _string_or_none(service.get("image"))
    if image:
        converted["image"] = image

    build = service.get("build")
    if build is not None:
        converted["build"] = _rewrite_build_definition(build)

    ports = _normalize_service_ports(
        _list_or_none(service.get("ports")),
        _list_or_none(service.get("expose")),
    )
    if ports:
        converted["ports"] = ports

    raw_environment = service.get("environment")
    environment = _normalize_service_environment(
        raw_environment if isinstance(raw_environment, (dict, list)) else None
    )
    if environment:
        converted["environment"] = environment

    depends_on = service.get("depends_on")
    if isinstance(depends_on, dict):
        converted["depends_on"] = depends_on

    volumes = _rewrite_volume_mounts(_list_or_none(service.get("volumes")))
    if volumes:
        converted["volumes"] = volumes

    command = _normalize_service_command(service.get("command"))
    if command is not None:
        converted["command"] = command

    healthcheck = service.get("healthcheck")
    if isinstance(healthcheck, dict):
        converted["healthcheck"] = healthcheck

    return converted


def _rewrite_build_definition(build: HarborValue) -> str | dict[str, str]:
    if isinstance(build, str):
        return _rewrite_relative_path(build)
    if isinstance(build, dict):
        context = _string_or_none(build.get("context"))
        if not context:
            raise ValueError("Harbor compose build definitions must include context")
        rewritten: dict[str, str] = {"context": _rewrite_relative_path(context)}
        dockerfile = _string_or_none(build.get("dockerfile"))
        if dockerfile:
            rewritten["dockerfile"] = dockerfile
        return rewritten
    raise ValueError("Unsupported Harbor compose build definition")


def _rewrite_relative_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return path.as_posix()
    normalized = (Path("harbor-environment") / path).as_posix()
    return f"./{normalized}"


def _rewrite_volume_mounts(raw_volumes: list[HarborValue] | None) -> list[str]:
    if raw_volumes is None:
        return []

    rewritten: list[str] = []
    for entry in raw_volumes:
        if not isinstance(entry, str):
            continue
        source, sep, remainder = entry.partition(":")
        if not sep:
            rewritten.append(entry)
            continue
        if source.startswith("/"):
            rewritten.append(entry)
            continue
        if source.startswith(".") or "/" in source:
            rewritten.append(f"{_rewrite_relative_path(source)}:{remainder}")
            continue
        rewritten.append(entry)
    return rewritten


def _normalize_service_ports(
    raw_ports: list[HarborValue] | None,
    raw_expose: list[HarborValue] | None,
) -> list[str]:
    ports: list[str] = []
    for collection in (raw_ports, raw_expose):
        if collection is None:
            continue
        for entry in collection:
            port = _extract_container_port(entry)
            if port is not None:
                ports.append(str(port))
    return _dedupe_strings(ports)


def _extract_container_port(entry: HarborValue) -> int | None:
    if isinstance(entry, int):
        return entry
    if isinstance(entry, str):
        tail = entry.split(":", maxsplit=entry.count(":"))[-1]
        container = tail.split("/", maxsplit=1)[0].strip().strip('"').strip("'")
        return int(container) if container.isdigit() else None
    return None


def _collect_exposed_ports(spec: HarborTaskSpec) -> set[int]:
    ports: set[int] = set()
    for server in spec.mcp_servers:
        if server.service_host and server.service_port:
            ports.add(server.service_port)
    return ports


def _infer_workdir(dockerfile_text: str | None) -> str:
    if not dockerfile_text:
        return _DEFAULT_WORKDIR
    # Harbor tasks normally use a single final runtime stage. We intentionally
    # follow the last WORKDIR we see, which can be wrong for unusual multi-stage
    # Dockerfiles that append another non-runtime stage after the final one.
    matches = re.findall(r"(?im)^\s*WORKDIR\s+(.+?)\s*$", dockerfile_text)
    if not matches:
        return _DEFAULT_WORKDIR
    candidate = matches[-1].strip()
    if candidate.startswith("["):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return _DEFAULT_WORKDIR
        if isinstance(parsed, list) and parsed:
            first = str(parsed[0]).strip()
            return first or _DEFAULT_WORKDIR
        return _DEFAULT_WORKDIR
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {'"', "'"}:
        candidate = candidate[1:-1].strip()
    return candidate or _DEFAULT_WORKDIR


def _normalize_env_mapping(raw: HarborObject | None) -> dict[str, str]:
    return {str(key): str(value) for key, value in (raw or {}).items()}


def _normalize_service_environment(
    raw: HarborObject | list[HarborValue] | None,
) -> HarborEnvironment:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    result: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        entry = item.strip()
        if not entry or entry in result:
            continue
        result.append(entry)
    return result


def _merge_service_environment(
    raw_env: HarborEnvironment,
    generated_env: dict[str, str],
) -> HarborEnvironment:
    if isinstance(raw_env, dict):
        merged = dict(raw_env)
        merged.update(generated_env)
        return merged

    passthrough_env: list[str] = []
    explicit_env: dict[str, str] = {}
    for item in raw_env:
        if "=" in item:
            key, value = item.split("=", 1)
            explicit_env[key] = value
            continue
        if item not in passthrough_env:
            passthrough_env.append(item)

    for key, value in generated_env.items():
        explicit_env[key] = value
        passthrough_env = [entry for entry in passthrough_env if entry != key]

    return [*passthrough_env, *(f"{key}={value}" for key, value in explicit_env.items())]


def _normalize_service_command(raw: HarborValue) -> str | list[str] | None:
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped or None

    items = _list_or_none(raw)
    if items is None:
        return None

    command: list[str] = []
    for item in items:
        if isinstance(item, (dict, list)) or item is None:
            return None
        command.append(str(item))
    return command or None


def _normalize_task_id(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip()).strip("-")
    return text or "harbor-task"


def _display_name_for_task(metadata: HarborObject, task_dir: Path) -> str:
    explicit = _string_or_none(metadata.get("title")) or _string_or_none(metadata.get("name"))
    if explicit:
        return explicit
    return task_dir.name.replace("-", " ").replace("_", " ").strip() or task_dir.name


def _string_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _timeout_seconds_or_none(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a number")
    timeout_seconds = float(value)
    if timeout_seconds <= 0:
        raise ValueError(f"{label} must be greater than zero")
    return timeout_seconds


def _require_table(raw: object, *, label: str) -> HarborObject:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be a table")
    return {str(key): cast(HarborValue, value) for key, value in raw.items()}


def _table_or_empty(raw: HarborValue, *, label: str) -> HarborObject:
    if raw is None:
        return {}
    return _require_table(raw, label=label)


def _list_or_none(raw: HarborValue) -> list[HarborValue] | None:
    if isinstance(raw, list):
        return raw
    return None


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
