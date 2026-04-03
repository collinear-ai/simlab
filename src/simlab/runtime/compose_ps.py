"""Helpers for parsing ``docker compose ps`` output."""

from __future__ import annotations

import re
from collections.abc import Collection

_COMPOSE_CONTAINER_SUFFIX = re.compile(r"-\d+$")


def _service_name_from_container(
    name: str,
    *,
    expected_services: Collection[str] | None = None,
) -> str:
    """Recover the compose service name from a container name."""
    trimmed = _COMPOSE_CONTAINER_SUFFIX.sub("", name)
    if expected_services:
        matches = [
            service
            for service in expected_services
            if trimmed == service or trimmed.endswith(f"-{service}")
        ]
        if matches:
            return max(matches, key=len)
    parts = trimmed.split("-")
    if len(parts) >= 3:
        return "-".join(parts[1:])
    return trimmed


def parse_ps_output(
    text: str,
    *,
    expected_services: Collection[str] | None = None,
) -> dict[str, str]:
    """Parse ``docker compose ps`` output into ``{service: status}``."""
    services: dict[str, str] = {}
    for line in text.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        name = _service_name_from_container(parts[0], expected_services=expected_services)
        raw_status = parts[1].lower()
        if (
            "healthy" in raw_status
            and "unhealthy" not in raw_status
            and "starting" not in raw_status
        ):
            services[name] = "healthy"
        elif "starting" in raw_status:
            services[name] = "starting"
        elif "unhealthy" in raw_status:
            services[name] = "unhealthy"
        elif "exited" in raw_status:
            services[name] = "exited"
        elif "up" in raw_status:
            services[name] = "running"
        else:
            services[name] = "unknown"
    return services
