"""Helpers for parsing ``docker compose ps`` output."""

from __future__ import annotations


def parse_ps_output(text: str) -> dict[str, str]:
    """Parse ``docker compose ps`` output into ``{service: status}``."""
    services: dict[str, str] = {}
    for line in text.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("-", 2)[-1].rsplit("-", 1)[0]
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
