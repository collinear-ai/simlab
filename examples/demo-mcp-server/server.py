"""Minimal MCP server for local testing (no auth). Exposes a few tools over Streamable HTTP."""

from __future__ import annotations

import os

from fastmcp import FastMCP

mcp = FastMCP(
    name="simlab-demo-mcp",
    instructions="Demo MCP server for testing simlab. No authentication required.",
)


@mcp.tool()
def echo(message: str) -> str:
    """Echo back the given message."""
    return f"Echo: {message}"


@mcp.tool()
def ping() -> str:
    """Return a simple pong response."""
    return "pong"


@mcp.tool()
def get_env(key: str) -> str:
    """Return the value of an environment variable (for testing)."""
    return os.environ.get(key, f"<not set: {key}>")


if __name__ == "__main__":
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8081"))
    mcp.run(transport="streamable-http", host=host, port=port)
