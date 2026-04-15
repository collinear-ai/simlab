#!/usr/bin/env python3
"""Fetch tool schemas from a running SimLab environment and save as JSON.

Queries each tool server's /tools endpoint, strips null values (which break
SimLab's TOML serialization), and writes a flat JSON array suitable for
`simlab tasks-gen run --tools`.

Usage:
    python fetch_tool_schema.py --env sft-scale --output tool_schema.json

    # Then use with task gen:
    simlab tasks-gen run --tools tool_schema.json --describe "..." --num-tasks 32
"""

import argparse
import json
import urllib.request
from pathlib import Path

import yaml
from daytona import Daytona
from simlab.catalog.registry import ToolRegistry
from simlab.composer.engine import EnvConfig


def strip_none(obj: object) -> object:
    """Recursively remove None values from dicts (TOML can't serialize null)."""
    if isinstance(obj, dict):
        return {k: strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_none(i) for i in obj]
    return obj


def main() -> None:
    """Fetch tool schemas from a running SimLab environment."""
    parser = argparse.ArgumentParser(description="Fetch tool schemas from SimLab env")
    parser.add_argument("--env", required=True, help="Environment name")
    parser.add_argument("--output", default="tool_schema.json", help="Output JSON file")
    parser.add_argument("--env-dir", default=None, help="Override environments directory")
    args = parser.parse_args()

    env_dir = Path(args.env_dir) if args.env_dir else Path(f"environments/{args.env}")
    config = EnvConfig(**yaml.safe_load((env_dir / "env.yaml").read_text()))

    state_file = env_dir / "daytona-state.json"
    if not state_file.exists():
        print(f"ERROR: No daytona-state.json in {env_dir}. Is the env running?")
        raise SystemExit(1)

    state = json.loads(state_file.read_text())
    sandbox = Daytona().get(state["sandbox_id"])
    registry = ToolRegistry()
    registry.load_all()

    all_tools = []
    for tool_name in config.tools:
        tool = registry.get_tool(tool_name)
        if tool and tool.tool_server_port:
            url = sandbox.get_preview_link(tool.tool_server_port).url
            try:
                resp = urllib.request.urlopen(f"{url}/tools", timeout=10)  # noqa: S310
                data = json.loads(resp.read())
                for t in data.get("tools", []):
                    all_tools.append(strip_none(t))
                    print(f"  {tool_name}: {t['name']}")
            except Exception as e:
                print(f"  {tool_name}: FAILED ({e})")

    Path(args.output).write_text(json.dumps(all_tools, indent=2))
    print(f"\nWrote {len(all_tools)} tools to {args.output}")


if __name__ == "__main__":
    main()
