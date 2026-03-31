#!/usr/bin/env python3
"""Patch a tasks-gen config.toml: set num_tasks and disable quality filter.

Usage:
    python patch_taskgen_config.py taskgen/config.toml 4
"""

import re
import sys
from pathlib import Path


def main() -> None:
    config_path = Path(sys.argv[1])
    num_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    text = config_path.read_text()
    text = re.sub(r"num_tasks\s*=\s*\d+", f"num_tasks = {num_tasks}", text)
    if "filter" not in text:
        text = text.replace("[generation]", "[generation]\nfilter = false")

    config_path.write_text(text)
    print(f"Patched {config_path}: num_tasks={num_tasks}, filter=false")


if __name__ == "__main__":
    main()
