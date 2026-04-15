#!/usr/bin/env python3
"""Patch a tasks-gen config.toml: set num_tasks, disable quality filter, optionally override model.

The API caps num_tasks at 200 per request. If the requested count exceeds 200,
this script sets it to 200 — the caller is responsible for batching multiple
generation runs and merging results.

Usage:
    python patch_taskgen_config.py taskgen/config.toml 10
    python patch_taskgen_config.py taskgen/config.toml 200   # max per API call
    python patch_taskgen_config.py taskgen/config.toml 10 --model claude-sonnet-4-6
"""

import re
import sys
from pathlib import Path

MAX_TASKS_PER_REQUEST = 200


def main() -> None:
    """Patch a tasks-gen config.toml with num_tasks, filter, and optional model."""
    # Parse positional and optional --model flag
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = dict(zip(sys.argv[1:], sys.argv[2:], strict=False))
    model = flags.get("--model")

    config_path = Path(args[0])
    num_tasks = int(args[1]) if len(args) > 1 else 10
    capped = min(num_tasks, MAX_TASKS_PER_REQUEST)

    text = config_path.read_text()
    text = re.sub(r"num_tasks\s*=\s*\d+", f"num_tasks = {capped}", text)
    # Replace/insert filter = false scoped to the [generation] section
    gen_match = re.search(r"(?m)^\[generation\]", text)
    if gen_match:
        # Find the end of the [generation] section (next [header] or EOF)
        next_section = re.search(r"(?m)^\[", text[gen_match.end() :])
        section_end = gen_match.end() + next_section.start() if next_section else len(text)
        section = text[gen_match.end() : section_end]
        if re.search(r"(?m)^filter\s*=", section):
            section = re.sub(r"(?m)^filter\s*=\s*\S+", "filter = false", section)
        else:
            section = "\nfilter = false" + section
        text = text[: gen_match.end()] + section + text[section_end:]

    if model:
        text = re.sub(r'(?m)^model\s*=\s*"[^"]+"', f'model = "{model}"', text)

    config_path.write_text(text)
    print(
        f"Patched {config_path}: num_tasks={capped}, filter=false"
        + (f", model={model}" if model else "")
    )
    if num_tasks > MAX_TASKS_PER_REQUEST:
        print(
            f"  Note: requested {num_tasks} but API caps at {MAX_TASKS_PER_REQUEST}. "
            f"Run generation multiple times and merge results."
        )


if __name__ == "__main__":
    main()
