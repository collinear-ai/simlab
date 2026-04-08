# Harbor Tasks

Harbor is a task format for running self-contained coding tasks in SimLab. A Harbor task bundles an instruction, environment (Dockerfile), and test-based verifier into a single directory. SimLab compiles it into a generated environment and runs the normal agent + verifier flow.

## Quick Start

```bash
simlab tasks run --harbor ./examples/harbor/hello-world \
  --agent-model gpt-4o-mini \
  --agent-api-key "$OPENAI_API_KEY"
```

No `env init` or `env up` needed — `--harbor` handles everything.

## Task Directory Structure

A Harbor task is a directory with this layout:

```
my-task/
  task.toml              # Task metadata and resource limits
  instruction.md         # The task instruction given to the agent
  environment/
    Dockerfile           # Base environment for the task
  tests/
    test.sh              # Verifier entry point (runs inside the environment)
    test_outputs.py      # (optional) pytest-based test assertions
```

### task.toml

Defines metadata and resource limits:

```toml
version = "1.0"

[metadata]
difficulty = "easy"
category = "programming"
tags = ["example", "hello-world"]

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
```

### instruction.md

The task instruction given to the agent. Example:

```markdown
Create a file called `hello.txt` with `Hello, world!` as the content.
```

### environment/Dockerfile

Defines the base container the agent works in:

```dockerfile
FROM ubuntu:24.04
WORKDIR /app
```

### tests/

The verifier runs `test.sh` inside the environment after the agent finishes. A passing verifier writes `1` to `/logs/verifier/reward.txt`.

`test.sh` can run pytest or any other test framework. Example using pytest:

```bash
#!/usr/bin/env bash
set -euo pipefail
uvx --with pytest pytest /tests/test_outputs.py -rA
echo 1 > /logs/verifier/reward.txt
```

## Running with Daytona

Add `--daytona` to run the Harbor task in a remote sandbox:

```bash
simlab tasks run --harbor ./my-task \
  --daytona \
  --agent-model gpt-4o-mini \
  --agent-api-key "$OPENAI_API_KEY"
```

## Keeping the Workspace

By default, the generated workspace is cleaned up after the run. To retain it for inspection:

```bash
simlab tasks run --harbor ./my-task \
  --keep-alive \
  --agent-model gpt-4o-mini
```

The workspace is preserved under `output/harbor_runs/`.

## Output Format

Harbor runs default to ATIF trajectory format. Output is saved to `output/agent_run_<task_id>_<timestamp>/`:

```
output/agent_run_<task_id>_<timestamp>/
  agent/trajectory.json    # ATIF trajectory
  verifier/reward.txt      # "1" or "0"
  verifier/reward.json     # e.g. {"reward": 1.0}
```

To override the format, pass `--tasks-rollout-format default`.

## Current Limits

- `--harbor` runs a single task directory, not a suite directory
- `--skip-env-setup` is not supported with `--harbor`
- `--rollout-count > 1` is not supported with `--harbor`
