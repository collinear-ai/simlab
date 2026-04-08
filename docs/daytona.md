# Daytona

Daytona provides remote sandbox execution for SimLab environments. Instead of running Docker containers locally, each rollout runs in an ephemeral Daytona sandbox.

## Setup

### 1. Install with Daytona support

```bash
uv tool install --python 3.13 "simulationlab[daytona]"
```

### 2. Get your API key

Sign up at [app.daytona.io](https://app.daytona.io) and create an API key.

### 3. Configure the key

Pick one:

```bash
# config.toml
[daytona]
api_key = "dtn_..."
```

```bash
# Environment variable
export SIMLAB_DAYTONA_API_KEY="dtn_..."
# or
export DAYTONA_API_KEY="dtn_..."
```

```bash
# CLI flag (--daytona-api-key is a root option, placed before the subcommand)
simlab --daytona-api-key "dtn_..." tasks run --daytona ...
```

Resolution order: CLI flag > `SIMLAB_DAYTONA_API_KEY` > `DAYTONA_API_KEY` > `[daytona].api_key` in config.toml.

## Running with Daytona

Add `--daytona` to any `tasks run` command:

```bash
simlab tasks run --env my-env --task my-task --daytona \
  --agent-model gpt-5.2 --agent-api-key "$OPENAI_API_KEY"
```

Daytona creates the sandbox, runs the full lifecycle (setup, seed, agent, verify, teardown), and destroys the sandbox when done.

## Parallel Rollouts

Run the same task multiple times in parallel to collect diverse agent trajectories:

```bash
simlab tasks run --env my-env --task my-task --daytona \
  --rollout-count 5 --max-parallel 3 \
  --agent-model gpt-5.2 --agent-api-key "$OPENAI_API_KEY"
```

| Flag | Description | Default |
|------|-------------|---------|
| `--rollout-count N` | Total rollouts to execute | 1 |
| `--max-parallel M` | Max concurrent Daytona sandboxes | 3 |

Each rollout creates its own ephemeral sandbox, runs the full lifecycle, then destroys the sandbox. One rollout failing does not cancel others.

Results are saved to `output/parallel_run_{task_id}_{timestamp}/` with per-rollout artifacts and an aggregated `summary.json` (see [Verifiers — Run Output Layout](./verifiers.md#run-output-layout)).

## Sandbox Lifecycle

1. Sandbox created from a Docker-in-Docker snapshot
2. Environment files (compose, .env, custom tools) uploaded
3. `docker compose up` runs inside the sandbox
4. Seed data injected
5. Agent executes the task
6. Verifier checks results
7. Sandbox destroyed

State is persisted in `environments/<env-name>/daytona-state.json` between operations.
