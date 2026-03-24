# Agent Baselining

Establish your agent's baseline performance by running it across a set of tasks, collect scores, and review the results. This gives you a starting point to measure improvements against as you iterate on prompts, tools, or harnesses.

## Prerequisites

- **SimLab** installed with Daytona support:
  ```bash
  uv add "simlab[daytona] @ git+https://github.com/collinear-ai/simlab.git"
  ```
- **API keys** exported:
  ```bash
  export SIMLAB_COLLINEAR_API_KEY="col_..."   # from platform.collinear.ai
  export DAYTONA_API_KEY="dtn_..."            # from app.daytona.io
  export OPENAI_API_KEY="sk-..."              # or your provider's key
  ```
- **Verifier** configured (used to score task results):
  ```bash
  export SIMLAB_VERIFIER_MODEL="gpt-5.2"
  export SIMLAB_VERIFIER_PROVIDER="openai"
  export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY"
  ```

## Step 1: Create your environment

Pick a pre-built template or build a custom environment.

**Option A — From a template:**

```bash
# See what's available
simlab templates list

# Create an environment from a template
simlab env init my-env --template <template>
```

**Option B — Custom environment:**

```bash
# Interactive — pick tools from the catalog
simlab env init my-env

# Or bring your own MCP servers
simlab env init my-env --mcp-servers ./mcp-servers.json
```

> **Note:** Custom environments (no template) require a local task bundle for listing and running tasks. Generate one with `simlab tasks-gen` or provide an existing directory via `--tasks-dir`. All `tasks list` and `tasks run` commands below must include `--tasks-dir <path>` when using a custom environment.

> See the [QUICKSTART](../QUICKSTART.md) for full details on MCP server configuration and custom environments.

## Step 2: Start the environment on Daytona

```bash
simlab env up my-env --daytona
```

This provisions a remote sandbox on Daytona with all the tool servers defined in your environment.

## Step 3: List available tasks

```bash
simlab tasks list --env my-env
```

Pick one or more task IDs to baseline against. If you generated a local task bundle, use `--tasks-dir` instead:

```bash
simlab tasks list --tasks-dir ./generated-tasks
```

## Step 4: Run parallel rollouts

Collect statistically meaningful data by running multiple rollouts across the tasks you want to baseline. Pass multiple task IDs to `--task`:

```bash
simlab tasks run \
  --env my-env \
  --task <task_id_1> <task_id_2> <task_id_3> \
  --daytona \
  --rollout-count 5 \
  --max-parallel 3 \
  --agent-model <model> \
  --agent-provider <provider> \
  --agent-api-key "$AGENT_API_KEY"
```

If using a local task bundle, add `--tasks-dir ./generated-tasks` to the command.

- `--task` — one or more task IDs, space-separated.
- `--rollout-count N` — total rollouts to execute per task.
- `--max-parallel M` — max concurrent Daytona sandboxes.

Each rollout gets its own ephemeral sandbox. Output is written to:

```
output/parallel_run_<task_id>_<timestamp>/
  rollout_0/
    artifacts.json
    verifier/reward.txt
    verifier/reward.json
  rollout_1/
    ...
  summary.json            # Aggregated: completed/failed counts, avg reward, avg steps
```

## Step 5: Review results

Open `summary.json` for each parallel run to see:

- **Average reward** — your agent's success rate on that task.
- **Average steps** — how many actions the agent needed.
- **Per-rollout details** — individual scores and durations.

Track these numbers as your baseline. When you change your agent's model, prompts, or tools, re-run the same tasks and compare.

## Step 6: Tear down

```bash
simlab env down my-env --daytona
```

## Next steps

- **Generate custom tasks** — Use `simlab tasks-gen` to create tasks tailored to your use case.
- **Try a custom agent** — Pass `--agent-import-path path.to.agent:MyAgent` to benchmark your own agent implementation.
- **Compare models** — Re-run the same tasks with a different `--agent-model` and compare summary scores side by side.
