# Agent Baselining

Run an agent across SimLab tasks on Daytona to establish baseline performance scores.

## Prerequisites

Before starting, confirm these are in place:

1. SimLab is installed with Daytona support: `simlab --version`
2. `SIMLAB_COLLINEAR_API_KEY` is set (from platform.collinear.ai)
3. `DAYTONA_API_KEY` is set (from app.daytona.io)
4. An agent API key is set (e.g., `OPENAI_API_KEY`)
5. Verifier is configured: `SIMLAB_VERIFIER_MODEL`, `SIMLAB_VERIFIER_PROVIDER`, and `SIMLAB_VERIFIER_API_KEY`

If any are missing, tell the user which env vars to export and wait before proceeding.

## Workflow

### 1. Gather inputs

Ask the user:
- What model and provider do you want to baseline? (e.g., `gpt-5.2` via `openai`)
- Do you have a preferred template, or do you want to see what's available?

If the user isn't sure about templates, run:

```bash
simlab templates list
```

Show the output and let them pick.

### 2. Create the environment

```bash
simlab env init baseline-env --template <template>
```

If the user wants a custom environment (no template), run `simlab env init baseline-env` without `--template` and follow the interactive prompts. If they have an MCP servers config, use `--mcp-servers <path>`.

**Important:** Custom environments (no template) require a local task bundle. The user must either generate tasks with `simlab tasks-gen` or provide an existing `--tasks-dir` path. All subsequent `tasks list` and `tasks run` commands must include `--tasks-dir <path>`.

### 3. Start the environment on Daytona

```bash
simlab env up baseline-env --daytona
```

Wait for this to complete before proceeding.

### 4. List available tasks

```bash
# Template-based environment:
simlab tasks list --env baseline-env

# Custom environment with a local task bundle:
simlab tasks list --tasks-dir <path>
```

Note the task IDs.

### 5. Run parallel rollouts

Run multiple rollouts across all tasks to get statistically meaningful scores. Pass all task IDs to `--task`:

```bash
simlab tasks run \
  --env baseline-env \
  --task <task_id_1> <task_id_2> <task_id_3> \
  --daytona \
  --rollout-count 5 \
  --max-parallel 3 \
  --agent-model <model> \
  --agent-provider <provider> \
  --agent-api-key "$AGENT_API_KEY"
```

If using a local task bundle, add `--tasks-dir <path>` to the command.

### 6. Collect and summarize results

Read every `summary.json` produced by the parallel runs:

```bash
cat output/parallel_run_<task_id>_*/summary.json
```

For each task, extract and compute:
- **Success rate** — average reward across rollouts (1.0 = all passed, 0.0 = all failed)
- **Efficiency** — average number of agent steps per rollout
- **Reliability** — number of completed vs failed rollouts (failures may indicate timeouts, tool errors, or agent crashes)
- **Variance** — spread between best and worst rollout scores; high variance suggests the agent's performance is inconsistent
- **Failure patterns** — for failed rollouts, read the corresponding `artifacts.json` to identify common failure modes (e.g., stuck in loops, wrong tool usage, misunderstood instructions)

If any tasks had a 0% success rate, briefly inspect one failed rollout's `artifacts.json` and note the likely cause.

### 7. Tear down

```bash
simlab env down baseline-env --daytona
```

### 8. Present results

Show the user a baseline report with:

**Per-task summary table:**

| Task | Success Rate | Avg Steps | Completed | Failed | Variance |
|------|-------------|-----------|-----------|--------|----------|

**Overall stats:**
- Total tasks run
- Overall success rate (across all tasks and rollouts)
- Best and worst performing tasks

**Failure analysis** (if any tasks scored below 100%):
- Which tasks failed and how often
- Common failure modes observed in `artifacts.json` (e.g., tool misuse, instruction misinterpretation, timeout)
- Actionable suggestions: if failures cluster around specific tool interactions, suggest reviewing the agent's tool-use prompts; if failures are random, suggest increasing rollout count for more signal

This is their baseline. They can re-run after changing models, prompts, or tools to measure improvement.

## Troubleshooting

- **`simlab: command not found`** — Install with `uv add "simlab[daytona] @ git+https://github.com/collinear-ai/simlab.git"`.
- **`SIMLAB_COLLINEAR_API_KEY` not set** — Get a key from platform.collinear.ai (Developers > API Keys).
- **`DAYTONA_API_KEY` not set** — Get a key from app.daytona.io.
- **No reward files** — The task may not have verifiers defined. Check `simlab tasks list` for tasks with verifiers.
