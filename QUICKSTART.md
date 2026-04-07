# SimLab CLI — Quickstart

## Install

```bash
uv tool install --python 3.13 "simulationlab[daytona]"
```

The PyPI package is named `simulationlab`. The installed CLI command is `simlab`.

## Prerequisites

- Python 3.13
- A Collinear API key ([platform.collinear.ai](https://platform.collinear.ai) → Developers → API Keys)
- An API key for any LiteLLM-supported model provider (OpenAI, Anthropic, etc.)
- *(Optional)* A Daytona API key ([app.daytona.io](https://app.daytona.io)) for remote sandbox execution — otherwise tasks run locally via Docker

## Authenticate

```bash
simlab auth login
```

This prompts for your Collinear API key and saves it to `~/.config/simlab/config.toml`. Then export your model provider key:

```bash
# Use whichever provider you prefer — SimLab uses LiteLLM under the hood.
export SIMLAB_AGENT_API_KEY="your-api-key"

# Optional: export Daytona key if using remote sandboxes
export DAYTONA_API_KEY="dtn_..."
```

---

## 1) Run an OOTB Task

Create an environment from a template, pick a task, and run it.

### Create an environment

```bash
simlab env init my-env --template hr
```

> To see all available templates: `simlab templates list`

### List available tasks

```bash
simlab tasks list --env my-env
```

### Run a task

`tasks run` automatically starts the environment, seeds data, runs the agent, verifies the result, and tears down when done.

```bash
# With Daytona (recommended — fast, ephemeral remote sandboxes)
simlab tasks run --env my-env \
  --task hr__0_weaver_flag_biased_compensation_adjustment_request \
  --daytona \
  --agent-model <model> \
  --agent-api-key "$SIMLAB_AGENT_API_KEY"

# Without Daytona (runs locally via Docker — first run may be slow while images pull)
simlab tasks run --env my-env \
  --task hr__0_weaver_flag_biased_compensation_adjustment_request \
  --agent-model <model> \
  --agent-api-key "$SIMLAB_AGENT_API_KEY"
```

Use any [LiteLLM-supported model](https://docs.litellm.ai/docs/providers) for `--agent-model` (e.g. `gpt-4o`, `claude-sonnet-4-20250514`, `gemini/gemini-2.5-pro`).

### View results

Results are saved to `output/agent_run_<task_id>_<timestamp>/`:

- `artifacts.json` — full rollout trace (messages, tool calls, observations)
- `verifier/reward.txt` — `1` (pass) or `0` (fail)
- `verifier/reward.json` — e.g. `{"reward": 1.0}`

---

## 2) Generate and Run Custom Tasks

Use the task generation pipeline to create your own tasks, then run them against the same environment.

### Set up the verifier (reward model)

Generated tasks use rubric-based verifiers that need a model to score results. Configure the verifier before running generated tasks:

```bash
export SIMLAB_VERIFIER_MODEL="<model>"       # e.g. gpt-4o, claude-sonnet-4-20250514
export SIMLAB_VERIFIER_PROVIDER="<provider>" # e.g. openai, anthropic
export SIMLAB_VERIFIER_API_KEY="your-api-key"
```

> Built-in tasks from step 1 use programmatic verifiers and don't require this setup.

### View available task generation templates

```bash
simlab tasks-gen templates
```

### Initialize a task generation config

```bash
simlab tasks-gen init --template hr --output-dir ./taskgen
```

This creates `./taskgen/config.toml`. Edit it to customize the number of tasks, complexity distribution, categories, and more.

### Generate tasks

```bash
simlab tasks-gen run --config ./taskgen/config.toml --out ./generated-tasks
```

This takes 5–10 minutes with defaults. For a faster test run, set `num_tasks = 2` in your config.

### List and run a generated task

```bash
simlab tasks list --tasks-dir ./generated-tasks
```

Pick a task ID from the list and run it:

```bash
simlab tasks run --env my-env \
  --tasks-dir ./generated-tasks \
  --task <task-id> \
  --daytona \
  --agent-model <model> \
  --agent-api-key "$SIMLAB_AGENT_API_KEY"
```

Results are saved to `output/` in the same format as built-in tasks.

> For the full task generation reference — config structure, alternative inputs, tips — see [Task Generation](./docs/task-generation.md).

---

## 3) Bring Your Own Agent

You can run any of the tasks above with your own agent instead of the built-in one. See [Agent Integrations](./docs/agent-integrations.md) for the full guide — contract, adapters (LangChain/LangGraph, OpenAI Agents SDK, Claude Agent SDK), and code examples.

### Run a built-in or custom task with your agent

```bash
simlab tasks run --env my-env \
  --task hr__0_weaver_flag_biased_compensation_adjustment_request \
  --agent-import-path my_package.agent:MyAgent
```

Or with a generated task:

```bash
simlab tasks run --env my-env \
  --tasks-dir ./generated-tasks \
  --task <task-id> \
  --agent-import-path my_package.agent:MyAgent
```

Add `--daytona` to run in a remote sandbox instead of local Docker.

Custom agents use their own model configuration — `--agent-model` and `--agent-api-key` do not apply.

---

Have ideas or hit an issue? Run `simlab feedback` or email simlab@collinear.ai.
