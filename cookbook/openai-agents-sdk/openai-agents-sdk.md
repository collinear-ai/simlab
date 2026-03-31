# OpenAI Agents SDK Agent for SimLab

This cookbook shows how to integrate your own OpenAI Agents SDK app into SimLab.

The intended split is:

- `openai_agents_sdk.custom_agent`
  - your OpenAI Agents SDK app code
- `openai_agents_sdk.simlab_adapter`
  - thin SimLab `BaseAgent` wrapper used with `--agent-import-path`
- `simlab.agents.adapters.openai_agents`
  - shared SimLab adapter module that exposes environment tools to the OpenAI Agents SDK

## What this is for

Use this recipe when you already have an agent built with the OpenAI Agents SDK and
want to run it inside SimLab without rewriting the agent around SimLab internals.

Your app should keep owning:

- instructions and prompting
- model choice
- agent topology
- output formatting

The SimLab adapter should only own:

- discovering tools from `BaseEnvironment`
- exposing those tools to the SDK
- recording tool activity into `RunArtifacts`
- mapping the SDK result into `context.final_observation`

## Configuration

Set your OpenAI credentials and cookbook-owned runtime vars:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_AGENTS_SDK_MODEL="gpt-4o-mini"
# Optional:
export OPENAI_AGENTS_SDK_INSTRUCTIONS="Use the available tools to complete the task end-to-end."
export OPENAI_AGENTS_SDK_MAX_TURNS="24"
```

If `OPENAI_AGENTS_SDK_MODEL` is unset, the cookbook uses the SimLab run model when
present, then falls back to `gpt-4o-mini`.

## Install

Run from `cli/simlab/cookbook/openai-agents-sdk`:

```bash
uv sync
```

This project installs `simulationlab` from the local repo path `../..`.

## Custom Agent Shape

The sample app is intentionally small:

- one `Agent`
- cookbook-owned instructions
- tools passed in from the SimLab adapter
- `Runner.run_sync(...)` for execution

That keeps the example close to what you may already have in production while
making the SimLab seam obvious.

## SimLab usage

Use the adapter with:

```bash
uv run python -m simlab.cli.main tasks run \
  --env your-env \
  --task your-task \
  --agent-import-path openai_agents_sdk.simlab_adapter:SimLabOpenAIAgentsSDKAgent
```

If you want a local smoke test, you can reuse the bundled example environment and
task bundle from `../langgraph-email-agent/examples/`:

```bash
uv run python -m simlab.cli.main \
  --environments-dir ../langgraph-email-agent/examples \
  env up langgraph-email
```

```bash
uv run python -m simlab.cli.main \
  --environments-dir ../langgraph-email-agent/examples \
  tasks run \
  --env langgraph-email \
  --tasks-dir ../langgraph-email-agent/examples/task-bundle \
  --task email_assistant_summary \
  --agent-import-path openai_agents_sdk.simlab_adapter:SimLabOpenAIAgentsSDKAgent
```

The task bundle is email-centric, but that is acceptable here because the cookbook is
demonstrating integration, not prescribing a specific workflow. The same
adapter shape works for any SimLab environment.
