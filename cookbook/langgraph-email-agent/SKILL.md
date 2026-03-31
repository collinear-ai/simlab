# Integrate A LangGraph Agent With SimLab

Use this recipe when the user has an existing LangGraph or LangChain-based agent and wants to run it inside SimLab.

The goal is not to replay the bundled email demo by default. The goal is to adapt the user's own agent onto SimLab's `BaseAgent`, `BaseEnvironment`, and `RunArtifacts` contract. The bundled email assistant in this folder is the reference example when the user does not already have an agent implementation.

## Prerequisites

Before changing code, confirm:

1. You know where the user's LangGraph agent code lives.
2. The relevant package dependencies are installed.
3. The user has a SimLab environment or wants to use the bundled example under `examples/`.

If the user does not yet have a LangGraph agent to integrate, use the bundled example in this folder as the starting point and say that explicitly.

## What To Build

For a SimLab integration, you usually need four pieces:

1. A SimLab `BaseAgent` implementation that becomes the `--agent-import-path` entrypoint.
2. LangChain tools built from the SimLab environment via `simlab.agents.adapters.langchain.build_langchain_tools(...)`.
3. A `RunArtifactsRecorder` so tool calls and outputs are recorded in SimLab artifacts.
4. A final output mapping step that writes the agent's result into `context.final_observation`.

Keep business prompts, graph topology, and model configuration in the user's agent package. Only the SimLab wiring belongs in the adapter layer.

## Workflow

### 1. Inspect the user's agent

Read the LangGraph entrypoint and determine:

- how the graph is invoked
- what input shape it expects
- how tools are passed in
- where the final answer is returned
- whether execution is sync, async, or both

If the graph already accepts LangChain `BaseTool` objects, reuse that seam directly.

### 2. Add the SimLab adapter

Create a `BaseAgent` wrapper that:

- accepts the SimLab `instruction`
- builds a `RunArtifactsRecorder(context)`
- builds tools with `build_langchain_tools(environment, recorder=recorder)`
- invokes the user's graph with those tools
- stores the final text or serialized result in `context.final_observation`

Prefer the shared adapter helpers in `cli/simlab/src/simlab/agents/adapters/` rather than re-implementing tool conversion locally.

### 3. Handle async correctly

SimLab tool enumeration and invocation are async-first.

When integrating with LangGraph or LangChain:

- prefer async graph execution when the app supports it
- prefer adapter utilities that already use `alist_tools()` and `acall_tool()`
- avoid deprecated sync-only environment calls unless there is no async path available

If the user's graph supports both sync and async entrypoints, prefer the async path.

### 4. Validate the agent contract

Before running end-to-end tasks, confirm:

- the agent has a stable `name()`
- `setup()` does not depend on hidden global state
- the adapter can run against a `BaseEnvironment`
- tool calls are recorded in `RunArtifacts`
- the final response lands in `context.final_observation`

If the user's graph returns a structured object, convert it into a deterministic string or JSON string before storing it as the final observation.

### 5. Run a smoke test

If the user already has a target SimLab environment and task, run that.

If not, use the bundled example from `cli/simlab/cookbook/langgraph-email-agent`:

```bash
uv run python -m simlab.cli.main \
  --environments-dir ./examples \
  env init langgraph-email \
  --force \
  --non-interactive \
  --mcp-servers ./examples/langgraph-email/mcp-servers.json
```

```bash
uv run python -m simlab.cli.main --environments-dir ./examples env up langgraph-email
```

```bash
uv run python -m simlab.cli.main --environments-dir ./examples tasks run \
  --env langgraph-email \
  --tasks-dir ./examples/task-bundle \
  --task email_assistant_summary \
  --agent-import-path langgraph_email_agent.simlab_adapter:SimLabLangGraphEmailAgent
```

Use the bundled example to prove the integration seam, not as the default destination for all work.

### 6. Review artifacts and failures

Inspect the latest output and summarize:

- whether the run completed
- whether the verifier passed
- which tools were called
- whether the final output shape matched the task requirements

Useful files:

- `output/agent_run_<task>_<timestamp>/artifacts.json`
- `output/agent_run_<task>_<timestamp>/verifier/reward.txt`
- `output/agent_run_<task>_<timestamp>/verifier/reward.json`

## Troubleshooting

- If tool calls do not appear in artifacts, verify the adapter passed a `RunArtifactsRecorder`.
- If the graph cannot see any tools, check that it is consuming the LangChain tools returned by `build_langchain_tools(...)`.
- If the run fails before tool execution, inspect the graph input/output mapping rather than the SimLab environment first.
- If `env up` fails in the bundled example, rerun `env init` to regenerate the local environment files.
- If the final response is empty or malformed, inspect the graph's terminal node and make the adapter's result mapping explicit.
