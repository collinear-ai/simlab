# Integrate A Claude Agent SDK Agent With SimLab

Use this recipe when you already have an agent built with the Claude Agent SDK and
want to run it in SimLab.

The goal is to preserve your app code and add only a thin SimLab adapter.

## What To Build

For a SimLab integration, keep the split explicit:

1. Your Claude Agent SDK app code.
2. SimLab `BaseAgent` adapter used as the `--agent-import-path` entrypoint.
3. MCP tool servers built from the SimLab `BaseEnvironment`, preferably via the shared adapter module.
4. `RunArtifacts` recording so tool calls and final output land in SimLab artifacts.

## Workflow

### 1. Inspect your SDK agent

Read the existing Claude Agent SDK entrypoint and determine:

- where `ClaudeAgentOptions` is constructed
- how tools are attached (MCP servers, allowed_tools)
- how the run is executed (`query()` or `ClaudeSDKClient`)
- where the final output is consumed (`ResultMessage.result`)

### 2. Preserve your app code

Keep business logic, prompts, and agent topology in your module.

Do not move that logic into the SimLab adapter unless the existing code has no clean
boundary at all.

### 3. Add the SimLab adapter

Create a `BaseAgent` wrapper that:

- accepts the SimLab `instruction`
- builds a `RunArtifactsRecorder(context)`
- exposes SimLab tools as Claude SDK MCP servers via `build_claude_agent_tools()`
- runs your SDK agent
- stores the final text in `context.final_observation`

### 4. Validate the contract

Before end-to-end testing, confirm:

- the adapter has a stable `name()`
- `setup()` does not depend on global state
- tool calls are recorded in `RunArtifacts`
- empty outputs become explicit errors

### 5. Smoke test

If you already have a target environment and task, use that.

If not, use an existing local SimLab cookbook example to prove the integration seam.
