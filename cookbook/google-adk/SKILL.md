# Integrate A Google ADK Agent With SimLab

Use this recipe when you want to run a Google Agent Development Kit agent inside SimLab.

The goal is to keep your ADK agent code intact and add only a thin SimLab adapter that:

- lists tools from the SimLab `BaseEnvironment`
- exposes them to ADK as function tools
- records tool calls and the final response into `RunArtifacts`

## What To Build

Keep the split explicit.

1. Your Google ADK agent code.
2. A SimLab `BaseAgent` adapter used as the `--agent-import-path` entrypoint.
3. ADK tools built from the SimLab `BaseEnvironment`.
4. `RunArtifacts` recording so tool calls and outputs land in SimLab artifacts.

## Workflow

### 1. Confirm ADK model credentials

Pick one of these credential paths.

- Gemini API
  - Set `GOOGLE_API_KEY`
  - Use a Gemini model like `gemini-2.5-flash`
- LiteLLM
  - Set `OPENAI_API_KEY` (or another LiteLLM provider key)
  - Use a provider model string like `openai/gpt-4o-mini`

### 2. Add the SimLab adapter

Create a `BaseAgent` wrapper that:

- accepts the SimLab `instruction`
- builds a `RunArtifactsRecorder(context)`
- exposes SimLab tools as ADK function tools
- runs your ADK agent loop
- stores the final text in `context.final_observation`

### 3. Smoke test

If you already have a target environment and task, use that.

If not, reuse a bundled SimLab environment from another cookbook to prove the integration seam.

