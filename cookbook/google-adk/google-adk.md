# Google ADK Agent for SimLab

This cookbook shows how to run a Google Agent Development Kit agent inside SimLab.

The intended split is:

- `google_adk_agent.custom_agent`
  - your Google ADK agent code and runner loop
- `google_adk_agent.simlab_adapter`
  - thin SimLab `BaseAgent` wrapper used with `--agent-import-path`
- `simlab.agents.adapters.google_adk`
  - shared SimLab adapter module that exposes environment tools to ADK

## What this is for

Use this recipe when you want a Google ADK agent to run in SimLab without rewriting
your agent around SimLab internals.

Your app should keep owning:

- instructions and prompting
- model choice
- output formatting

The SimLab adapter should only own:

- discovering tools from `BaseEnvironment`
- exposing those tools to ADK
- recording tool activity into `RunArtifacts`
- mapping the ADK result into `context.final_observation`

## Configuration

Set the ADK model and credentials.

### Option A Gemini API

```bash
export GOOGLE_API_KEY="..."
export GOOGLE_ADK_MODEL="gemini-2.5-flash"
```

### Option B LiteLLM with OpenAI

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_ADK_MODEL="openai/gpt-4o-mini"
```

Optional cookbook owned instructions.

```bash
export GOOGLE_ADK_INSTRUCTIONS="Use the available tools to complete the user's request end-to-end. Base factual claims on tool output. Keep the final response concise."
```

Demo instructions for the bundled email plus SEC EDGAR smoke test.

```bash
export GOOGLE_ADK_INSTRUCTIONS="Use the available tools to complete the user request end-to-end. Base factual claims on tool output. For inbox triage, call email-env__search_emails with a high enough limit to see the key threads, then call email-env__get_email for the CEO, finance, Northstar, Microsoft renewal, and legal emails before writing the final response. In the sales call brief, use the exact labels from the task prompt without extra markdown formatting on the labels. For public company context, call edgar-mcp__get_company_info with identifier MSFT and use its output. Keep the final response concise."
```

## Install

Run from `cli/simlab/cookbook/google-adk`:

```bash
uv sync
```

This project installs `simulationlab` from the local repo path `../..`.

## SimLab usage

Use the adapter with:

```bash
uv run python -m simlab.cli.main tasks run \
  --env your-env \
  --task your-task \
  --agent-import-path google_adk_agent.simlab_adapter:SimLabGoogleADKAgent
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
  --agent-import-path google_adk_agent.simlab_adapter:SimLabGoogleADKAgent
```

The task bundle is email-centric, but that is acceptable here because the cookbook is
demonstrating integration, not prescribing a specific workflow. The same adapter shape
works for any SimLab environment.
