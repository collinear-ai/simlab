# LangGraph Email Assistant for SimLab

This cookbook is a standalone customer-style project. It owns its own dependencies, configuration, and runtime code, then plugs into SimLab through `--agent-import-path`.

## What is here

- `langgraph_email_agent.email_assistant`
  - generic LangGraph email assistant that takes any LangChain chat model plus LangChain tools
- `langgraph_email_agent.simlab_adapter`
  - `BaseAgent` adapter for SimLab
- `examples/langgraph-email/`
  - minimal SimLab environment config with the `email` HTTP tool server plus a direct SEC EDGAR MCP server
- `examples/task-bundle/`
  - local task bundle that exercises both `email-env` and `edgar-mcp`, plus local programmatic verifiers

## Sample tasks

The cookbook now includes three local tasks:

- `email_assistant_summary`
  - triage a busy inbox, produce a prioritized todo list, and add public-company context to a Microsoft renewal brief
- `meeting_prep_packet`
  - prepare a customer meeting brief with action items plus SEC EDGAR-backed public-company context
- `counterparty_brief_apple`
  - prepare a renewal brief for Apple and include public-company context

## Configuration

This cookbook does not read `SIMLAB_*` variables.

Set the cookbook-owned runtime vars instead:

```bash
export LANGGRAPH_EMAIL_BACKEND="openai-compatible"
export LANGGRAPH_EMAIL_MODEL="gpt-5.2"
export OPENAI_API_KEY="sk-..."
# Optional:
export LANGGRAPH_EMAIL_BASE_URL="http://localhost:4000/v1"
export LANGGRAPH_EMAIL_TEMPERATURE="0"
export LANGGRAPH_EMAIL_MAILBOX_OWNER="agent@weaverenterprises.com"
```

## Install

Run from `cli/simlab/cookbook/langgraph-email-agent`:

```bash
uv sync
```

This project installs `simulationlab` from the local repo path `../..`.
Use `uv run python -m simlab.cli.main ...` for the cookbook commands below. That is
more reliable than `uv run simlab ...` because it does not depend on a console-script
shim being present in the current environment, and it does not depend on any unrelated
active venv.

## Standalone Python usage

```python
from langgraph_email_agent.email_assistant import LangGraphEmailAssistant
from langgraph_email_agent.model_factory import build_chat_model_from_env

model = build_chat_model_from_env()
assistant = LangGraphEmailAssistant(model=model, max_steps=8)
```

The standalone assistant expects LangChain `BaseTool` / `StructuredTool` instances. The SimLab adapter provides that bridge automatically.

## Built-in workflows

The standalone assistant ships with a few opinionated workflows:

- `inbox-triage`
  - rank inbox threads by urgency, business impact, and deadline risk
- `todo-builder`
  - turn important threads into concrete prioritized actions
- `sales-call-brief`
  - normalize a sales or renewal thread into an account/stage/blockers/next-step summary with external company context
- `meeting-prep-packet`
  - prepare a focused customer meeting brief with agenda, stakeholders, risks, and public-company context

The assistant also scopes inbox work to a single mailbox owner. By default the SimLab adapter uses `agent@weaverenterprises.com`, and you can override that with `LANGGRAPH_EMAIL_MAILBOX_OWNER`.

## SimLab smoke test

1. From `cli/simlab/cookbook/langgraph-email-agent`, generate the local environment files:

```bash
uv run python -m simlab.cli.main --environments-dir ./examples env init langgraph-email --force --non-interactive --mcp-servers ./examples/langgraph-email/mcp-servers.json
```

This example keeps email as an HTTP tool server for seeding and exposes SEC EDGAR through the command-based MCP gateway. The generated `.env` includes a default `SEC_EDGAR_USER_AGENT` value for the example; edit it if you want a different identity string.
`env init` also generates `docker-compose.yml`, `mcp-gateway-config.json`, and
the `gateway/` build context under `examples/langgraph-email/`; those generated
files are not source-owned cookbook assets.

2. Start the example environment:

```bash
uv run python -m simlab.cli.main --environments-dir ./examples env up langgraph-email
```

3. Run the local task bundle with the cookbook adapter:

```bash
uv run python -m simlab.cli.main --environments-dir ./examples tasks run \
  --env langgraph-email \
  --tasks-dir ./examples/task-bundle \
  --task email_assistant_summary \
  --agent-import-path langgraph_email_agent.simlab_adapter:SimLabLangGraphEmailAgent
```

Swap `--task email_assistant_summary` for `meeting_prep_packet` or `counterparty_brief_apple` to try the other bundled examples.

The summary task seeds 10 emails, scopes the assistant to one mailbox, and asks for public-company context in the sales brief without explicitly naming the tool. The verifier still checks that the workflow performed the company lookup. Depending on the task, the final response should contain either:

- `## Inbox Triage`
- `## Todo List`
- `## Sales Call Brief`

or:

- `## Todo List`
- `## Meeting Prep Packet`

The example inbox includes executive, customer-risk, sales, and low-priority threads. The task also includes a local Python verifier, so you do not need reward-model configuration for this smoke test. Artifacts are written under `./output/.../artifacts.json`, and verifier reward files are written under `./output/.../verifier/`.

4. Tear down the environment:

```bash
uv run python -m simlab.cli.main --environments-dir ./examples env down langgraph-email
```
