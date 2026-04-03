# Claude Agent SDK + SimLab Integration Guide

## Prerequisites

- Python 3.13+
- An Anthropic API key (`ANTHROPIC_API_KEY`)
- SimLab installed (`pip install simulationlab` or editable from the monorepo)

## Installation

```bash
cd cli/simlab/cookbook/claude-agent-sdk
uv sync
```

## Usage

### With SimLab CLI

```bash
simlab tasks run --env my-env --task <task_id> \
  --agent-import-path claude_agent_sdk_cookbook.simlab_adapter:SimLabClaudeAgentSDKAgent
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `CLAUDE_AGENT_SDK_MODEL` | `claude-sonnet-4-20250514` | Model to use |
| `CLAUDE_AGENT_SDK_INSTRUCTIONS` | (built-in) | Custom system prompt |
| `CLAUDE_AGENT_SDK_MAX_TURNS` | `24` | Max agent loop turns |

## Architecture

The integration has two layers:

1. **Adapter** (`simlab.agents.adapters.claude_agent`) — converts SimLab tools into
   Claude SDK MCP servers and records tool calls back into `RunArtifacts`.

2. **Cookbook** (`claude_agent_sdk_cookbook`) — wraps the adapter in a `BaseAgent`
   subclass that wires up `ClaudeAgentOptions` and calls `query()`.

## How tools work

SimLab tools are grouped by `tool_server` and exposed as in-process MCP servers
via `create_sdk_mcp_server()`. The agent discovers them through the
`mcp__<server>__<tool>` naming convention. Tool errors from SimLab are propagated
as `is_error: True` in the MCP response payload.
