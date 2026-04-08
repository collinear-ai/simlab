# Global Config

SimLab resolves configuration in this order: **config file < environment variables < CLI flags**.

## Config File

Default location: `~/.config/simlab/config.toml`

Override with `--config-file` or the `SIMLAB_CONFIG` environment variable.

### Full Reference

```toml
# Top-level settings
collinear_api_key = "col_..."
scenario_manager_api_url = "https://rl-gym-api.collinear.ai"
environments_dir = "./environments"

# Agent (reference agent powered by LiteLLM)
[agent]
model = "gpt-4o-mini"
provider = "openai"
api_key = "sk-..."
base_url = ""                # optional custom endpoint

# Verifier (reward model for task verification)
[verifier]
model = "gpt-4o-mini"
provider = "openai"
api_key = "sk-..."
base_url = ""

# NPC chat (LLM for NPC responses)
[npc_chat]
model = "gpt-4o-mini"
provider = "openai"
api_key = "sk-..."
base_url = ""

# Daytona (remote sandbox execution)
[daytona]
api_key = "dtn_..."

# Task output format
[tasks]
rollout_format = "default"   # "default" or "atif"

# CLI telemetry
[telemetry]
disabled = false
state_path = "/tmp/simlab-telemetry.json"
```

### Minimal Config

For most users, you only need:

```toml
collinear_api_key = "col_..."

[agent]
model = "gpt-4o-mini"
api_key = "sk-..."
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SIMLAB_COLLINEAR_API_KEY` | Collinear API key |
| `SIMLAB_SCENARIO_MANAGER_API_URL` | Override Scenario Manager API URL |
| `SIMLAB_ENVIRONMENTS_DIR` | Root directory for environments |
| `SIMLAB_AGENT_MODEL` | Agent model name |
| `SIMLAB_AGENT_PROVIDER` | Agent LiteLLM provider |
| `SIMLAB_AGENT_API_KEY` | Agent API key |
| `SIMLAB_AGENT_BASE_URL` | Agent custom endpoint |
| `OPENAI_API_KEY` | Fallback agent API key (when provider is `openai`) |
| `SIMLAB_VERIFIER_MODEL` | Verifier model name |
| `SIMLAB_VERIFIER_PROVIDER` | Verifier LiteLLM provider |
| `SIMLAB_VERIFIER_API_KEY` | Verifier API key |
| `SIMLAB_VERIFIER_BASE_URL` | Verifier custom endpoint |
| `SIMLAB_NPC_CHAT_MODEL` | NPC chat model (default: `gpt-4o-mini`) |
| `SIMLAB_NPC_CHAT_PROVIDER` | NPC chat LiteLLM provider |
| `SIMLAB_NPC_CHAT_API_KEY` | NPC chat API key (falls back to agent key) |
| `SIMLAB_NPC_CHAT_BASE_URL` | NPC chat custom endpoint |
| `SIMLAB_DAYTONA_API_KEY` | Daytona API key |
| `DAYTONA_API_KEY` | Daytona API key (alternative) |
| `SIMLAB_TASKS_ROLLOUT_FORMAT` | Output format (`default` or `atif`) |
| `SIMLAB_DISABLE_TELEMETRY` | Set to `1` to disable CLI telemetry |
| `SIMLAB_TELEMETRY_STATE_PATH` | Path for persisted telemetry state |
| `SIMLAB_CONFIG` | Override config file path |

### Provider-Specific API Key Fallbacks

When `[agent].api_key` is unset, SimLab falls back to the standard environment variable for the configured provider:

| Provider | Env Var |
|----------|---------|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `together_ai` | `TOGETHERAI_API_KEY` |
| `mistral` | `MISTRAL_API_KEY` |
| `cohere` | `COHERE_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |

## CLI Flags

Root-level flags override everything:

```bash
simlab --collinear-api-key "col_..." \
       --scenario-manager-api-url "http://localhost:9011" \
       --environments-dir ./my-envs \
       <command>
```

Task-run flags override agent settings:

```bash
simlab tasks run --agent-model gpt-4o --agent-api-key "$OPENAI_API_KEY" ...
```

## Environments Directory

All environment data lives under `environments/<env-name>/`. The root is configurable:

1. `--environments-dir` CLI flag (highest priority)
2. `SIMLAB_ENVIRONMENTS_DIR` env var
3. `environments_dir` in `config.toml`
4. Default: `./environments` relative to working directory

### Environment Directory Contents

Each environment directory contains:

| File | Description |
|------|-------------|
| `env.yaml` | Environment definition — template, tools, overrides |
| `docker-compose.yml` | Generated compose file |
| `.env` | Generated environment variables |
| `custom-tools/` | (optional) Env-local tool definitions |
| `mcp-servers.json` | (optional) Custom MCP server config |
| `mcp-gateway-config.json` | (optional) Generated for command-based MCP servers |
| `daytona-state.json` | (optional) Daytona sandbox state |
| `verifiers/` | (optional) Cached verifier bundles |

## Authentication

### Interactive Login (recommended)

```bash
simlab auth login
```

Prompts for your API key and saves it to `~/.config/simlab/config.toml`.

```bash
simlab auth status    # verify stored key and target API URL
```

### Environment Variable

```bash
export SIMLAB_COLLINEAR_API_KEY="col_..."
```

### CLI Flag

```bash
simlab --collinear-api-key "col_..." <command>
```

Get your Collinear API key at [platform.collinear.ai](https://platform.collinear.ai) — click **Developers** in the lower left, then open the **API Keys** tab.
