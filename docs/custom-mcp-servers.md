# Custom MCP Servers

You can add custom MCP servers at ```env init``` so the reference agent can call MCP tools directly alongside built-in SimLab tools.

## Transport Patterns

SimLab supports two MCP transport patterns:

- **URL-based**: SimLab connects directly to an HTTP MCP endpoint. No extra container is added.
- **Command-based**: SimLab adds an `mcp-gateway` container that starts your stdio MCP servers and exposes them over HTTP inside the environment.

## Config Format

The config file must be a JSON object with a top-level `mcpServers` key:

```json
{
  "mcpServers": {
    "docs": {
      "url": "https://example.com/mcp"
    },
    "weather": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/adhikasp/mcp-weather.git", "mcp-weather"],
      "env": {
        "ACCUWEATHER_API_KEY": "replace-me"
      }
    }
  }
}
```

### Rules

- Each server must define exactly one of `url` or `command`.
- Server names may contain only letters, numbers, `_`, and `-`.
- Server names must not collide with built-in SimLab tool server names (e.g. `email`, `calendar`).
- `args` is optional for command-based servers.
- `env` is optional for command-based servers — use it for API keys and secrets.

## Adding MCP Servers at Init

```bash
# The command is the same regardless of transport type; the behavior depends on
# what's in mcp-config.json (url entries → direct connect, command entries → gateway).
simlab env init my-mcp-env --mcp-servers mcp-config.json --non-interactive

# Combine with a template
simlab env init my-mcp-env --template hr --mcp-servers mcp-config.json --non-interactive
```

After `env init`, SimLab persists your MCP config as `environments/<env-name>/mcp-servers.json`. If the config includes command-based servers, SimLab also generates:

- `environments/<env-name>/mcp-gateway-config.json`
- An `mcp-gateway` service in `docker-compose.yml`

## MCP Env Vars and API Keys

For command-based servers, set secrets in `environments/<env-name>/.env` before running tasks.

### Simple Case (unique env var names)

If an env var name is used by only one MCP server, set it directly:

```bash
# environments/my-mcp-env/.env
ACCUWEATHER_API_KEY=your-real-key
```

### Scoped Case (shared env var names)

If multiple MCP servers use the same env var name, use the scoped form:

```bash
# environments/my-mcp-env/.env
SIMLAB_MCP_WEATHER__API_KEY=weather-key
SIMLAB_MCP_DOCS__API_KEY=docs-key
```

Server names are normalized by uppercasing and replacing non-alphanumeric characters with `_`. For example, `my-docs` becomes `SIMLAB_MCP_MY_DOCS__API_KEY`.

### Resolution Order

1. `SIMLAB_MCP_<SERVER>__<KEY>` (scoped)
2. Raw `<KEY>`, but only when exactly one command-based server declares that key
3. The default value from `mcp-servers.json`

## Runtime Behavior

When you run tasks, the reference agent gets tools from both catalog tool servers and MCP servers:
- **URL-based** servers are contacted directly.
- **Command-based** servers are reached through the MCP gateway container.

The gateway container is built from source in the env dir by default. For building or pushing the gateway image yourself, see `src/simlab/gateway/README.md`.

## Local Smoke Test

To verify the full MCP flow without external auth, use the demo server:

```bash
cd cli/simlab/examples/demo-mcp-server
docker build -t simlab-demo-mcp .
docker run --rm -p 8081:8081 simlab-demo-mcp
```

Then, from `cli/simlab` in another terminal:

```bash
simlab --environments-dir ./environments env init my-demo-env \
  --mcp-servers examples/mcp-servers-local-demo.json --non-interactive

simlab --environments-dir ./environments tasks run \
  --env my-demo-env --tasks-dir examples --task list-tools \
  --agent-model gpt-4o-mini
```

You still need a reference-agent API key (e.g. `OPENAI_API_KEY` or `SIMLAB_AGENT_API_KEY`).

## Example Configs

See the `examples/` directory for sample configs:

- `examples/mcp-servers-url-only.json`
- `examples/mcp-servers-command-only.json`
- `examples/mcp-servers-mixed.json`
- `examples/mcp-servers-local-demo.json`
