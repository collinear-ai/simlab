# MCP gateway (command-based MCP servers)

This directory contains the **MCP gateway** used when an environment has command-based MCP servers. The gateway reads a JSON config, starts each server as a stdio subprocess, and exposes them over streamable HTTP on port 8080.

## Local use (no manual build)

When you run `simlab env init` with a config that includes command-based MCP servers, the CLI copies this gateway source (Dockerfile, requirements.txt, run_gateway.py) into the environment directory at `environments/<env-name>/gateway/`. When you run `simlab env up`, Docker Compose builds the image from that copied context and starts the service. **You do not need to build or push the image yourself** for normal local or Daytona use. No image name or tag is hardcoded in the CLI — the generated compose uses only `build`; the image gets Compose’s default name (e.g. `my-env-mcp-gateway`).

## Building the image manually

**Recommended (works whether you run from repo or have the CLI installed):** Use the gateway files already copied into an environment:

```bash
cd environments/my-mcp-env/gateway
docker build -t simlab-mcp-gateway:latest .
```

Run `simlab env init my-mcp-env --mcp-servers ...` first so that env has a `gateway/` directory.

**From the repo** (only when developing from the simlab source tree):

```bash
cd cli/simlab
docker build -t simlab-mcp-gateway:latest -f src/simlab/gateway/Dockerfile src/simlab/gateway
```

**When the CLI is installed as a package** (no repo clone): Use the “from an environment” method above, or locate the gateway source in the install and build from it:

```bash
# Print the gateway dir (e.g. .../site-packages/simlab/gateway)
python -c "from pathlib import Path; import simlab.composer.engine as e; print(Path(e.__file__).resolve().parent.parent / 'gateway')"
# Then build from that path, e.g.:
GATEWAY_DIR=$(python -c "from pathlib import Path; import simlab.composer.engine as e; print(Path(e.__file__).resolve().parent.parent / 'gateway')")
docker build -t simlab-mcp-gateway:latest -f "$GATEWAY_DIR/Dockerfile" "$GATEWAY_DIR"
```

## Pushing to a registry

1. Build and tag with your registry and optional version (use one of the build methods above, e.g. from an env’s `gateway/` or from the installed package path):

   ```bash
   docker build -t ghcr.io/myorg/simlab-mcp-gateway:0.1.0 -f Dockerfile .   # from env's gateway/
   docker push ghcr.io/myorg/simlab-mcp-gateway:0.1.0
   ```

2. To use the pre-built image instead of building from source, edit the generated `environments/<env-name>/docker-compose.yml`: replace the `mcp-gateway` service’s `build` block with `image: ghcr.io/myorg/simlab-mcp-gateway:0.1.0`. The service name in compose stays `mcp-gateway` (the CLI and tasks run use that name for the gateway URL). The container still expects the config file at `/config/mcp-gateway-config.json` (mounted from the env dir) and env vars as defined in the compose file.

## Config and runtime

- The gateway reads config from the `CONFIG_PATH` env var (the generated compose sets it to `/config/mcp-gateway-config.json`) or from the `MCP_GATEWAY_CONFIG` env var (JSON string).
- It listens on port 8080 and exposes the MCP streamable HTTP endpoint at `/mcp`.
- Command-based server env defaults are stored in `mcp-gateway-config.json`, but container env can override them at runtime.
- Use `SIMLAB_MCP_<SERVER>__<KEY>` for an explicit per-server override.
- Raw `<KEY>` also works when that key is declared by exactly one command-based server in the config.
- If multiple command-based servers share the same env key (for example both use `API_KEY`), use the scoped names in `.env` or compose `environment`:

```bash
SIMLAB_MCP_WEATHER__API_KEY=weather_real_key
SIMLAB_MCP_DOCS__API_KEY=docs_real_key
```

- Override precedence for each subprocess is:
  1. `SIMLAB_MCP_<SERVER>__<KEY>`
  2. Raw `<KEY>` when unique across command-based servers
  3. The default from `mcp-gateway-config.json`
