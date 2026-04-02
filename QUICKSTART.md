# Simlab CLI — Quickstart

## Install

Install the published package:

```bash
uv tool install --python 3.13 simulationlab
```

If you want Daytona support from the packaged CLI:

```bash
uv tool install --python 3.13 "simulationlab[daytona]"
```

The PyPI package is named `simulationlab`. The installed CLI command is `simlab`.

### Install from source

```bash
git clone https://github.com/collinear-ai/simlab.git
cd simlab/cli/simlab
uv tool install --python 3.13 .
# or with extras:
uv tool install --python 3.13 ".[daytona]"
```

Then run with `simlab <command>`. To run directly from the repo without installing:

```bash
uv run simlab <command>
```

## Prerequisites

- Python 3.13
- Docker + Docker Compose
- Collinear API key for SimLab commands
- `OPENAI_API_KEY` set in your environment

Optional global config lives at `~/.config/simlab/config.toml` by default, or at the path in `SIMLAB_CONFIG`.

Example:

```toml
collinear_api_key = "col_..."
scenario_manager_api_url = "https://rl-gym-api.collinear.ai"
environments_dir = "./environments"

[daytona]
api_key = "daytona_..."

[agent]
model = "gpt-4o-mini"
provider = "openai"
api_key = "sk-..."

[verifier]
model = "gpt-4o-mini"
provider = "openai"
api_key = "sk-..."

[tasks]
rollout_format = "default"

[telemetry]
disabled = false
state_path = "/tmp/simlab-telemetry.json"
```

---

## 0) Authenticate

After signing in to `https://platform.collinear.ai`, open the platform dashboard,
click **Developers** in the lower left, and then open the **API Keys** tab in
the Developer Resources pop-up.

Then choose one auth path:

### A) Interactive login

```bash
simlab auth login
```

This prompts for your API key and saves it to `~/.config/simlab/config.toml`.
To verify your stored key and see which API URL is targeted:

```bash
simlab auth status
```

### B) Environment variable or CLI flag

```bash
export SIMLAB_COLLINEAR_API_KEY="<your-collinear-api-key>"
simlab --collinear-api-key "<your-collinear-api-key>" <command>
```

All SimLab commands are blocked until `collinear_api_key` is set in
`config.toml`, `SIMLAB_COLLINEAR_API_KEY` is set, or you provide the root CLI
flag.

---

## Environments layout

All environment data lives under **`environments/<env-name>/`**. The environments root is configurable:

- **config.toml** (optional key `environments_dir`)
- **Env var** `SIMLAB_ENVIRONMENTS_DIR`
- **CLI** `--environments-dir` (root option)

Default when unset: `./environments` relative to the current working directory.

Each environment directory contains:

- `env.yaml` — template, tools, overrides
- `docker-compose.yml` — generated compose file
- `.env` — generated env file
- `custom-tools/` — (optional) env-local tool definitions
- `mcp-servers.json` — (optional) custom MCP server config when using `--mcp-servers` at init
- `mcp-gateway-config.json` — (optional) generated for the MCP gateway when you have command-based MCP servers
- `daytona-state.json` — (when using `--daytona`) Daytona sandbox state
- `verifiers/` — (optional) env-scoped verifier cache when running tasks with `--env`

**Listing (tools, templates, tasks)** uses the Scenario Manager API by default (`https://rl-gym-api.collinear.ai`).

- Collinear auth comes from `collinear_api_key` in `config.toml`, `SIMLAB_COLLINEAR_API_KEY`, or the root `--collinear-api-key` flag.
- Scenario Manager URL comes from `scenario_manager_api_url` in `config.toml`, `SIMLAB_SCENARIO_MANAGER_API_URL`, or root `--scenario-manager-api-url`.
- Environments root comes from `environments_dir` in `config.toml`, `SIMLAB_ENVIRONMENTS_DIR`, or root `--environments-dir`.

```bash
simlab tools list
simlab templates list
simlab templates list --env my-env   # uses that env's scenario_manager_api_url
simlab tasks list --env my-env    # tasks for the template in that env
simlab tasks list --tasks-dir ./generated-tasks   # or a local bundle
```

---

## 1) Initialize an Environment

**`env init`** creates the env dir, writes `env.yaml`, and generates `docker-compose.yml` and `.env`. You do not need to run `env up` separately — `tasks run` automatically starts the environment, seeds data, and tears it down when done. To regenerate generated files after editing `env.yaml`, `custom-tools/*.yaml`, or `mcp-servers.json`, run **`simlab env init my-env --force`**. Interactive `env up`, `tasks run`, and `tasks seed` will also prompt if generated files are stale.

### A) Use a template

```bash
simlab env init my-env --template hr
```

`env init --template ...` stores the template's resolved scenario guidance in `env.yaml` as
`scenario_guidance_md`. That field is the runtime source of truth for prompt-level scenario guidance.

If you already have guidance in a markdown file, you can import it at init time:

```bash
simlab env init my-env --template hr --scenario-guidance-file ./guidance.md
```

The file is read once and stored in `env.yaml` as `scenario_guidance_md`. Simlab does not keep a
runtime dependency on the original file.

To customize guidance for a non-template or already-generated environment, edit
`environments/<env-name>/env.yaml` directly and add or change `scenario_guidance_md`, then regenerate:

```yaml
name: my-env
tools:
  - email
scenario_guidance_md: |
  # Scenario Guidance
  Prefer specialized workflows over generic ones.
  Verify important actions before sending messages.
```

```bash
simlab env init my-env --force
```

### B) Pick tools interactively

```bash
simlab env init my-env
```

### C) Custom MCP servers

You can add **custom MCP servers** at init so the reference agent can call MCP tools directly.

SimLab supports two MCP transport patterns:

- **URL-based**: SimLab connects straight to an HTTP MCP endpoint. No extra container is added.
- **Command-based**: SimLab adds an `mcp-gateway` container. The gateway starts your stdio MCP servers and exposes them over HTTP inside the environment.

The config file must be a JSON object with a top-level `mcpServers` object:

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

Rules:

- Each server must define exactly one of `url` or `command`.
- Server names may contain only letters, numbers, `_`, and `-`.
- Server names must not collide with built-in SimLab tool server names such as `email` or `calendar`.
- `args` is optional for command-based servers.
- `env` is optional for command-based servers and is the right place to declare required variables such as API keys.

Examples:

- [examples/mcp-servers-url-only.json](examples/mcp-servers-url-only.json)
- [examples/mcp-servers-command-only.json](examples/mcp-servers-command-only.json)
- [examples/mcp-servers-mixed.json](examples/mcp-servers-mixed.json)
- [examples/mcp-servers-local-demo.json](examples/mcp-servers-local-demo.json)

```bash
# URL-only
simlab env init my-mcp-env --mcp-servers examples/mcp-servers-url-only.json --non-interactive

# Command-based — adds MCP gateway; edit .env for API keys
simlab env init my-mcp-env --mcp-servers examples/mcp-servers-command-only.json --non-interactive

# Mix catalog tools + MCP (template + MCP)
simlab env init my-mcp-env --template hr_recruiting --mcp-servers examples/mcp-servers-mixed.json --non-interactive
```

After `env init`, SimLab persists your MCP config as `environments/<env-name>/mcp-servers.json`.
If the config includes any command-based servers, SimLab also generates:

- `environments/<env-name>/mcp-gateway-config.json`
- an `mcp-gateway` service in `docker-compose.yml`

#### MCP env vars and API keys

For command-based servers, put secrets in `environments/<env-name>/.env` before `simlab env up`.

If an env var name is used by only one MCP server, you can set it directly:

```bash
# environments/my-mcp-env/.env
ACCUWEATHER_API_KEY=your-real-key
```

If multiple MCP servers use the same env var name, use the scoped form instead:

```bash
# environments/my-mcp-env/.env
SIMLAB_MCP_WEATHER__API_KEY=weather-key

### D) Env-local custom tools

You can scaffold environment-specific tool definitions without editing the
built-in catalog:

```bash
simlab env custom-tools add my-env harbor-main
```

That command will:

- create `environments/my-env/custom-tools/harbor-main.yaml`
- add `harbor-main` to `env.yaml`
- regenerate the generated environment files immediately

Use `--force` to overwrite an existing scaffold:

```bash
simlab env custom-tools add my-env harbor-main --force
```

After editing the YAML by hand, run:

```bash
simlab env init my-env --force
```

To inspect an env-local tool:

```bash
simlab tools info harbor-main --env my-env
```
SIMLAB_MCP_DOCS__API_KEY=docs-key
```

Resolution order for command-based MCP env vars:

1. `SIMLAB_MCP_<SERVER>__<KEY>`
2. Raw `<KEY>`, but only when exactly one configured command-based server declares that key
3. The default value from `mcp-servers.json`

Server names are normalized for scoped env vars by uppercasing and replacing non-alphanumeric characters with `_`. For example, `my-docs` becomes `SIMLAB_MCP_MY_DOCS__API_KEY`.

This means a config like:

```json
{
  "mcpServers": {
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

is typically paired with:

```bash
# environments/my-mcp-env/.env
ACCUWEATHER_API_KEY=your-real-key
```

You can also place the same variables under the `mcp-gateway` service in `docker-compose.yml`, but `.env` is the intended default.

When you run tasks, the reference agent gets tools from both catalog tool servers and MCP servers. URL-based MCP servers are contacted directly; command-based MCP servers are reached through the gateway. The gateway container is built from source in the env dir by default; to build or push the gateway image yourself, see **src/simlab/gateway/README.md**.

#### Local smoke test

To verify the full MCP flow without external auth, use the demo server:

```bash
cd cli/simlab/examples/demo-mcp-server
docker build -t simlab-demo-mcp .
docker run --rm -p 8081:8081 simlab-demo-mcp
```

Then, from `cli/simlab` in another terminal:

```bash
simlab --environments-dir ./environments env init my-demo-env --mcp-servers examples/mcp-servers-local-demo.json --non-interactive
simlab --environments-dir ./environments tasks run --env my-demo-env --tasks-dir examples --task list-tools --agent-model gpt-4o-mini
```

You still need a reference-agent API key such as `OPENAI_API_KEY` or `SIMLAB_AGENT_API_KEY`.

- **`--force`** (init): overwrite an existing environment without prompting; use to regenerate `docker-compose.yml` and `.env` from the current `env.yaml`.
- If you want to experiment with large-scale parallel rollouts, reach out to us directly or join the Discord!
---

## 1.5) Generate Custom Tasks (optional)

Generate your own task definitions using the task generation pipeline:

```bash
# Quick start with a template
simlab tasks-gen init --template recruiting --output-dir ./taskgen

# Interactive wizard
simlab tasks-gen init --output-dir ./taskgen

# Validate before running
simlab tasks-gen validate ./taskgen/config.toml

# Run generation
simlab tasks-gen run --config ./taskgen/config.toml
```

Available templates: `recruiting`, `people_mgmt`, `coding`, `customer_support`.

---

## 2) Choose a Task

List the tasks available for the scenario/template stored in your env:

```bash
simlab tasks list --env my-env
```

If you generated a local task bundle, browse that bundle directly:

```bash
simlab tasks list --tasks-dir ./generated-tasks
```

**Important:** Before running tasks, configure your reward model credentials so task verification works correctly. Set these as environment variables, or add a `[verifier]` section to your `config.toml` (see Step 4 for details).

```bash
export SIMLAB_VERIFIER_MODEL="gpt-5.2"
export SIMLAB_VERIFIER_PROVIDER="openai"
export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY"
```

Then run a specific task by ID. `tasks run` automatically starts the environment, seeds data, runs the agent, verifies, and tears down when done. For `hr`, an available task is `100_weaver_schedule_phone_screen`:

```bash
simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --agent-model gpt-5.2 --agent-api-key "$OPENAI_API_KEY"
```

(You can also set `[agent].model` / `[agent].api_key` in `config.toml`, or use `SIMLAB_AGENT_MODEL`, `SIMLAB_AGENT_API_KEY`, and `OPENAI_API_KEY` for the OpenAI fallback.)

For a generated local task bundle, point `tasks run` at the bundle directory:

```bash
simlab tasks run --env my-env --tasks-dir ./generated-tasks --task generated-task-id --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

For a single Harbor task directory, point `tasks run` at the Harbor task
instead of an env:

```bash
simlab tasks run --harbor ./examples/harbor/hello-world --agent-model gpt-5.2
```

Harbor runs compile the task into a generated env and local task bundle first,
then execute the usual startup, agent, and verifier flow. `--daytona` works for
Harbor runs too. `--keep-alive` retains the generated Harbor workspace under
`output/harbor_runs/`.

To emit an ATIF trajectory for any task run, pass
`--tasks-rollout-format atif` or set `rollout_format: atif` in the env's
`env.yaml`. You can also set `[tasks].rollout_format` in global config or
`SIMLAB_TASKS_ROLLOUT_FORMAT=atif`. Harbor runs default to `atif` unless you
override them.

Current Harbor limits:

- `--harbor` runs a single Harbor task directory, not a suite directory
- `--skip-env-setup` is not supported with `--harbor`
- `--rollout-count > 1` is not supported with `--harbor`

If the environment is already running (e.g. from a previous run or manual `env up`), `tasks run` detects it and skips startup/teardown.

---

## 3) Run With a Built‑In or Custom Agent

The **reference agent** (used by default) uses [LiteLLM](https://docs.litellm.ai/) and gets its settings from global config first, then env vars, then CLI flags. Global config lives under `[agent]`: `model`, `provider`, `api_key`, and `base_url`. Env aliases are `SIMLAB_AGENT_MODEL`, `SIMLAB_AGENT_PROVIDER`, `SIMLAB_AGENT_API_KEY`, and `SIMLAB_AGENT_BASE_URL`. For OpenAI only, `OPENAI_API_KEY` is also accepted as a fallback when `[agent].api_key` is unset. CLI flags like `--agent-model` and `--agent-api-key` override all of these. These options apply only to the reference agent; custom agents (`--agent-import-path`) use their own configuration.

### A) Built‑in/reference agent (default)

```bash
simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

### B) With Daytona

```bash
simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --daytona --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

Use `--daytona` to run in a Daytona sandbox; omit it for local Docker.
The Daytona API key is resolved from `--daytona-api-key`, `[daytona].api_key` in `config.toml`, `SIMLAB_DAYTONA_API_KEY`, then `DAYTONA_API_KEY`.

### C) Parallel rollouts (Daytona only)

Run the same task multiple times in parallel to collect diverse agent trajectories for training or evaluation:

```bash
simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --daytona \
  --rollout-count 5 --max-parallel 3 \
  --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

- `--rollout-count N` — total number of rollouts to execute (default: 1).
- `--max-parallel M` — max concurrent Daytona sandboxes (default: 3).

Each rollout creates its own ephemeral sandbox (no `env up` needed), runs the full lifecycle (setup → seed → agent → verify → teardown), then destroys the sandbox. One rollout failing does not cancel others.

Results are saved to `output/parallel_run_{task_id}_{timestamp}/` with per-rollout artifacts and an aggregated `summary.json` (see **Run output layout** below).

### D) Custom agent

```bash
simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --agent-import-path path.to.agent:MyAgent
```

(Custom agents use their own credentials; `--agent-model` and `--agent-api-key` do not apply.)

Your agent must implement the `BaseAgent` contract and populate `RunArtifacts` during execution. Custom agents use their own credentials; the CLI's agent API key and model settings do not apply.

If you are integrating a framework-backed agent, start with
[`docs/agent-integrations.md`](./docs/agent-integrations.md). SimLab now ships
an adapter layer for framework-neutral tool descriptors and artifact recording,
plus an optional LangChain/LangGraph bridge behind the `langchain` extra.

If `--agent-import-path` is omitted, the CLI uses the baked‑in reference agent.

### Run output layout

**Single rollout** (default) writes to a namespaced directory under `output/`:

- `output/agent_run_<task_id>_<timestamp>/artifacts.json` — default rollout artifacts (messages, tool calls, etc.).
- `output/agent_run_<task_id>_<timestamp>/agent/trajectory.json` — ATIF rollout artifacts when `rollout_format` resolves to `atif`.
- If the task defines **verifiers**, the CLI runs them after the agent and writes Harbor-style reward files in the same run directory:
  - `output/agent_run_<task_id>_<timestamp>/verifier/reward.txt` — `1` or `0`.
  - `output/agent_run_<task_id>_<timestamp>/verifier/reward.json` — e.g. `{"reward": 1.0}`.

Tasks that have no `verifiers` (or `evaluators`) in their JSON only produce the
selected rollout artifact file; there is no verifier step and no reward files.

**Parallel rollouts** (`--rollout-count > 1`) write to:

```
output/parallel_run_<task_id>_<timestamp>/
  rollout_0/
    artifacts.json
    verifier/reward.txt
    verifier/reward.json
  rollout_1/
    ...
  summary.json
```

When `rollout_format` resolves to `atif`, each rollout directory writes
`agent/trajectory.json` instead of `artifacts.json`.

`summary.json` includes completed/failed counts, average reward, average steps, and per-rollout details.

---

## 4) Verifiers

When a task JSON includes a `verifiers` (or `evaluators`) list with `func: python_module` and a `module` path, the CLI runs those verifiers after the agent and writes reward files under that run's directory (see **Run output layout** above).

**When using the API** (default or `scenario_manager_api_url` in env): the CLI downloads the verifier bundle from the Scenario Manager API on first use and caches it under `environments/<env-name>/verifiers/` (e.g. `environments/my-env/verifiers/`).

```bash
simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

Available tasks with verifiers include `100_weaver_schedule_phone_screen` under the `hr` template and `0_flag_biased_compensation_adjustment_request` under the `hr_people_management` template.

Verifiers run **locally**. You must configure the credentials for the reward model when a verifier uses one. Configure reward model settings via the `[verifier]` section in global config or via env variables:

```bash
export SIMLAB_VERIFIER_MODEL="gpt-5.2"
export SIMLAB_VERIFIER_PROVIDER="openai"
export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY"
```

Telemetry settings are also global. Use `[telemetry]` in `config.toml` or `SIMLAB_DISABLE_TELEMETRY=1` to disable CLI telemetry.

---

## 5) Send Feedback

Have ideas or hit an issue? Send feedback directly from the CLI:

```bash
# Inline
simlab feedback "The task gen took a long time and I wasn't sure if it was stuck"

# Interactive (prompts for input)
simlab feedback

# With environment context
simlab feedback --env my-env "Seeding seems slow for this template"
```

Feedback is sent through the telemetry pipeline to the Collinear team. If telemetry is disabled (`SIMLAB_DISABLE_TELEMETRY=1`), the command will let you know and suggest emailing `simlab@collinear.ai` instead.

---

## 6) How to test

### Unit tests (CLI)

From the repo root or `cli/simlab`:

```bash
cd cli/simlab
uv run pytest -v --ignore=tests/e2e
```

### Clear verifier cache (after changing shims or testing fresh)

Verifier bundles are cached under `environments/<env-name>/verifiers/` (e.g. `environments/my-env/verifiers/`). Clear that directory for an env so the next run re-downloads and uses the current CLI verifier runtime (e.g. after changing `verifier_runtime/` or `collinear.core.verifier`):

```bash
rm -rf environments/my-env/verifiers
```

### Listing and run against hosted API (no local server)

Uses default `https://rl-gym-api.collinear.ai`. No Scenario Manager or monorepo needed.

```bash
cd cli/simlab
simlab env init my-env --template hr
simlab tools list
simlab templates list
simlab tasks list --env my-env
# tasks run auto-starts the environment, runs the agent, and tears down when done
uv run simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

Verifiers are downloaded from the API on first run and cached under `environments/my-env/verifiers/`.

### Run against local Scenario Manager API

Use your own Scenario Manager API (e.g. in the monorepo) instead of the hosted one. This only starts the **API and its database** (scenarios, tasks, verifier bundles). It does not start a Daytona sandbox or tool servers.

```bash
# From repo root: start the Scenario Manager API + Postgres only
docker compose up -d postgres scenario-manager-api
export SIMLAB_SCENARIO_MANAGER_API_URL=http://localhost:9011
cd cli/simlab
simlab templates list
simlab tasks list --env my-env
```

To run a task, `tasks run` auto-starts the environment (local Docker or `--daytona`):

```bash
uv run simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

The API serves verifier bundles via `GET /scenarios/{scenario_id}/verifiers/bundle`; it needs the monorepo `src/` (mounted in Docker) to read scenario verifier files.

---

## 7) Hosted API Authentication

If you are using hosted Collinear APIs, configure your key first:

```bash
# Recommended — interactive login that saves to config.toml
simlab auth login

# Check stored key, source, and target API URL
simlab auth status
```

Or use an environment variable / CLI flag:

```bash
export SIMLAB_COLLINEAR_API_KEY="<your-collinear-api-key>"
simlab --collinear-api-key "<your-collinear-api-key>" <command>
```

---

## 8) Tear Down

`tasks run` automatically tears down the environment when the run completes. If you started the environment manually with `env up`, or need to tear it down separately:

```bash
simlab env down my-env
```

---

## Notes

- **Scenario Manager** auth uses `collinear_api_key` in `config.toml`, `SIMLAB_COLLINEAR_API_KEY`, or the root `--collinear-api-key` flag. The API endpoint uses `scenario_manager_api_url`, `SIMLAB_SCENARIO_MANAGER_API_URL`, or root `--scenario-manager-api-url`. Not used for the LLM.
- The **reference agent** (LiteLLM) uses the `[agent]` config section, then `SIMLAB_AGENT_*` env vars, then `OPENAI_API_KEY` as an OpenAI-specific fallback. Override on the command line with `--agent-model`, `--agent-api-key`, etc. See [LiteLLM docs](https://docs.litellm.ai/) for provider-specific environment variables.
- `simlab tasks run` executes the **built‑in reference agent** unless `--agent-import-path` is provided.
- **Custom MCP servers** are supported at env init via `--mcp-servers <path-to-json>`. The runtime uses an MCP client directly (no /tools or /step bridge). See `examples/README.md` for config examples and walkthrough.
- Most simlab-related env vars use the **SIMLAB_** prefix (e.g. `SIMLAB_COLLINEAR_API_KEY`, `SIMLAB_SCENARIO_MANAGER_API_URL`, `SIMLAB_ENVIRONMENTS_DIR`, `SIMLAB_VERIFIER_*`, `SIMLAB_AGENT_*`). For Daytona, `SIMLAB_DAYTONA_API_KEY` is used when set; otherwise `DAYTONA_API_KEY` is used.
