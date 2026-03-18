# Simlab CLI v0.1 — Quickstart

## Install

Install the published package:

```bash
uv tool install simulationlab
```

If you want Daytona support from the packaged CLI:

```bash
uv tool install "simulationlab[daytona]"
```

The PyPI package is named `simulationlab`. The installed CLI command is `simlab`.

## Prerequisites

- Python 3.13
- Docker + Docker Compose
- Collinear API key for SimLab commands
- `OPENAI_API_KEY` set in your environment

If you are running from this repo, use:

```bash
uv run simlab <command>
```

If you installed the CLI as a package, use:

```bash
simlab <command>
```

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
- `daytona-state.json` — (when using `--daytona`) Daytona sandbox state
- `verifiers/` — (optional) env-scoped verifier cache when running tasks with `--env`

**Listing (tools, templates, tasks)** uses the Scenario Manager API by default (`https://rl-gym-api.collinear.ai`).

- Collinear auth comes from `collinear_api_key` in `config.toml`, `SIMLAB_COLLINEAR_API_KEY`, or the root `--collinear-api-key` flag.
- Scenario Manager URL comes from `scenario_manager_api_url` in `config.toml`, `SIMLAB_SCENARIO_MANAGER_API_URL`, or root `--scenario-manager-api-url`.
- Environments root comes from `environments_dir` in `config.toml`, `SIMLAB_ENVIRONMENTS_DIR`, or root `--environments-dir`.

```bash
simlab tools list
simlab templates list
simlab tasks list --env my-env    # tasks for the template in that env
simlab tasks list --tasks-dir ./generated-tasks   # or a local bundle
```

---

## 1) Start a Local Environment

**`env init`** creates the env dir, writes `env.yaml`, and generates `docker-compose.yml` and `.env`. **`env up`** starts the containers; it requires `docker-compose.yml` to exist (run `env init` first). To regenerate compose after editing `env.yaml`, run **`simlab env init my-env --force`**. To tear down and start containers again without regenerating compose, use **`simlab env up my-env --rebuild`**.

### A) Use a template

```bash
simlab env init my-env --template hr
simlab env up my-env
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
simlab env up my-env
```

Optional (remote sandbox):

```bash
simlab env up my-env --daytona
```

- **`--force`** (init): overwrite an existing environment without prompting; use to regenerate `docker-compose.yml` and `.env` from the current `env.yaml`.
- **`--rebuild`** (up): run `docker compose down` then `docker compose up -d` (tear down and start containers again). Does not regenerate compose files.

---

## 1.5) Generate Custom Tasks (optional)

Generate your own task definitions using the task generation pipeline:

```bash
# Quick start with a preset
simlab tasks-gen init --preset recruiting --output-dir ./taskgen

# Interactive wizard
simlab tasks-gen init --output-dir ./taskgen

# Validate before running
simlab tasks-gen validate ./taskgen/config.toml

# Run generation
simlab tasks-gen run --config ./taskgen/config.toml
```

Available presets: `recruiting`, `people_mgmt`, `coding`, `customer_support`.

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

Then run a specific task by ID. For `hr`, an available task is `100_weaver_schedule_phone_screen`:

```bash
simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

(You can also set `[agent].model` / `[agent].api_key` in `config.toml`, or use `SIMLAB_AGENT_MODEL`, `SIMLAB_AGENT_API_KEY`, and `OPENAI_API_KEY` for the OpenAI fallback.)

For a generated local task bundle, point `tasks run` at the bundle directory:

```bash
simlab tasks run --env my-env --tasks-dir ./generated-tasks --task generated-task-id --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

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

Use `--daytona` when the environment is up with Daytona; omit it for local Docker.
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

If `--agent-import-path` is omitted, the CLI uses the baked‑in reference agent.

### Run output layout

**Single rollout** (default) writes to a namespaced directory under `output/`:

- `output/agent_run_<task_id>_<timestamp>/artifacts.json` — run artifacts (messages, tool calls, etc.).
- If the task defines **verifiers**, the CLI runs them after the agent and writes Harbor-style reward files in the same run directory:
  - `output/agent_run_<task_id>_<timestamp>/verifier/reward.txt` — `1` or `0`.
  - `output/agent_run_<task_id>_<timestamp>/verifier/reward.json` — e.g. `{"reward": 1.0}`.

Tasks that have no `verifiers` (or `evaluators`) in their JSON only produce `artifacts.json`; there is no verifier step and no reward files.

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

`summary.json` includes completed/failed counts, average reward, average steps, and per-rollout details.

---

## 4) Verifiers

When a task JSON includes a `verifiers` (or `evaluators`) list with `func: python_module` and a `module` path, the CLI runs those verifiers after the agent and writes reward files under that run's directory (see **Run output layout** above).

**When using the API** (default or `scenario_manager_api_url` in env): the CLI downloads the verifier bundle from the Scenario Manager API on first use and caches it under `environments/<env-name>/verifiers/` (e.g. `environments/my-env/verifiers/`).

```bash
simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

Available tasks with verifiers include `100_weaver_schedule_phone_screen` under the `hr` template and `0_flag_biased_compensation_adjustment_request` under the `hr_people_management` template.

Verifiers run **locally**. You must configure the credentials for the LLM-as-a-judge when a verifier uses one. Configure judge settings via the `[verifier]` section in global config or via env variables:

```bash
export SIMLAB_VERIFIER_MODEL="gpt-5.2"
export SIMLAB_VERIFIER_PROVIDER="openai"
export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY"
```

Telemetry settings are also global. Use `[telemetry]` in `config.toml` or `SIMLAB_DISABLE_TELEMETRY=1` to disable CLI telemetry.

---

## 5) How to test

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
# Run a task (env must be up: simlab env up my-env, or use --daytona)
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

To run a task you still need an environment with tool servers: `simlab env up my-env` (local Docker) or `simlab env up my-env --daytona`. Then:

```bash
uv run simlab tasks run --env my-env --task 100_weaver_schedule_phone_screen --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

The API serves verifier bundles via `GET /scenarios/{scenario_id}/verifiers/bundle`; it needs the monorepo `src/` (mounted in Docker) to read scenario verifier files.

---

## 6) Hosted API Authentication

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

## 7) Tear Down

```bash
simlab env down my-env
```

---

## Notes

- **Scenario Manager** auth uses `collinear_api_key` in `config.toml`, `SIMLAB_COLLINEAR_API_KEY`, or the root `--collinear-api-key` flag. The API endpoint uses `scenario_manager_api_url`, `SIMLAB_SCENARIO_MANAGER_API_URL`, or root `--scenario-manager-api-url`. Not used for the LLM.
- The **reference agent** (LiteLLM) uses the `[agent]` config section, then `SIMLAB_AGENT_*` env vars, then `OPENAI_API_KEY` as an OpenAI-specific fallback. Override on the command line with `--agent-model`, `--agent-api-key`, etc. See [LiteLLM docs](https://docs.litellm.ai/) for provider-specific environment variables.
- `simlab tasks run` executes the **built‑in reference agent** unless `--agent-import-path` is provided.
- For MCP‑direct support and "installed agent" mode, see the Future Improvements section in `DESIGN.md`.
- Most simlab-related env vars use the **SIMLAB_** prefix (e.g. `SIMLAB_COLLINEAR_API_KEY`, `SIMLAB_SCENARIO_MANAGER_API_URL`, `SIMLAB_ENVIRONMENTS_DIR`, `SIMLAB_VERIFIER_*`, `SIMLAB_AGENT_*`). For Daytona, `SIMLAB_DAYTONA_API_KEY` is used when set; otherwise `DAYTONA_API_KEY` is used.
