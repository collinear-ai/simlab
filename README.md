<p align="center">
  <img src="https://raw.githubusercontent.com/collinear-ai/simlab/main/assets/simlab-banner.png" alt="SimLab" width="700">
</p>

<p align="center">
  <em>A self-serve simulation lab, enabling you to build and refine long-horizon AI agents.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/simulationlab/"><img src="https://img.shields.io/pypi/v/simulationlab?style=flat-square" alt="PyPI"></a>
  <a href="https://docs.collinear.ai"><img src="https://img.shields.io/badge/Docs-docs.collinear.ai-blue?style=flat-square" alt="Docs"></a>
  <a href="https://collinear.ai"><img src="https://img.shields.io/badge/Website-collinear.ai-orange?style=flat-square" alt="Website"></a>
  <img src="https://img.shields.io/badge/python-3.13-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.13">
  <a href="https://github.com/collinear-ai/simlab/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License"></a>
  <a href="https://discord.gg/FfHVP6Yc"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>
</p>

---

SimLab is the data layer for adaptively composing RL simulations and evaluating and refining agents. SimLab is toolset, agent harness and sandbox agnostic. Browse pre-built scenario templates or bring your own CLI/MCP toolset.

- **Browse & compose** environments from a catalog of tool servers and scenario templates
- **Run agents** against tasks using any LLM provider (OpenAI, Fireworks, custom endpoints)
- **Generate custom tasks** with built-in task generation pipelines
- **Evaluate automatically** with verifiers and reward model scoring
- **Scale to the cloud** with Daytona for remote sandbox execution

## Quickstart

### Install
```bash
uv tool install simulationlab
# or, if you want Daytona support:
uv tool install "simulationlab[daytona]"

# pipx works too:
pipx install simulationlab
```

The PyPI package is named `simulationlab`. The installed CLI command is `simlab`.
SimLab currently supports Python 3.13.

Get your API keys and export them:
```bash
export SIMLAB_COLLINEAR_API_KEY="col_..."   # from platform.collinear.ai (Developers > API Keys)
export DAYTONA_API_KEY="dtn_..."            # from app.daytona.io
export OPENAI_API_KEY="sk-..."              # from platform.openai.com/api-keys
export SIMLAB_VERIFIER_MODEL="gpt-5.2"      # reward model (tasks also come with a programmatic verifier so can skip this)
export SIMLAB_VERIFIER_PROVIDER="openai"    # litellm compatible provider name
export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY" # corresponding key
```

### Create an environment:
```bash
# Create and start an environment on Daytona
simlab env init my-env --template hr_recruiting
simlab env up my-env --daytona
```
> To list templates run `simlab templates list`

Create and list tasks in directory `./generated-tasks`
```bash
simlab tasks-gen init --preset recruiting # Can go to the config.toml to setup number of tasks etc.
simlab tasks list --tasks-dir ./generated-tasks # takes 5-10 mins with the default setting, choose haiku and 2 tasks for a faster generation.
```


### Run a task with task id `task_id`.

```bash
simlab tasks run --env my-env \
  --task task_id \
  --tasks-dir ./generated-tasks \
  --agent-model gpt-5.2 \
  --agent-api-key "$OPENAI_API_KEY"
```

### Tear down
```bash
simlab env down my-env --daytona
```

<details>
<summary><strong>Running locally with Docker</strong></summary>

If you have Docker + Docker Compose installed, you can run environments on your machine instead of Daytona:

```bash
simlab env init my-env --template hr_recruiting
simlab env up my-env
simlab env down my-env
```

No `DAYTONA_API_KEY` required. First run may take several minutes while images are pulled/built.

</details>

For the full walkthrough — configuration, custom agents, task generation, verifiers, and more — see the **[Quickstart Guide](https://github.com/collinear-ai/simlab/blob/main/QUICKSTART.md)**.

## API Keys

| Key | Required | How to get it |
|-----|----------|---------------|
| **Collinear API key** | Yes | [platform.collinear.ai](https://platform.collinear.ai) (Developers > API Keys) |
| **OpenAI API key** | For running agents | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Daytona API key** | Yes (default runtime) | [app.daytona.io](https://app.daytona.io) |

## Configuration

SimLab resolves configuration in this order: **config file < environment variables < CLI flags**.

Config file: `~/.config/simlab/config.toml` (override with `--config-file` or `SIMLAB_CONFIG`)

```toml
collinear_api_key = "col_..."

[agent]
model = "gpt-5-mini"
provider = "openai"
api_key = "sk-..."

[daytona]
api_key = "dtn_..."

[verifier]
model = "claude-sonnet-4-6"
api_key = "sk-ant-..."
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SIMLAB_COLLINEAR_API_KEY` | Collinear API key |
| `SIMLAB_AGENT_API_KEY` | Agent API key |
| `OPENAI_API_KEY` | Fallback agent API key (when provider is `openai`) |
| `DAYTONA_API_KEY` | Daytona API key |
| `SIMLAB_DAYTONA_API_KEY` | Daytona API key (alternative) |
| `SIMLAB_SCENARIO_MANAGER_API_URL` | Override Scenario Manager API URL |
| `SIMLAB_VERIFIER_MODEL` | Verifier model |
| `SIMLAB_VERIFIER_API_KEY` | Verifier API key |
| `SIMLAB_ENVIRONMENTS_DIR` | Root directory for environments |
| `SIMLAB_DISABLE_TELEMETRY` | Set to `1` to disable CLI telemetry |

## CLI Reference

| Command | Description |
|---------|-------------|
| `simlab env init <name>` | Create a new environment (from template or interactive) |
| `simlab env up <name>` | Start environment containers (local Docker or `--daytona`) |
| `simlab env down <name>` | Stop and remove environment containers |
| `simlab env seed <name>` | Seed initial data into a running environment |
| `simlab tasks list` | List available tasks for an environment |
| `simlab tasks run` | Run an agent against a task and evaluate results |
| `simlab tasks-gen init` | Initialize task generation config (with presets) |
| `simlab tasks-gen validate` | Validate a task generation config |
| `simlab tasks-gen run` | Generate custom tasks via the API |
| `simlab templates list` | List available scenario templates |
| `simlab templates info <name>` | Show details for a specific template |
| `simlab tools list` | List available tool servers |
| `simlab tools info <name>` | Show details for a specific tool server |

Run `simlab --help` for full usage details.

## Documentation

- [Quickstart Guide](https://github.com/collinear-ai/simlab/blob/main/QUICKSTART.md) — full setup and usage walkthrough
- [Docs](https://docs.collinear.ai) — complete documentation
- [Collinear Platform](https://platform.collinear.ai) — get your API key

## License

This project is licensed under the [Apache 2.0 License](https://github.com/collinear-ai/simlab/blob/main/LICENSE).

## Contact

Questions or feedback? Reach out to us:

- [nazneen@collinear.ai](mailto:nazneen@collinear.ai)
- [sachin@collinear.ai](mailto:sachin@collinear.ai)
- [adit@collinear.ai](mailto:adit@collinear.ai)
