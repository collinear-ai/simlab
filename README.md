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
  <a href="https://discord.gg/HZ4xqpf8VC"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>
</p>

---

Agents fail in production on multi-step workflows: malformed tool calls, state drift, unrecoverable retry loops. Traditional evals don't catch these. SimLab is a self-serve CLI for spinning up realistic simulation environments, running your agent through long-horizon tasks, and verifying the results programmatically.

SimLab is toolset, agent harness, and sandbox agnostic. Browse pre-built scenario templates or bring your own CLI/MCP toolset.

- **Simulate realistic workflows** — spin up environments with seeded data, tool servers, and NPC interactions that mirror production
- **Run any agent** against tasks using any LLM provider (OpenAI, Gemini, Anthropic, Fireworks, or custom endpoints)
- **Run Harbor tasks directly** from a Harbor task directory with `tasks run --harbor`
- **Generate custom tasks** with built-in task generation pipelines
- **Verify programmatically** — deterministic verifiers score pass/fail on actual environment state, not LLM-as-judge
- **Scale to the cloud** with [Daytona](https://daytona.io) for remote sandbox execution — no local Docker required

## How it works

1. **Pick a scenario** — choose from pre-built templates (HR, coding, project management, etc.)
2. **Run your agent** — SimLab handles seeding, tool servers, and orchestration
3. **Get a verdict** — programmatic verifiers score pass/fail with detailed execution traces
4. *(Optional)* **Generate more tasks** — use the built-in task generation pipeline to create custom tasks for your scenario

## Quickstart

### Install

```bash
uv tool install --python 3.13 "simulationlab[daytona]"
```

Requires Python 3.13.

### Authenticate

You need two keys to get started: a Collinear API key and an LLM provider key. Daytona is optional (omit `--daytona` to run locally via Docker).

```bash
simlab auth login                          # saves Collinear key (required)
export SIMLAB_AGENT_API_KEY="sk-..."       # your LLM key — OpenAI/Anthropic/etc (required)
export DAYTONA_API_KEY="dtn_..."           # optional — omit to use local Docker
# Provider examples: openai, anthropic, gemini, groq, mistral, together_ai, deepseek, openrouter
```

### Run your first task

```bash
simlab templates list                      # see available templates
simlab env init my-env --template hr       # HR workflows: recruiting, onboarding, compensation
simlab tasks list --env my-env
simlab tasks run --env my-env \
  --task hr__0_weaver_flag_biased_compensation_adjustment_request \
  --daytona \
  --agent-model <model> \
  --agent-api-key "$SIMLAB_AGENT_API_KEY"
```

For the full walkthrough — task generation, custom agents, verifiers, and more — see the **[Quickstart Guide](https://github.com/collinear-ai/simlab/blob/main/QUICKSTART.md)**.

### Run a Harbor task directly

If you have a single Harbor task directory, you can run it without creating a
named SimLab environment first:

```bash
simlab tasks run --harbor ./examples/harbor/hello-world \
  --agent-model <model>
```

This compiles the Harbor task into a generated SimLab env and local task
bundle, then runs the normal agent + verifier flow. Add `--daytona` to run the
generated Harbor env in Daytona instead of local Docker. Use `--keep-alive` to
retain the generated Harbor workspace under `output/harbor_runs/` for
inspection after the run.

## API Keys

| Key | Required | How to get it |
|-----|----------|---------------|
| **Collinear API key** | Yes | [platform.collinear.ai](https://platform.collinear.ai) (Developers > API Keys) |
| **LLM API key** | For running agents | Any [LiteLLM-supported](https://docs.litellm.ai/docs/providers) provider (OpenAI, Gemini, Anthropic, Fireworks, etc.) |
| **Daytona API key** | Optional (recommended) | [app.daytona.io](https://app.daytona.io) — cloud sandboxes so you don't need local Docker |

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

[npc_chat]
model = "gpt-4o-mini"       # LLM model for NPC chat responses (default: gpt-4o-mini)
api_key = "sk-..."           # API key (falls back to agent key, then provider env vars)
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
| `SIMLAB_NPC_CHAT_MODEL` | NPC chat LLM model (default: `gpt-4o-mini`) |
| `SIMLAB_NPC_CHAT_API_KEY` | NPC chat API key (falls back to agent key) |
| `SIMLAB_ENVIRONMENTS_DIR` | Root directory for environments |
| `SIMLAB_DISABLE_TELEMETRY` | Set to `1` to disable CLI telemetry |

## CLI Reference

| Command | Description |
|---------|-------------|
| `simlab env init <name>` | Create a new environment (from template or interactive) |
| `simlab env custom-tools add <env> <name>` | Scaffold and enable an env-local custom tool |
| `simlab env down <name>` | Stop and remove environment containers |
| `simlab env seed <name>` | Seed initial data into a running environment |
| `simlab tasks list` | List available tasks for an environment |
| `simlab tasks run` | Run an agent against a task from an env, local bundle, or Harbor task directory |
| `simlab tasks-gen init` | Initialize task generation config (with templates) |
| `simlab tasks-gen validate` | Validate a task generation config |
| `simlab tasks-gen run` | Generate custom tasks via the API |
| `simlab templates list` | List available scenario templates |
| `simlab templates info <name>` | Show details for a specific template |
| `simlab tools list` | List available tool servers |
| `simlab tools info <name>` | Show details for a specific tool server |

Run `simlab --help` for full usage details.

## Documentation

- [Quickstart Guide](https://github.com/collinear-ai/simlab/blob/main/QUICKSTART.md) — full setup and usage walkthrough
- [Env-Local Custom Tools](https://github.com/collinear-ai/simlab/blob/main/docs/custom-tools.md) — add custom tool definitions under one environment
- [Agent Integrations](https://github.com/collinear-ai/simlab/blob/main/docs/agent-integrations.md) — adapter architecture and custom framework integration guide
- [Docs](https://docs.collinear.ai) — complete documentation
- [Collinear Platform](https://platform.collinear.ai) — get your API key

## License

This project is licensed under the [Apache 2.0 License](https://github.com/collinear-ai/simlab/blob/main/LICENSE).

## Contact

Questions or feedback? Reach out to us:

- [nazneen@collinear.ai](mailto:nazneen@collinear.ai)
- [sachin@collinear.ai](mailto:sachin@collinear.ai)
- [adit@collinear.ai](mailto:adit@collinear.ai)
