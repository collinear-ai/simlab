# Task Generation

Generate custom task definitions using SimLab's task generation pipeline. The pipeline runs server-side — you provide a config, SimLab generates tasks with instructions, rubrics, verifiers, and NPCs.

## Quick Start

```bash
# View available task gen templates
simlab tasks-gen templates

# Initialize from a template
simlab tasks-gen init --template hr --output-dir ./taskgen

# Review and edit the config
cat ./taskgen/config.toml

# Validate before running
simlab tasks-gen validate ./taskgen/config.toml

# Run generation (takes 5-10 min with defaults)
simlab tasks-gen run --config ./taskgen/config.toml --out ./generated-tasks
```

## Templates

Templates are pre-built config files that give you a working starting point. List available templates:

```bash
simlab tasks-gen templates
```

| Template | Description |
|----------|-------------|
| `hr` | Full HR workflows (recruiting + people management) |
| `coding` | Software engineering tasks |
| `customer-support` | Helpdesk and ticket management |
| `crm` | CRM workflows |
| `erp` | ERP workflows |
| `project-management` | Project management workflows |
| `blank` | Minimal starter — just the structure, you fill in the details |

```bash
simlab tasks-gen init --template coding --output-dir ./taskgen
```

This copies the template TOML into your output directory. Edit it to customize the agent role, tools, scenarios, workflows, NPCs, categories, and generation settings before running.

### Starting Without a Template

If none of the templates match your use case, run `init` without `--template`:

```bash
simlab tasks-gen init --output-dir ./taskgen
```

This creates a minimal `config.toml` with placeholder values. You'll need to fill in:

- **`[agent]`** — your agent's role and what it does
- **`[[toolset]]`** — the tools available to the agent (name, description, operations)
- **`[scenario]`** — scenario name, role label, conventions, and policies
- **`[generation]`** — how many tasks to generate and the complexity distribution

Optional sections (`[[workflows]]`, `[[npcs]]`, `[[categories]]`, `[workspace]`) give the pipeline more context to generate higher-quality tasks. The more detail you provide, the better the output.

You can also pass tool definitions directly at run time instead of writing a full config — see [Alternative Inputs](#alternative-inputs) below.

## Config Structure

The generated `config.toml` has these sections:

```toml
[agent]
role = "HR recruiting coordinator"
description = "Handles end-to-end recruiting workflows..."

[[toolset]]
name = "HRIS"
description = "Query/update employee records"
operations = ["search", "read", "create", "update"]

[[toolset]]
name = "Email"
description = "Send and read emails"
operations = ["send", "read"]

[scenario]
name = "recruiting"
role_label = "HR recruiting professional"
conventions = "- Always check calendars before scheduling\n..."
policies = ["Offers require VP approval for >$200k"]

[workspace]
email_domain = "weaver.com"
agent_email = "hr@weaver.com"

[[workflows]]
name = "Schedule panel interview"
steps = [
  "Check interviewer availability",
  "Create calendar event",
  "Send confirmation email",
  "Update candidate status in HRIS",
]

[[npcs]]
role = "Hiring Manager"
typical_asks = "Scheduling preferences, candidate feedback"

[[categories]]
id = "schedule_interviews"
label = "Schedule interviews"

[generation]
num_tasks = 10

[generation.complexity]
easy = 0.3
medium = 0.5
hard = 0.2

[pipeline]
model = "claude-haiku-4-5"
```

## Validation

Always validate before running:

```bash
simlab tasks-gen validate ./taskgen/config.toml
```

This parses the TOML and checks it against the `TaskGenRequest` schema locally, without hitting the server.

## Running Generation

```bash
simlab tasks-gen run --config ./taskgen/config.toml --out ./generated-tasks
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to config TOML | (required, or use `--tools`/`--from-schema`) |
| `--out` | Output directory for generated tasks | `./generated-tasks` |
| `--num-tasks` | Override number of tasks | From config (default: 10) |
| `--model` | Override pipeline model | From config (default: `claude-haiku-4-5`) |
| `--describe` | Vague description of what to test | — |
| `--from-schema` | Tool schema JSON (OpenAI format) | — |
| `--tools` | MCP tool definitions JSON | — |
| `--verbose` | Show detailed output | Off |

The pipeline shows live progress as it steps through generation, filtering, and quality checks.

### Alternative Inputs

Instead of a full config TOML, you can pass tool definitions directly:

```bash
# From MCP tool definitions JSON
simlab tasks-gen run --tools ./my-tools.json --out ./generated-tasks

# From OpenAI-format tool schema
simlab tasks-gen run --from-schema ./openai-tools.json --out ./generated-tasks --num-tasks 20
```

## Output Bundle

Generated tasks are written to the output directory:

```
generated-tasks/
  tasks/          # Task JSON files
  instructions/   # Task instruction files
  rubrics/        # Evaluation rubrics
  verifiers/      # Verifier modules
  npcs/           # NPC definitions
  skills.md       # Generated skills document
```

## Using Generated Tasks

List your generated tasks:

```bash
simlab tasks list --tasks-dir ./generated-tasks
```

Run a generated task against an environment:

```bash
simlab tasks run --env my-env --tasks-dir ./generated-tasks \
  --task <task-id> --agent-model gpt-4o-mini --agent-api-key "$OPENAI_API_KEY"
```

## Checking Job Status

If you need to check on a running or past generation job:

```bash
simlab tasks-gen status <job-id>
```

## Tips

- **Faster generation**: Use `claude-haiku-4-5` (default) and reduce `num_tasks` to 2-3 for testing.
- **More tasks**: If quality filtering removes too many tasks, increase `num_tasks` to 2-3x your target.
- **Fewer filtered tasks**: Reduce the number of categories or set `filter = false` under `[generation]`.
