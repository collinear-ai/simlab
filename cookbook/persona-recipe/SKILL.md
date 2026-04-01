# Custom NPC Personas

Define custom NPC personas in a SimLab task generation config and generate tasks with tailored stakeholder interactions.

## Prerequisites

Before starting, confirm these are in place:

1. SimLab is installed: `simlab --version`
2. `SIMLAB_COLLINEAR_API_KEY` is set (from platform.collinear.ai)

If SimLab is not installed, point the user to `cli/simlab/QUICKSTART.md`. If the API key is missing, ask the user to export it and wait before proceeding.

## Workflow

### 1. Gather inputs

Ask the user:
- What domain are your NPCs for? (e.g., HR, customer support, sales, or something custom)
- Do you want to start from a template or build from scratch?

If starting from a template, show the options:

```bash
simlab tasks-gen templates
```

Available templates: `hr`, `coding`, `crm`, `customer-support`, `erp`, `project-management`, `blank`.

### 2. Initialize the config

```bash
# From a template
simlab tasks-gen init --template <template> --output-dir ./persona-demo

# Or start blank
simlab tasks-gen init --output-dir ./persona-demo
```

### 3. Customize NPCs

Read the generated config:

```bash
cat persona-demo/config.toml
```

Show the user the existing `[[npcs]]` sections and ask:
- What personas should the agent interact with?
- For each persona: what's their role and what do they typically ask?

Edit the `[[npcs]]` sections in `persona-demo/config.toml` based on the user's answers. Each NPC needs:

```toml
[[npcs]]
role = "<persona role/title>"
typical_asks = "<comma-separated list of things this persona asks about>"
```

Add, remove, or modify NPCs as needed. Also update `[agent]`, `[[toolset]]`, `[scenario]`, and `[[categories]]` if the user's domain differs from the template.

### 4. Validate

```bash
simlab tasks-gen validate persona-demo/config.toml
```

If validation fails, fix the config and re-validate.

### 5. Generate tasks

```bash
simlab tasks-gen run --config persona-demo/config.toml --out ./persona-demo/output
```

Wait for this to complete before proceeding. Progress is streamed live (typically 2-5 minutes).

**Important:** The quality filter is strict — set `num_tasks` to 2-3x the user's target to account for filtering (e.g., `num_tasks = 25` for a target of 10).

### 6. Inspect results

Show the user what was generated:

```bash
ls persona-demo/output/
```

Read the NPC profiles:

```bash
cat persona-demo/output/npcs/profiles.json
```

Show a sample task to see NPC references:

```bash
ls persona-demo/output/tasks/
cat persona-demo/output/tasks/<first-file> | python -m json.tool
```

Point out how the NPCs appear in the task's `"task"` field and `"npcs"` array.

### 7. Present summary

Show the user a summary:

Run these counts and present them in a table:

```bash
echo "Tasks:     $(ls persona-demo/output/tasks/*.json 2>/dev/null | wc -l)"
echo "Rubrics:   $(ls persona-demo/output/rubrics/*.md 2>/dev/null | wc -l)"
echo "Verifiers: $(ls persona-demo/output/verifiers/*.py 2>/dev/null | wc -l)"
echo "NPC profiles: $(cat persona-demo/output/npcs/profiles.json 2>/dev/null | python -m json.tool | grep profile_id | wc -l)"
```

Highlight which NPCs from their config appear in the generated tasks.

## Troubleshooting

- **`simlab: command not found`** — Install with `uv add "simlab @ git+https://github.com/collinear-ai/simlab.git"`.
- **`SIMLAB_COLLINEAR_API_KEY` not set** — Get a key from platform.collinear.ai (Developer Resources > API Keys). Then run: `simlab auth login`.
- **Config validation fails** — Each `[[npcs]]` entry needs both `role` and `typical_asks`. Run `simlab tasks-gen validate` for details.
- **Tasks don't reference NPCs** — Make `typical_asks` more descriptive (3-5 specific interaction types, comma-separated).
