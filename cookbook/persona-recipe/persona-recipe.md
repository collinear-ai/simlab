# Custom NPC Personas

Define your own NPC personas so generated tasks include realistic stakeholder interactions tailored to your domain.

## Prerequisites

- **SimLab** installed:
  ```bash
  uv add "simlab @ git+https://github.com/collinear-ai/simlab.git"
  ```
- **API keys** exported:
  ```bash
  export SIMLAB_COLLINEAR_API_KEY="col_..."   # from platform.collinear.ai
  ```

## Step 1: Initialize a config

Start from a template to get a working config.toml with example NPCs already defined:

```bash
# See available templates
simlab tasks-gen templates

# From a template
simlab tasks-gen init --template hr --output-dir ./persona-demo

# Or start blank and build your own
simlab tasks-gen init --output-dir ./persona-demo
```

This creates `persona-demo/config.toml` with toolset, scenario, workflows, and a default set of NPCs.

> **Tip:** Available templates: `hr`, `coding`, `crm`, `customer-support`, `erp`, `project-management`, `blank`.

## Step 2: Customize your NPCs

Open `persona-demo/config.toml` and find the `[[npcs]]` sections near the bottom. Each NPC has two fields:

- **`role`** — The persona's job title or archetype (e.g., "VP of Engineering", "Frustrated Customer")
- **`typical_asks`** — What this persona typically requests or asks the agent about

Here's what the HR template gives you:

```toml
[[npcs]]
role = "Hiring Manager"
typical_asks = "Scheduling preferences, candidate feedback, offer approvals"

[[npcs]]
role = "Candidate"
typical_asks = "Interview logistics, offer details, timeline questions"
```

Replace or extend these with your own personas. For example, to model a customer support scenario with your own stakeholders:

```toml
[[npcs]]
role = "Tier-1 Support Agent"
typical_asks = "Escalation requests, ticket handoffs, knowledge base lookups"

[[npcs]]
role = "Angry Enterprise Customer"
typical_asks = "SLA violation complaints, demands to speak with a manager, refund requests"

[[npcs]]
role = "Product Manager"
typical_asks = "Bug severity assessment, feature request prioritization, release timeline questions"

[[npcs]]
role = "Legal Compliance Officer"
typical_asks = "Data deletion requests, GDPR compliance checks, audit trail reviews"
```

You can define as many NPCs as you need. The task generator will create tasks that involve interactions with these personas — the richer the `typical_asks`, the more varied the generated tasks.

> **Note:** Don't forget to also update the `[agent]`, `[[toolset]]`, and `[scenario]` sections if your NPCs belong to a different domain than the template.

## Step 3: Validate your config

Make sure the config is valid before generating:

```bash
simlab tasks-gen validate persona-demo/config.toml
```

You should see `Config is valid!` along with a summary of your settings.

## Step 4: Generate tasks

Run the task generation pipeline:

```bash
simlab tasks-gen run --config persona-demo/config.toml --out ./persona-demo/output
```

This submits the config to the SimLab server, which runs a multi-step pipeline:

1. Summarizes your scenario and tools
2. Drafts tasks based on your NPCs, categories, and workflows
3. Refines and filters for quality
4. Generates rubrics and verifiers
5. Augments tasks with NPC-specific details (secrets, interaction patterns)

Progress is streamed live in the terminal. The pipeline typically takes 2-5 minutes depending on `num_tasks`.

> **Tip:** The quality filter is strict — not all generated tasks will pass. Set `num_tasks` to 2-3x your target to account for filtering. For example, if you want 10 tasks, set `num_tasks = 25`.

## Step 5: Inspect the output

The generated bundle is written to `persona-demo/output/`:

```
persona-demo/output/
  tasks/          # Task JSON files (includes instructions, NPC refs, seed data)
  rubrics/        # Evaluation rubrics
  verifiers/      # Auto-generated verifier code
  npcs/           # NPC profiles (profiles.json)
  skills.md       # Agent skills document
```

Check how your NPCs appear in the generated tasks:

```bash
# See the generated NPC profiles
cat persona-demo/output/npcs/profiles.json | python -m json.tool

# Look at a task to see NPC interactions
cat "$(ls persona-demo/output/tasks/*.json | head -1)" | python -m json.tool | head -80
```

Each task JSON references your NPCs by profile ID in the `"npcs"` array and weaves their personas into the `"task"` instruction text.

## Next steps

- **Run the tasks** — Use `simlab tasks run` to execute rollouts against the generated tasks. See the [Agent Baselining](../agent-baselining/) recipe.
- **Iterate on personas** — Adjust `typical_asks` to steer the kinds of interactions generated. More specific asks produce more targeted tasks.
- **Add more NPCs** — More personas means more diverse multi-stakeholder tasks. Try adding NPCs with conflicting priorities for harder scenarios.
- **Customize categories** — Edit the `[[categories]]` section to control what types of tasks get generated alongside your NPCs.

## Troubleshooting

- **`simlab: command not found`** — Install with `uv add "simlab @ git+https://github.com/collinear-ai/simlab.git"`.
- **`Config validation failed`** — Check that each `[[npcs]]` entry has both `role` and `typical_asks` fields. Run `simlab tasks-gen validate` for details.
- **No NPC profiles in output** — The `npcs/` directory is populated from seed data. If you're using a custom scenario without seed data, NPC profiles may be empty — the tasks will still reference your NPCs in instructions.
- **Tasks don't mention my NPCs** — Make sure `typical_asks` is descriptive. One-word values produce generic tasks. Aim for a comma-separated list of 3-5 specific interaction types.
