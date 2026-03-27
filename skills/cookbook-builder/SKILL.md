# Cookbook Builder

Brainstorm, design, and scaffold new SimLab cookbook recipes from a rough idea.

## Prerequisites

Before starting, confirm:

1. SimLab is installed: `simlab --version`
2. `SIMLAB_COLLINEAR_API_KEY` is set (needed for discovery commands)

If SimLab is not installed, point the user to `cli/simlab/QUICKSTART.md`. If the API key is missing, ask the user to export it and wait before proceeding.

## Workflow

### 1. Explore SimLab's current capabilities

Run these commands and note the output — you will use this throughout the session:

```bash
simlab templates list
```

```bash
simlab tools list
```

Also read `cli/simlab/cookbook/cookbook-intro.md` to see existing recipes.

Summarize what you found: available templates, available tool servers. Keep this as your reference for what SimLab can do today.

### 2. Brainstorm the recipe

The user has an idea for a cookbook recipe. Ask clarifying questions **one at a time** to refine it. Use multiple-choice questions where possible.

Questions to cover (adapt to what the user already told you):

- What is the goal? What will someone accomplish by following this recipe?
- Who is the target audience? (new SimLab users, agent developers, benchmark authors, etc.)
- Which SimLab features does it use? (specific templates, tool servers, task generation, parallel rollouts, etc.)
- What does success look like? How will the user know the recipe worked?
- Are there external dependencies? (third-party datasets, APIs, tools outside SimLab)

Brainstorming is complete when you can define a concrete step-by-step workflow. The user can also say "that's enough" to move on.

### 3. Gap analysis

Compare what the recipe needs against what you discovered in step 1. Check for:

- **Templates** — Does the recipe need a template not in `simlab templates list`?
- **Tool servers** — Does it need tool servers not in `simlab tools list`?
- **CLI features** — Does it need CLI flags or subcommands that don't exist? Verify with `simlab <subcommand> --help`.
- **Verifiers** — Does it need verification approaches SimLab doesn't support? Check `cli/simlab/src/simlab/verifiers/` and existing template verifiers.

**If no gaps:** Tell the user and proceed to step 4.

**If gaps are found:** For each gap, STOP and present it:

> "This recipe needs [X], but SimLab doesn't currently support it.
>
> 1. **Proceed** — I'll note this as a dependency in the recipe docs
> 2. **Pivot** — let's adjust the idea to work within current capabilities
> 3. **Plan SimLab work** — let's pause the cookbook and plan the SimLab-side implementation first"

Wait for the user's decision on each gap before continuing. Do NOT proceed past this step without resolving all gaps.

### 4. Design the recipe

Based on brainstorming and gap analysis, define:

- The recipe name (kebab-case, e.g., `custom-benchmarks`). The folder name must match the human guide filename: `<recipe-name>/<recipe-name>.md`
- The step-by-step workflow (what the user will do, in order)
- Which supporting files are needed (converter scripts, configs, verifiers)
- Prerequisites (env vars, API keys, tools)
- Outline of the human guide sections
- Outline of the agent guide sections

Present this structure to the user for confirmation before generating drafts.

### 5. Generate drafts

Read `cli/simlab/skills/cookbook-builder/cookbook-conventions.md` for format guidance.

Generate fully fleshed-out drafts:

- **Human guide** (`<recipe-name>.md`) — complete, following the conventions
- **Agent guide** (`SKILL.md`) — complete, following the conventions
- **Supporting files** — working scripts/configs as identified in step 4

Present each draft to the user. Wait for approval before writing any files. If the user requests changes, revise and re-present.

### 6. Scaffold

First, check that `cli/simlab/cookbook/<recipe-name>/` does not already exist:

```bash
ls cli/simlab/cookbook/<recipe-name>/ 2>/dev/null && echo "EXISTS" || echo "OK"
```

If it exists, ask the user whether to overwrite or pick a different name.

Write all files to `cli/simlab/cookbook/<recipe-name>/`.

Then update `cli/simlab/cookbook/cookbook-intro.md` — add a row to the Recipes table:

```
| [recipe-name](recipe-name/) | One-line description. |
```

Tell the user the recipe has been scaffolded and suggest they review the files.

## Troubleshooting

- **`simlab: command not found`** — See `cli/simlab/QUICKSTART.md` for installation instructions.
- **`simlab templates list` returns an error** — Check that `SIMLAB_COLLINEAR_API_KEY` is set.
- **Gap analysis is uncertain** — When unsure whether a feature exists, check `simlab <command> --help` or search the SimLab source at `cli/simlab/src/simlab/`.
