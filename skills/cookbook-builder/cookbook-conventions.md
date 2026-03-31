# Cookbook Conventions

Format reference for SimLab cookbook recipes. Each recipe lives in its own folder under `cli/simlab/cookbook/` and contains a human guide, an agent guide, and optional supporting files.

## Human Guide (`<recipe-name>.md`)

Structure:

1. **Title** — `# Recipe Name` followed by a one-line description
2. **Prerequisites** — what needs to be installed and which env vars must be set, with exact `export` commands
3. **Step-by-step workflow** — numbered sections (`## Step N: ...`), each with:
   - Explanation of what's happening and why
   - Bash commands in fenced code blocks
   - Expected output or what to look for
   - Notes/tips in blockquotes where helpful
4. **Next steps** (optional) — suggestions for what to try after completing the recipe
5. **Troubleshooting** (optional) — bullet list of common errors and fixes, if the recipe has known failure modes

### Example structure

~~~
# Recipe Name

One-line description.

## Prerequisites

- **SimLab** installed: ...
- **API keys** exported: ...

## Step 1: ...
## Step 2: ...
## Step N: Tear down

## Next steps
## Troubleshooting
~~~

## Agent Guide (`SKILL.md`)

Structure:

1. **Title** — `# Recipe Name` followed by a one-line description (no YAML frontmatter)
2. **Prerequisites** — numbered checklist of what must be in place. If anything is missing, tell the user what to set and **wait before proceeding**
3. **Workflow** — numbered sections (`### N. Step name`) matching the human guide, each with:
   - Bash commands in fenced code blocks
   - Explicit "Wait for this to complete before proceeding" gates where commands are long-running
   - Decision points where the agent should ask the user (e.g., which template to use)
4. **Results collection** — how to read output files, what metrics to extract, table format for presenting results
5. **Troubleshooting** — bullet list matching the human guide's troubleshooting section

### Key conventions

- Commands use `simlab` CLI directly (not `uv run simlab`)
- When a command depends on user input, show the template with `<placeholder>` syntax
- Always include a tear-down step (`simlab env down`)
- Results tables use pipe-delimited markdown format

## Supporting Files

Supporting files are optional and recipe-specific. Include them when the recipe needs:

- **Dataset converter** — Python script transforming an external dataset into SimLab task format
- **Config template** — `.toml` or `.yaml` pre-filled for the recipe's use case
- **Custom verifier** — Python module implementing recipe-specific verification

Conventions:
- Scripts live in the recipe folder alongside the guides
- Python scripts should be runnable from `cli/simlab/` as `python cookbook/<recipe>/script.py`
- Config files use clear, descriptive names

## Index Entry

When adding a recipe, append one row to the Recipes table in `cli/simlab/cookbook/cookbook-intro.md`:

```
| [recipe-name](recipe-name/) | One-line description. |
```
