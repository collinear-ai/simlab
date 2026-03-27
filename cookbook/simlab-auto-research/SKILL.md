# Autonomous Prompt Optimization for SimLab

You are running an autonomous experiment loop that optimizes a system prompt
for an AI agent evaluated on SimLab tasks. You will modify `system-prompt.md`,
run the agent against tasks, measure the average reward, and iterate.

This follows the [auto-research](https://github.com/karpathy/autoresearch)
pattern: instead of modifying `train.py` to lower `val_bpb`, you modify a
system prompt to raise `avg_reward`.

## Prerequisites

Before starting, confirm these are in place:

1. SimLab is installed: `simlab --version`
2. `SIMLAB_COLLINEAR_API_KEY` is set (from platform.collinear.ai)
3. `DAYTONA_API_KEY` is set (from app.daytona.io) — if using Daytona
4. An agent API key is set: `SIMLAB_AGENT_API_KEY` or `OPENAI_API_KEY`
5. Verifier is configured: `SIMLAB_VERIFIER_MODEL`, `SIMLAB_VERIFIER_PROVIDER`,
   `SIMLAB_VERIFIER_API_KEY`
6. The cookbook package is installed: `uv sync` from this directory
7. `system-prompt.md` exists in the working directory

If any prerequisite is missing, tell the user which env vars or steps are needed
and wait before proceeding.

## Setup (Run Once)

### 1. Clone auto-research for reference

```bash
git clone https://github.com/karpathy/autoresearch ./autoresearch-reference
```

Read `autoresearch-reference/program.md` to understand the original pattern
you are adapting. Note the structure: fixed evaluation, mutable training code,
experiment loop with keep/revert, results logged to TSV.

### 2. Create a git branch for this session

```bash
git checkout -b auto-research/session-$(date +%Y%m%d-%H%M%S)
```

### 3. Choose a scenario and initialize the environment

Ask the user which SimLab template to use. If they are unsure, show available
templates:

```bash
simlab templates list
```

**Option A — From a template:**

```bash
simlab env init auto-research-env --template <template>
```

**Option B — Custom environment:**

```bash
# With custom MCP servers
simlab env init auto-research-env --mcp-servers ./mcp-servers.json

# Then generate tasks for the custom environment
simlab tasks-gen init --env auto-research-env
simlab tasks-gen run --env auto-research-env
```

Custom environments require a local task bundle. All subsequent commands must
include `--tasks-dir <path>`. Set `TASKS_DIR=<path>` so `run_experiment.sh`
picks it up automatically.

### 4. List available tasks

```bash
# Template-based:
simlab tasks list --env auto-research-env

# Custom environment with local task bundle:
simlab tasks list --tasks-dir <path>
```

Note all task IDs. These are the tasks you will run every experiment.

### 5. Run the baseline experiment

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

This runs all tasks against the current `system-prompt.md` and prints
`avg_reward=<float>`.

### 6. Initialize results.tsv and commit

Create `results.tsv` with the baseline result:

```
commit	avg_reward	tasks_run	status	description
<short_hash>	<avg_reward>	<task_count>	baseline	Initial system prompt
```

Commit and record:

```bash
git add system-prompt.md
git commit -m "baseline: avg_reward=<avg_reward>"
```

The short commit hash goes in the first column of results.tsv.

## Experiment Loop (Repeat Forever)

LOOP FOREVER:

### 1. Analyze recent results

Read `results.tsv` and the latest run outputs under `output/`. Identify:

- Which tasks consistently fail (reward=0.0)?
- Which tasks pass? What patterns do they share?
- What errors appear in failed runs?

If a task failed, read `output/agent_run_<task_id>_*/artifacts.json` for the
most recent run to understand the failure mode (e.g., wrong tool usage, ran out
of steps, misunderstood the instruction).

### 2. Form a hypothesis

Based on the analysis, form a specific hypothesis about what to change in the
system prompt. Examples:

- "The agent does not explore the workspace before acting. Add an explicit
  exploration step."
- "The agent forgets to verify its work. Add a mandatory verification step."
- "The agent writes partial files. Add instruction to always write complete
  files."
- "The agent gets stuck in tool-call loops. Add a maximum retry instruction."
- "The prompt is too long and dilutes focus. Remove the least impactful
  section."

### 3. Modify `system-prompt.md`

Edit the system prompt based on your hypothesis. Make **one focused change per
experiment**. Do not change multiple things at once — you need clear signal on
what works.

### 4. Commit the change

```bash
git add system-prompt.md
git commit -m "experiment: <short description of change>"
```

### 5. Run the experiment

```bash
./run_experiment.sh
```

Read the `avg_reward` from the output.

### 6. Evaluate

Compare the new `avg_reward` to the previous best.

**If improved (higher avg_reward):**

Append to `results.tsv`:

```
<short_hash>	<avg_reward>	<task_count>	keep	<description of change>
```

This commit advances the branch.

**If equal or worse:**

Revert the system prompt to the previous version:

```bash
git revert HEAD --no-edit
```

Append to `results.tsv` with status `revert`:

```
<short_hash>	<avg_reward>	<task_count>	revert	<description of change>
```

### 7. Go to step 1

**NEVER STOP.** Once the experiment loop has begun, do NOT pause to ask if you
should continue. The user might be away and expects you to continue working
indefinitely until manually stopped.

If you run out of ideas, think harder:
- Re-read artifacts from failed tasks for new angles
- Try combining near-miss ideas from previous experiments
- Try radically different prompt structures
- Try shorter prompts (remove sections and see if reward holds)
- Study the tool schemas in the artifacts and add tool-specific guidance

## Simplicity Criterion

All else being equal, simpler is better. A small improvement that adds ugly
complexity to the prompt is not worth it. Conversely, removing sections and
getting equal or better results is a great outcome — that is a simplification
win.

When evaluating whether to keep a change, weigh the complexity cost against
the improvement magnitude. A 0.01 avg_reward improvement that adds 20 lines
of convoluted instructions? Probably not worth it. A 0.01 improvement from
deleting a section? Definitely keep.

## Strategy Tips

- Start with broad improvements (task decomposition, workspace exploration)
  before fine-tuning specific task types.
- If the agent fails because it uses tools incorrectly, study the tool names
  and schemas in the artifacts and add usage guidance to the prompt.
- If the agent runs out of steps, add instructions for efficiency (plan first,
  then execute — minimize back-and-forth).
- If reward has plateaued after 5+ experiments, try a fundamentally different
  prompt structure rather than incremental edits.
- Track what works and what does not in a mental model. The results.tsv is your
  experiment log — use it.

## Troubleshooting

- **`simlab: command not found`** — Install with
  `uv add "simlab[daytona] @ git+https://github.com/collinear-ai/simlab.git"`.
- **`SIMLAB_COLLINEAR_API_KEY` not set** — Get a key from platform.collinear.ai
  (Developers > API Keys).
- **`DAYTONA_API_KEY` not set** — Get a key from app.daytona.io.
- **`FileNotFoundError: system-prompt.md`** — Run from the cookbook directory
  or set `SYSTEM_PROMPT_PATH` to the full path.
- **All tasks return reward=0.0** — Check that verifier keys are set, the agent
  model is valid, and the environment can start.
- **Agent times out** — Increase `MAX_STEPS` or add efficiency instructions
  to the prompt.
- **`run_experiment.sh` cannot find tasks** — Ensure the environment is
  initialized. For custom environments, set `TASKS_DIR`.
