# Autonomous Prompt Optimization with SimLab

This cookbook applies Andrej Karpathy's [auto-research](https://github.com/karpathy/autoresearch) pattern to agent prompt engineering. An outer AI agent (Claude Code, Cursor, Codex, etc.) autonomously iterates on a system prompt to maximize an AI agent's performance on SimLab tasks.

## The Idea

In auto-research, an outer agent modifies `train.py` to improve neural network training, measuring progress via validation loss (BPB). Here, the same loop applies to agent development:

| auto-research | simlab-auto-research |
|---|---|
| `train.py` (mutable) | `system-prompt.md` (mutable) |
| `prepare.py` (fixed) | SimLab environment + tasks (fixed) |
| `program.md` | `SKILL.md` |
| `val_bpb` (lower = better) | `avg_reward` 0.0-1.0 (higher = better) |
| `results.tsv` | `results.tsv` |

The outer agent modifies the system prompt, runs the agent against SimLab tasks, measures the average reward, and keeps or reverts the change. No GPU or model training is involved — just prompt optimization.

## What You Need

- **SimLab** installed (`pip install simulationlab` or `uv add simulationlab`)
- **API keys**: `SIMLAB_COLLINEAR_API_KEY`, `DAYTONA_API_KEY` (if using Daytona), and an agent API key (`OPENAI_API_KEY` or your provider's key)
- **Verifier config**: `SIMLAB_VERIFIER_MODEL`, `SIMLAB_VERIFIER_PROVIDER`, `SIMLAB_VERIFIER_API_KEY`
- **A coding agent** (Claude Code, Cursor, Codex, etc.) to serve as the outer "researcher" agent

## Quick Start

### 1. Install

From `cli/simlab/cookbook/simlab-auto-research`:

```bash
uv sync
```

### 2. Clone auto-research for reference

```bash
git clone https://github.com/karpathy/autoresearch ./autoresearch-reference
```

Read `autoresearch-reference/program.md` to see how Karpathy structures the autonomous experiment loop. Our `SKILL.md` adapts this same pattern for agent prompt optimization.

### 3. Set environment variables

```bash
export SIMLAB_COLLINEAR_API_KEY="col_..."
export DAYTONA_API_KEY="dtn_..."
export OPENAI_API_KEY="sk-..."
export SIMLAB_AGENT_API_KEY="$OPENAI_API_KEY"
export SIMLAB_VERIFIER_MODEL="gpt-4o"
export SIMLAB_VERIFIER_PROVIDER="openai"
export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY"
```

### 4. Choose a scenario and initialize

Pick any SimLab template:

```bash
# See available templates
simlab templates list

# Initialize with your chosen template
simlab env init auto-research-env --template coding
```

Or bring your own custom environment with MCP servers:

```bash
simlab env init auto-research-env --mcp-servers ./mcp-servers.json
```

### 5. Run the baseline

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

This runs all tasks against the initial `system-prompt.md` and prints the average reward. The environment starts automatically.

### 6. Start the autonomous loop

Point your outer agent at the `SKILL.md`:

> Read `cookbook/simlab-auto-research/SKILL.md` and follow the workflow.

The outer agent will:

1. Analyze which tasks fail and why (reading `artifacts.json`)
2. Hypothesize a system prompt improvement
3. Edit `system-prompt.md`
4. Run `./run_experiment.sh`
5. Keep or revert based on the reward
6. Repeat forever

### 7. Monitor progress

Watch `results.tsv` for the reward trajectory:

```bash
cat results.tsv
```

Check the git log for the full history of prompt changes:

```bash
git log --oneline
```

### 8. Tear down when done

```bash
simlab env down auto-research-env
```

## How It Works

### The ConfigurableAgent

The cookbook provides a `ConfigurableAgent` that implements SimLab's `BaseAgent` contract. It behaves identically to SimLab's built-in reference agent except for one thing: its system prompt is loaded from `system-prompt.md` at the start of each run instead of being hardcoded.

When `simlab tasks run` is invoked with `--agent-import-path`, it loads this agent:

```bash
simlab tasks run \
  --env auto-research-env \
  --task <task_id> \
  --agent-import-path simlab_auto_research.configurable_agent:ConfigurableAgent
```

### The Experiment Script

`run_experiment.sh` automates running all tasks and computing the average reward. It outputs a single metric (`avg_reward`) that the outer agent uses to decide whether to keep or revert a prompt change.

### The Prompt File

`system-prompt.md` is the only file the outer agent modifies. It contains the system prompt injected into every LLM call the agent makes. Small changes here can dramatically affect task completion rates.

### results.tsv

A tab-separated log of all experiments, following the auto-research convention:

```
commit	avg_reward	tasks_run	status	description
a1b2c3d	0.3333	3	baseline	Initial system prompt
b2c3d4e	0.5000	3	keep	Added workspace exploration step
c3d4e5f	0.3333	3	revert	Removed tool usage section (hurt performance)
d4e5f6g	0.6667	3	keep	Added verify-before-finishing instruction
```

## Customization

- **Any scenario**: Use any SimLab template (`coding`, `hr`, `customer-service`, `finance`), not just coding
- **Custom environments**: Use `--mcp-servers` for custom tool configurations and set `TASKS_DIR` for local task bundles
- **Different model**: Set `SIMLAB_AGENT_API_KEY` and configure the model via SimLab's agent flags
- **Specific tasks**: Pass task IDs to the experiment script: `./run_experiment.sh hello_world cli_task_manager`
- **System prompt location**: Set `SYSTEM_PROMPT_PATH` to use a different file
- **Max steps**: Set `MAX_STEPS` to allow more agent steps per task
