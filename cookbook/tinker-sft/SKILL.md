# SFT with Tinker

Fine-tune a model on expert agent trajectories collected from SimLab using Tinker's supervised learning pipeline.

## Quick start

For a fully automated run, use the included script:

```bash
./cookbook/tinker-sft/run.sh
```

Override defaults with flags (e.g., `--template crm_sales --num-tasks 10 --rollout-count 8 --student-model Qwen/Qwen3-8B --renderer qwen3`). The script handles env setup, task generation, rollouts, conversion, and training. See the step-by-step workflow below for details on each stage.

## Prerequisites

Before starting, confirm these are in place:

1. SimLab is installed with Daytona support: `simlab --version`
2. `SIMLAB_COLLINEAR_API_KEY` is set (from platform.collinear.ai)
3. `DAYTONA_API_KEY` is set (from app.daytona.io)
4. An expert model API key is set (e.g., `OPENAI_API_KEY`)
5. `TINKER_API_KEY` is set (check repo root `.env` or export manually)
6. `tinker-cookbook` is installed: `python -c "import tinker_cookbook"`
7. Verifier is configured: `SIMLAB_VERIFIER_MODEL`, `SIMLAB_VERIFIER_PROVIDER`, and `SIMLAB_VERIFIER_API_KEY`

If any are missing, tell the user which env vars to export or packages to install and wait before proceeding.

## Workflow

### 1. Gather inputs

Ask the user:
- What **expert model** and provider should generate the demonstrations? (e.g., `gpt-4o-mini` via `openai`)
- What **student model** do you want to fine-tune? (e.g., `Qwen/Qwen3-4B-Instruct-2507`)
- Do you have an existing SimLab environment with tasks, or do you need to create one?
- How many rollouts per task? (recommend 10 for a good dataset)

### 2. Create environment and generate tasks

If no environment exists, create one. Always use `--non-interactive` to avoid interactive prompt errors:

```bash
simlab env init <env_name> --template <template> --non-interactive
simlab env up <env_name> --daytona
```

Wait for this to complete before proceeding.

**Generate tasks** if needed:

```bash
simlab tasks-gen init --template <template> --output-dir ./taskgen
```

Patch the config to set task count and disable the quality filter (which is often too strict and yields zero tasks):

```bash
python cookbook/tinker-sft/scripts/patch_taskgen_config.py taskgen/config.toml <num_tasks>
```

Run generation and list tasks:

```bash
simlab tasks-gen run --config ./taskgen/config.toml
simlab tasks list --tasks-dir ./generated-tasks
```

Note the task IDs.

### 3. Collect expert trajectories

Run the expert model across tasks. The `--task` flag accepts one task ID per invocation — run once per task:

```bash
simlab tasks run \
  --env <env_name> \
  --task <task_id> \
  --tasks-dir ./generated-tasks \
  --daytona \
  --rollout-count <count> \
  --max-parallel 3 \
  --agent-model <expert_model> \
  --agent-provider <provider> \
  --agent-api-key "$AGENT_API_KEY"
```

Repeat for each task ID. Some rollouts may fail due to Daytona sandbox timeouts — this is normal.

### 4. Convert artifacts to Tinker JSONL

Use [`scripts/export_sft.py`](scripts/export_sft.py):

```bash
# Filter to successful rollouts only:
python cookbook/tinker-sft/scripts/export_sft.py output/ training_data.jsonl

# Include all rollouts (no reward filtering):
python cookbook/tinker-sft/scripts/export_sft.py output/ training_data.jsonl 0.0
```

Show the user the trajectory count. If zero, suggest using `min_reward=0.0` or check verifier config.

Verify a sample:

```bash
head -1 training_data.jsonl | python -m json.tool
```

### 5. Run SFT

Use [`scripts/train_sft.py`](scripts/train_sft.py):

```bash
python cookbook/tinker-sft/scripts/train_sft.py \
  --model <student_model> \
  --renderer <renderer> \
  --data training_data.jsonl
```

Common model/renderer pairings:
- `Qwen/Qwen3-4B-Instruct-2507` with `qwen3_instruct`
- `Qwen/Qwen3-8B` with `qwen3`
- `meta-llama/Llama-3.1-8B-Instruct` with `llama3`

Additional flags: `--lr`, `--max-length`, `--lora-rank`, `--epochs`.

### 6. Evaluate the fine-tuned model

Evaluation requires wrapping Tinker's sampling client in a SimLab `BaseAgent`. This depends on Tinker's sampling/inference API — see the [Tinker docs on sampling](https://tinker-docs.thinkingmachines.ai/training-sampling) for how to obtain a sampling client from a trained checkpoint.

### 7. Tear down

```bash
simlab env down <env_name> --daytona
```

## Troubleshooting

- **`simlab: command not found`** — Install with `uv add "simlab[daytona] @ git+https://github.com/collinear-ai/simlab.git"`.
- **`ModuleNotFoundError: tinker_cookbook`** — Install with `uv pip install tinker-cookbook`.
- **`TINKER_API_KEY` not set** — Check repo root `.env` or export manually.
- **Interactive prompt errors (`OSError: Invalid argument`)** — Always use `--non-interactive` with `simlab env init`.
- **Task generation yields zero tasks** — Use `scripts/patch_taskgen_config.py` to set `filter = false`.
- **`--task` rejects multiple IDs** — `--task` accepts one task ID per invocation.
- **Zero trajectories after conversion** — Use `min_reward=0.0` (third arg to `export_sft.py`).
- **Daytona sandbox timeouts** — Transient. Re-run or increase `--rollout-count`.
- **`'dict' object has no attribute 'function'`** — Use `scripts/train_sft.py` which deserializes tool calls correctly.
- **`NoneType is not iterable` on `tool_calls`** — Do not use `FromConversationFileBuilder`. Use `scripts/train_sft.py`.
