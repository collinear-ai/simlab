# Prime-RL Training with SimLab Trajectories

Train agent models with Prime Intellect's prime-rl using SimLab-collected trajectories.

## Prerequisites

Before starting, confirm:

1. SimLab is installed: `simlab --version`
2. prime CLI is installed: `prime --version`
3. `SIMLAB_COLLINEAR_API_KEY` is set
4. `PRIME_API_KEY` is set
5. `OPENAI_API_KEY` is set (for baseline agent)

If any prerequisite is missing, tell the user what to set and **wait before proceeding**.

## Workflow

### 1. Install cookbook dependencies

```bash
cd cookbook/prime-rl-training
uv sync
```

### 2. Create SimLab environment

```bash
simlab templates list
```

Ask the user which template to use (default: `customer_service`).

```bash
simlab env init prime-rl-env --template <template>
```

### 3. Generate tasks

```bash
simlab tasks-gen init --preset customer_support --output-dir ./taskgen
simlab tasks-gen run --config taskgen/config.toml
```

Wait for task generation to complete before proceeding.

### 4. List tasks and select for rollouts

```bash
simlab tasks list --tasks-dir ./generated-tasks
```

Note the task IDs.

### 5. Start environment and collect trajectories

```bash
simlab env up prime-rl-env
```

Wait for all services to become healthy, then run tasks:

```bash
simlab tasks run \
  --env prime-rl-env \
  --task <task_id> \
  --tasks-dir ./generated-tasks \
  --agent-model gpt-5.2 \
  --agent-api-key "$OPENAI_API_KEY"
```

Repeat for each task. Wait for each to complete.

### 6. Convert trajectories to SFT dataset

```bash
python -m prime_rl_training.collect sft \
  --output-dir ./output \
  --save-path ./dataset \
  --min-reward 0.0 \
  --include-failed \
  --format jsonl
```

Verify the dataset:
```bash
wc -l dataset/train.jsonl
head -1 dataset/train.jsonl | python -m json.tool
```

Present the trajectory count and a sample to the user.

### 7. Push verifiers environment to Prime Intellect

```bash
prime env push -p ./prime-envs/simlab_tasks
```

Wait for confirmation. Note the environment ID from the output (e.g., `<username>/simlab-tasks`).

### 8. Check model availability

```bash
prime rl models --plain
```

Present available models. Recommend `Qwen/Qwen3.5-9B` or another available model.

### 9. Configure and launch RL training

Update `configs/rl.toml` with the correct model and environment ID, then:

```bash
prime rl run configs/rl.toml
```

Note the run ID from the output.

### 10. Monitor training

```bash
prime rl logs <run-id> -f
prime rl metrics <run-id> --plain
prime rl progress <run-id> --plain
```

Present metrics to the user.

### 11. Tear down SimLab environment

```bash
simlab env down prime-rl-env
```

## Results collection

After training completes:

```bash
prime rl get <run-id> --plain
prime rl checkpoints <run-id> --plain
```

Present results:

| Metric | Value |
|--------|-------|
| Run ID | ... |
| Model | ... |
| Steps completed | ... |
| Final reward | ... |
| Checkpoint ID | ... |

## Troubleshooting

- **`simlab: command not found`** — Install with `uv pip install simulationlab`
- **`prime: command not found`** — Install with `pip install prime`
- **No trajectories collected** — Ensure the SimLab environment is running (`simlab env up`) and API keys are valid
- **Port conflict on env up** — Edit `docker-compose.yml` to change conflicting port mappings
- **`prime rl models` shows "At Capacity"** — Try a different model or wait
- **Environment push needs username** — The first push prompts for a Prime Intellect username (one-time setup)
