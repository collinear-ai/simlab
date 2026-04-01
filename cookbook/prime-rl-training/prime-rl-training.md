# Prime-RL Training with SimLab Trajectories

Train an agent model using Prime Intellect's prime-rl with trajectories collected from SimLab environments. This cookbook covers the full pipeline: collecting tool-use trajectories, converting to training data, building a verifiers environment, and launching RL training on Prime Intellect's hosted platform.

## What is here

- `prime_rl_training.trajectory_converter`
  - converts SimLab `artifacts.json` files into prime-rl compatible SFT datasets (HuggingFace messages format)
- `prime_rl_training.simlab_env`
  - verifiers-compatible environment wrapper for SimLab task data
- `prime_rl_training.collect`
  - CLI to collect and convert trajectories (`python -m prime_rl_training.collect sft|rl`)
- `prime-envs/simlab_tasks/`
  - standalone verifiers environment package ready to push to Prime Intellect's Environments Hub
- `configs/`
  - prime-rl TOML configs for SFT warmup and hosted RL training
- `run_pipeline.sh`
  - end-to-end automation script

## Prerequisites

- **SimLab** installed:
  ```bash
  uv pip install "simulationlab[daytona]"
  ```
- **prime CLI** installed:
  ```bash
  pip install prime
  ```
- **API keys** exported:
  ```bash
  export SIMLAB_COLLINEAR_API_KEY="col_..."     # from platform.collinear.ai
  export PRIME_API_KEY="pit_..."                 # from Prime Intellect platform
  export OPENAI_API_KEY="sk-..."                 # for running the baseline agent
  export DAYTONA_API_KEY="dtn_..."               # for remote sandbox execution (optional)
  ```
- **Verifier** configured (for scoring SimLab rollouts):
  ```bash
  export SIMLAB_VERIFIER_MODEL="gpt-5.2"
  export SIMLAB_VERIFIER_PROVIDER="openai"
  export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY"
  ```

## Install

Run from `cookbook/prime-rl-training`:

```bash
uv sync
```

This project installs `simulationlab` from the local repo path `../..`.

## Step 1: Create a SimLab environment

Pick an environment template with tool-use tasks:

```bash
simlab templates list
simlab env init prime-rl-env --template customer_service
```

> **Tip:** Use `erp`, `crm_sales`, `customer_service`, or `project_management` for different task patterns. Each template provides different tool servers.

## Step 2: Generate tasks

```bash
simlab tasks-gen init --preset customer_support --output-dir ./taskgen
# Optionally edit taskgen/config.toml (task count, difficulty, model)
simlab tasks-gen run --config taskgen/config.toml
```

Or list template tasks:

```bash
simlab tasks list --env prime-rl-env
```

## Step 3: Collect trajectories with a baseline agent

Run tasks with a capable model to generate training trajectories:

```bash
simlab tasks run \
  --env prime-rl-env \
  --task <task_id_1> <task_id_2> <task_id_3> \
  --tasks-dir ./generated-tasks \
  --agent-model gpt-5.2 \
  --agent-api-key "$OPENAI_API_KEY"
```

For parallel rollouts with Daytona:

```bash
simlab tasks run \
  --env prime-rl-env \
  --task <task_id> \
  --tasks-dir ./generated-tasks \
  --daytona \
  --rollout-count 5 \
  --max-parallel 3 \
  --agent-model gpt-5.2 \
  --agent-api-key "$OPENAI_API_KEY"
```

Each rollout produces:
```
output/agent_run_<task>_<ts>/
  artifacts.json          # Full trajectory (messages, tool calls, results)
  verifier/
    reward.json           # Structured reward (0.0-1.0)
```

> **Note:** Aim for 50-200 successful trajectories for meaningful SFT warmup.

## Step 4: Convert trajectories to SFT dataset

```bash
python -m prime_rl_training.collect sft \
  --output-dir ./output \
  --save-path ./dataset \
  --min-reward 0.5 \
  --format jsonl
```

This produces `dataset/train.jsonl` in prime-rl's messages format — one row per trajectory with full tool-call history. Only trajectories with reward >= 0.5 are included.

Inspect the dataset:
```bash
head -1 dataset/train.jsonl | python -m json.tool
```

**Optional — Push to HuggingFace Hub:**
```bash
python -m prime_rl_training.collect sft \
  --output-dir ./output \
  --push-to myorg/simlab-sft-data
```

## Step 5: SFT warmup (local, requires GPU)

> **Note:** SFT training requires the open-source [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and a GPU. Skip to Step 6 for hosted RL-only training.

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl && uv sync --all-extras

# Run SFT (edit configs/sft.toml to set your dataset path first)
uv run sft @ path/to/cookbook/configs/sft.toml
```

The SFT config trains only on assistant messages (system/user/tool masked out), teaching the model tool-use patterns from successful SimLab trajectories.

## Step 6: Build and push a verifiers environment

The `prime-envs/simlab_tasks/` directory contains a ready-made verifiers environment with embedded SimLab customer support prompts and quality+completeness rubrics.

Push to Prime Intellect's Environments Hub:

```bash
prime env push -p ./prime-envs/simlab_tasks
```

Verify:
```bash
prime env info <your-username>/simlab-tasks
```

> **Customizing:** Edit `prime-envs/simlab_tasks/simlab_tasks.py` to add your own task prompts, adjust rubric weights, or load from a HuggingFace dataset instead of the embedded examples.

## Step 7: Run RL training on Prime Intellect

```bash
# Check available models
prime rl models

# Edit configs/rl.toml:
#   - Set model (e.g., Qwen/Qwen3.5-9B)
#   - Set env ID to your pushed environment (e.g., your-username/simlab-tasks)

prime rl run configs/rl.toml
```

Monitor:
```bash
prime rl logs <run-id> -f        # stream logs
prime rl metrics <run-id>        # training metrics
prime rl rollouts <run-id>       # sample rollouts
prime rl progress <run-id>       # step progress
```

## Step 8: Deploy and evaluate

After training, deploy the LoRA adapter:

```bash
prime rl checkpoints <run-id>
prime deployments create <adapter-id>
```

Evaluate the trained model back through SimLab:

```bash
simlab tasks run \
  --env prime-rl-env \
  --task <task_id> \
  --tasks-dir ./generated-tasks \
  --agent-model <deployed-model-id> \
  --agent-provider openai-compatible \
  --agent-api-key "$PRIME_API_KEY" \
  --agent-base-url "https://api.pinference.ai/api/v1"
```

Compare reward scores against your Step 3 baseline.

## Step 9: Tear down

```bash
simlab env down prime-rl-env
# With Daytona:
simlab env down prime-rl-env --daytona
```

## Next steps

- **More data:** Increase rollout count and add more task templates for broader coverage.
- **Multi-environment RL:** Add multiple `[[env]]` sections to `rl.toml` for diverse training signal.
- **Curriculum learning:** Start with easier templates (e.g., `erp`) and progress to harder ones (e.g., `customer_service`).
- **Custom rubrics:** Modify the verifiers environment rubric to reward specific agent behaviors.

## Troubleshooting

- **`No trajectories found`** — Check that SimLab rollouts completed and `artifacts.json` files exist in the output directory.
- **`prime rl models` shows "At Capacity"** — Try a different model or wait for availability.
- **SFT dataset is too small** — Lower `--min-reward` to include more trajectories, or run more rollouts.
- **RL reward stays at 0** — The rubric may be too strict for the base model. Try starting from an SFT-warmed checkpoint, or adjust rubric weights in the verifiers environment.
- **Port conflicts on `simlab env up`** — Another service is using the port. Edit `docker-compose.yml` port mappings or stop conflicting services.
