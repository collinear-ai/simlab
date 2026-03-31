# SFT with Tinker

Collect expert agent trajectories from SimLab and fine-tune a smaller model to imitate them using supervised fine-tuning (SFT) on [Tinker](https://tinker-docs.thinkingmachines.ai/). The expert model generates demonstrations across SimLab tasks, successful trajectories are converted to Tinker's conversation format, and the student model learns to reproduce the expert's tool-calling behavior.

## Quick start

Run the entire pipeline end-to-end with a single command:

```bash
./cookbook/tinker-sft/run.sh
```

This uses defaults (ERP template, gpt-4o-mini expert, Qwen3-4B student). Override with flags:

```bash
./cookbook/tinker-sft/run.sh \
  --template crm_sales \
  --expert-model gpt-5.2 \
  --student-model Qwen/Qwen3-8B \
  --renderer qwen3 \
  --num-tasks 10 \
  --rollout-count 8
```

The script creates a timestamped working directory with all artifacts. See below for what each step does.

## Prerequisites

- **SimLab** installed with Daytona support:
  ```bash
  uv add "simlab[daytona] @ git+https://github.com/collinear-ai/simlab.git"
  ```
- **tinker-cookbook** installed:
  ```bash
  uv pip install tinker-cookbook
  ```
- **API keys** exported (or set in repo root `.env`):
  ```bash
  export SIMLAB_COLLINEAR_API_KEY="col_..."   # from platform.collinear.ai
  export DAYTONA_API_KEY="dtn_..."            # from app.daytona.io
  export OPENAI_API_KEY="sk-..."              # or your expert model provider's key
  export TINKER_API_KEY="tml-..."             # from Tinker dashboard
  ```
- **Verifier** configured (to score trajectories):
  ```bash
  export SIMLAB_VERIFIER_MODEL="gpt-4o-mini"
  export SIMLAB_VERIFIER_PROVIDER="openai"
  export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY"
  ```

## Step 1: Create an environment and generate tasks

Pick a pre-built template or build a custom environment. Use `--non-interactive` to skip interactive prompts (required in scripted/agent workflows):

```bash
simlab templates list
simlab env init my-env --template <template> --non-interactive
simlab env up my-env --daytona
```

**Generate tasks** if your template doesn't include them:

```bash
simlab tasks-gen init --template <template> --output-dir ./taskgen
```

Edit `taskgen/config.toml` to set `num_tasks` and adjust complexity. If the quality filter is too aggressive and yields zero tasks, set `filter = false` under `[generation]`:

```toml
[generation]
num_tasks = 10
complexity = { easy = 0.3, medium = 0.5, hard = 0.2 }
filter = false
```

Or use the helper script to patch it automatically:

```bash
python scripts/patch_taskgen_config.py taskgen/config.toml 10
```

Then generate and list:

```bash
simlab tasks-gen run --config ./taskgen/config.toml
simlab tasks list --tasks-dir ./generated-tasks
```

## Step 2: Collect expert trajectories

Run a strong expert model across your tasks with multiple rollouts. The `--task` flag accepts one task ID per invocation — run the command once per task:

```bash
simlab tasks run \
  --env my-env \
  --task <task_id> \
  --tasks-dir ./generated-tasks \
  --daytona \
  --rollout-count 10 \
  --max-parallel 3 \
  --agent-model gpt-4o-mini \
  --agent-provider openai \
  --agent-api-key "$OPENAI_API_KEY"
```

Output is written to:

```
output/parallel_run_<task_id>_<timestamp>/
  rollout_0/
    artifacts.json          # Full trajectory (messages, tool_calls, tool_results)
    verifier/reward.json    # {"reward": 1.0} or {"reward": 0.0}
  rollout_1/
    ...
  summary.json
```

> **Note:** Some rollouts may fail due to Daytona sandbox timeouts. This is normal — the pipeline only needs successful rollouts.

## Step 3: Convert to Tinker format

Use [`scripts/export_sft.py`](scripts/export_sft.py) to convert SimLab artifacts to Tinker's JSONL format. The script reconstructs full OpenAI-format messages — recovering complete tool observations from the `tool_results` array instead of using the compact summaries stored in `messages`.

```bash
# Filter to successful rollouts only (default):
python scripts/export_sft.py output/ training_data.jsonl

# Include all rollouts (no reward filtering):
python scripts/export_sft.py output/ training_data.jsonl 0.0
```

> **Tip:** Inspect the output to verify tool calls look correct:
> ```bash
> head -1 training_data.jsonl | python -m json.tool
> ```

## Step 4: Configure and run SFT

Use [`scripts/train_sft.py`](scripts/train_sft.py) to run Tinker SFT. The script uses tinker-cookbook's low-level API — it deserializes tool calls into `ToolCall` pydantic objects (required by Tinker's renderers) and runs LoRA fine-tuning through the Tinker service.

```bash
python scripts/train_sft.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --renderer qwen3_instruct \
  --data training_data.jsonl
```

Common model/renderer pairings:
- `Qwen/Qwen3-4B-Instruct-2507` with `qwen3_instruct`
- `Qwen/Qwen3-8B` with `qwen3`
- `meta-llama/Llama-3.1-8B-Instruct` with `llama3`

Additional flags: `--lr`, `--max-length`, `--lora-rank`, `--epochs`. Run `python scripts/train_sft.py --help` for details.

**Key design notes:**

- **`TrainOnWhat.ALL_ASSISTANT_MESSAGES`** — The model learns all assistant turns including tool calls. Prompt/user/tool-response tokens get zero loss weight.
- **`ToolCall.model_validate(tc)`** — Required because Tinker's renderers expect pydantic objects, not plain dicts.
- **Do not use `FromConversationFileBuilder`** — HuggingFace `datasets` fills `None` for missing `tool_calls` keys, which breaks the renderer. The script uses the low-level API instead.

## Step 5: Evaluate the fine-tuned model

Evaluation requires wrapping Tinker's sampling client in a SimLab `BaseAgent` so it can be run via `simlab tasks run --agent-import-path`. This depends on Tinker's sampling/inference API, which is beyond the scope of this cookbook. See the [Tinker docs on sampling](https://tinker-docs.thinkingmachines.ai/training-sampling) for how to obtain a sampling client from a trained checkpoint.

## Step 6: Tear down

```bash
simlab env down my-env --daytona
```

## Troubleshooting

- **`simlab: command not found`** — Install with `uv add "simlab[daytona] @ git+https://github.com/collinear-ai/simlab.git"`.
- **`ModuleNotFoundError: tinker_cookbook`** — Install with `uv pip install tinker-cookbook`.
- **`TINKER_API_KEY` not set** — Check repo root `.env` or export manually.
- **Interactive prompt errors (`OSError: Invalid argument`)** — Use `--non-interactive` with `simlab env init`.
- **Task generation yields zero tasks** — Set `filter = false` under `[generation]` in `config.toml`, or use `scripts/patch_taskgen_config.py`.
- **`--task` rejects multiple IDs** — `--task` accepts one task ID. Run the command once per task.
- **Zero trajectories after conversion** — No rollouts passed verification. Pass `0.0` as the third argument to `export_sft.py`.
- **Daytona sandbox timeouts** — Transient. Re-run or increase `--rollout-count`.
- **`'dict' object has no attribute 'function'`** — Tool calls must be deserialized via `ToolCall.model_validate(tc)`. Use `scripts/train_sft.py` which handles this.
- **`NoneType is not iterable` on `tool_calls`** — Do not use `FromConversationFileBuilder`. Use `scripts/train_sft.py` instead.

## Next steps

- **Increase data quality** — Run more rollouts, use a stronger expert, or filter by step efficiency (prefer shorter successful trajectories).
- **Scale up** — Fine-tune larger student models (e.g., Qwen3-32B) for better performance.
- **RL fine-tuning** — Use the SFT model as a starting point for on-policy RL, where Tinker samples trajectories and SimLab verifiers provide the reward signal.
