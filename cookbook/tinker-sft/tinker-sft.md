# SFT with Tinker

Collect expert agent trajectories from SimLab and fine-tune a smaller model to imitate them using supervised fine-tuning (SFT) on [Tinker](https://tinker-docs.thinkingmachines.ai/). The expert model generates demonstrations across SimLab tasks, successful trajectories are converted to Tinker's conversation format, and the student model learns to reproduce the expert's tool-calling behavior. An eval pass on held-out tasks measures the quality gap between the base and fine-tuned model.

## Quick start

Run the entire pipeline end-to-end with a single command:

```bash
./cookbook/tinker-sft/run.sh
```

This uses defaults (HR template, gpt-5.4-mini expert, **Qwen/Qwen3.5-4B student**, 256 train + 32 eval tasks, 2 rollouts). Override with flags:

```bash
./cookbook/tinker-sft/run.sh \
  --template hr \
  --expert-model gpt-5.4-mini \
  --student-model Qwen/Qwen3.5-4B \
  --renderer qwen3 \
  --num-train-tasks 256 \
  --num-eval-tasks 32 \
  --rollout-count 2 \
  --max-train-tokens 64000 \
  --batch-size 32 \
  --grad-accum 4
```

The script is **resumable** — re-running it skips tasks that already have rollout artifacts. It creates a timestamped working directory with all artifacts. See below for what each step does.

> **Status (2026-04-08): mid-migration to Qwen3.5-4B.** The previous run
> trained Qwen3-4B-Instruct-2507 (32k ctx) and discovered that 32% of HR
> rollouts exceed 32k tokens (median 19k, p90 136k, max 380k). Most
> training sequences were silently truncated and eval rollouts hit the
> context window after ~7 tool calls. The new plan is Qwen3.5-4B (64k
> ctx; the only sub-9B Tinker model with >32k context) with rollouts
> filtered to ≤64k tokens. Implementation TODOs are listed in the banner
> at the top of `run.sh`; until they are done, the pipeline does not
> run end-to-end on Qwen3.5-4B.

## Prerequisites

- **SimLab** installed as a uv tool with `daytona` AND `npc` extras (npc is required for rollouts that spawn NPC chat servers):
  ```bash
  uv tool install --python 3.13 'simulationlab[daytona,npc] @ git+https://github.com/collinear-ai/simlab.git'
  ```
  When developing locally, install from a path: `uv tool install --force --reinstall --python 3.13 'simulationlab[daytona,npc] @ /path/to/simlab'`
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

## Step 1: Create an environment

Use `--non-interactive` to skip interactive prompts (required in scripted/agent workflows):

```bash
simlab templates list
simlab env init my-env --template hr --non-interactive
simlab env up my-env --daytona
```

**Important:** Daytona sandboxes auto-stop after 15 min idle by default. For long pipeline runs, disable auto-stop immediately after `env up`:

```bash
SANDBOX_ID=$(python3 -c "import json; print(json.load(open('environments/my-env/daytona-state.json'))['sandbox_id'])")
/Users/me/.local/share/uv/tools/simulationlab/bin/python -c "
from daytona import Daytona
Daytona().get('$SANDBOX_ID').set_autostop_interval(0)
"
```

If a sandbox stops mid-run, rollouts will fail with `Timed out waiting for endpoints` and `dependency failed to start`. Recover with `simlab env up my-env --daytona` and re-disable auto-stop.

## Step 2: Fetch tool schemas

The verifier LLM needs the real MCP tool schemas from the running environment to generate meaningful programmatic verifiers. Without schemas, verifiers reference non-existent tools and fail validation.

```bash
python cookbook/tinker-sft/scripts/fetch_tool_schema.py \
  --env my-env --output tool_schema.json --env-dir environments/my-env
```

This queries each tool server's `/tools` endpoint in the Daytona sandbox and writes a flat JSON array.

## Step 3: Generate tasks

Generate separate **train** and **eval** task sets. The task-gen API caps at **200 tasks per request**. For larger counts, generate in batches and merge:

```bash
# Single batch (up to 200):
simlab tasks-gen init --template hr --output-dir ./taskgen
python scripts/patch_taskgen_config.py taskgen/config.toml 200 --model claude-sonnet-4-6
simlab tasks-gen run --config ./taskgen/config.toml

# For 1000+ tasks: repeat with separate output dirs, then merge task JSON files
# into a single tasks/ directory. Filenames must match internal meta.task_id.
```

The quality filter often rejects generated tasks. It's disabled by `patch_taskgen_config.py` (sets `filter = false` under `[generation]`).

Before collecting rollouts, confirm verifier generation actually succeeded:

```bash
find ./gen-batch1/verifiers -maxdepth 1 -name '*.FAILED.py' | wc -l
find ./gen-batch1/verifiers -maxdepth 1 -name '*.py' ! -name '__init__.py' ! -name 'common.py' | wc -l
```

If most verifiers are `.FAILED.py`, the verifier LLM did not have tool schemas. Re-fetch schemas and re-run task-gen with `--tools`.

## Step 4: Collect expert trajectories

The `--task` flag accepts **one task ID per invocation**. Loop over tasks:

```bash
for task_id in $(ls train-tasks/tasks/*.json | xargs -I{} basename {} .json); do
  simlab tasks run \
    --env my-env \
    --task "$task_id" \
    --tasks-dir ./train-tasks \
    --daytona \
    --rollout-count 2 \
    --max-parallel 5 \
    --agent-model openai/gpt-5.4-mini \
    --agent-provider openai \
    --agent-api-key "$OPENAI_API_KEY" \
    || echo "Warning: some rollouts failed for $task_id"

  # Clean stale sandbox state to prevent cascading cleanup errors
  find output -name "daytona-state.json" -not -path "*/environments/*" -exec rm -f {} \;
done
```

**Important notes:**

- Use `--rollout-count 2` or higher. With `--rollout-count 1`, SimLab reuses the shared environment sandbox (which may be stale); higher counts create fresh per-rollout sandboxes.
- Each task takes ~50 seconds. 256 tasks ~ 3.5 hours.
- **Resumability**: check if artifacts already exist for a task before running it.

## Step 5: Convert to Tinker format

Use [`scripts/export_sft.py`](scripts/export_sft.py) to convert SimLab artifacts to Tinker's JSONL format:

```bash
# Include all rollouts, drop any > 64k tokens (Qwen3.5-4B context cap):
python scripts/export_sft.py output/ training_data.jsonl 0.0 --max-tokens 64000
```

**Format requirements (Qwen3.5-4B):**

- **Tool calls** are XML, not JSON arguments:
  ```
  <tool_call>
  <function=lookup>
  <parameter=name>
  Lisa
  </parameter>
  </function>
  </tool_call>
  ```
  Qwen3.5 was pretrained on this format. The previous JSON-arguments format
  (`{"name": ..., "arguments": ...}`) used by Qwen3-4B-Instruct-2507 is **not** compatible.
- **Tool responses** are wrapped in `<tool_response>...</tool_response>` and consecutive
  tool messages are merged into a single `<|im_start|>user` block by the chat template.
- **Assistant turns** are prefixed with an empty `<think>\n\n</think>\n\n` block to
  match `enable_thinking=false` in the Qwen3.5 chat template (the expert
  trajectories have no real reasoning content).

> **TODO:** the current `export_sft.py` still emits the JSON-arguments format
> and does not filter by token count. Implementing those two changes is the
> blocker before this pipeline runs end-to-end on Qwen3.5-4B.

## Step 6: Configure and run SFT

Use [`scripts/train_sft.py`](scripts/train_sft.py) to run Tinker SFT:

```bash
python scripts/train_sft.py \
  --model Qwen/Qwen3.5-4B \
  --renderer qwen3 \
  --data training_data.jsonl \
  --max-length 65536 \
  --save-name simlab-hr-sft
```

Tinker per-model context windows for sub-9B models (probed empirically):

| Model | Params | Reasoning? | Tinker ctx |
|---|---|---|---|
| meta-llama/Llama-3.2-1B / 3B | 1B / 3B | no | 32,768 |
| meta-llama/Llama-3.1-8B / 8B-Instruct | 8B | no | 32,768 |
| Qwen/Qwen3-4B-Instruct-2507 *(prior default)* | 4B | no | 32,768 |
| Qwen/Qwen3-8B / 8B-Base | 8B | hybrid | 32,768 |
| **Qwen/Qwen3.5-4B** *(current default)* | **4B** | hybrid | **65,536** |

`--max-length` should track the model's context window. The previous
default of 16384 silently truncated >50% of HR trajectories (median ~19k tokens).

**Key design notes:**

- **`TrainOnWhat.ALL_ASSISTANT_MESSAGES`** — The model learns all assistant turns including tool calls. Prompt/user/tool-response tokens get zero loss weight.
- **`ToolCall.model_validate(tc)`** — Required because Tinker's renderers expect pydantic objects, not plain dicts.
- **Do not use `FromConversationFileBuilder`** — HuggingFace `datasets` fills `None` for missing `tool_calls` keys, which breaks the renderer. The script uses the low-level API instead.
- **`--save-name`** — Saves a checkpoint and returns a Tinker sampling client for evaluation.

## Step 7: Evaluate the fine-tuned model

Use [`scripts/run_eval.sh`](scripts/run_eval.sh) to evaluate both the fine-tuned and base models on held-out eval tasks:

```bash
# Evaluate the SFT checkpoint:
bash scripts/run_eval.sh \
  --tasks-dir ./eval-tasks \
  --env my-env \
  --checkpoint "tinker://simlab-hr-sft" \
  --label sft

# Evaluate the base model for comparison:
bash scripts/run_eval.sh \
  --tasks-dir ./eval-tasks \
  --env my-env \
  --label base
```

The eval uses [`scripts/eval_agent.py`](scripts/eval_agent.py) (`TinkerEvalAgent`), which wraps Tinker's sampling client in a SimLab `BaseAgent`. The agent:
1. Builds tool specs from the SimLab environment's `list_tools()`
2. Calls `client.sample()` for the next assistant turn, decodes tokens itself, and parses Qwen3.5 XML tool calls
3. Dispatches tool calls to the environment, collecting observations
4. SimLab's verifier scores the final trajectory

**Failure modes that count as task failure (not crash):**

- **Context-window exceeded** (`Prompt length plus max_tokens exceeds the model's context window`).
  The agent must catch this, mark the rollout `error="context_window_exceeded"`,
  stop the loop, and return current state to the verifier (which will score
  whatever partial work exists, typically 0). Not a crash.

> **TODO:** `eval_agent.py` currently parses the JSON-arguments tool-call
> format and re-raises context-window errors as crashes. Both must change
> before evaluating Qwen3.5-4B.

The eval script outputs per-task rubric scores and aggregates (avg score, min/max, avg steps, tool usage rate) to `eval_results_<label>.json`.

> **TODO:** the aggregator in `scripts/run_eval.sh` reads `rubric_result`
> from `reward.json`, but the actual schema is `{reward, verifier_results}`.
> Fix so `eval_results_<label>.json` contains the real reward distribution.

Override model/renderer:

```bash
bash scripts/run_eval.sh --tasks-dir ./eval-tasks --env my-env \
  --base-model Qwen/Qwen3.5-4B --renderer qwen3
```

See the [Tinker docs on sampling](https://tinker-docs.thinkingmachines.ai/training-sampling) for how to obtain a sampling client from a trained checkpoint.

## Step 8: Tear down

```bash
simlab env down my-env --daytona
```

## Troubleshooting

- **Interactive prompt errors (`OSError: Invalid argument`)** — Use `--non-interactive` with `simlab env init`.
- **Task generation yields zero tasks** — Set `filter = false` under `[generation]` in `config.toml`, or use `scripts/patch_taskgen_config.py`.
- **Task gen `num_tasks` exceeds 200** — API caps at 200 per request. Generate in batches and merge JSON files into one `tasks/` directory.
- **`--task` rejects multiple IDs** — `--task` accepts one task ID per invocation. Loop over tasks.
- **`--rollout-count 1` uses stale sandbox** — Use `--rollout-count 2` or higher to get fresh per-rollout sandboxes.
- **Daytona sandbox cleanup failures cascade** — Remove stale `daytona-state.json` files between tasks: `find output -name "daytona-state.json" -not -path "*/environments/*" -exec rm -f {} \;`
- **Zero trajectories after conversion** — No rollouts passed verification. Pass `0.0` as the third argument to `export_sft.py`.
- **`'dict' object has no attribute 'function'`** — Use `scripts/train_sft.py` which deserializes tool calls correctly.
- **`NoneType is not iterable` on `tool_calls`** — Do not use `FromConversationFileBuilder`. Use `scripts/train_sft.py`.
- **Verifiers are all `.FAILED.py`** — The verifier LLM had no tool schemas. Fetch schemas with `fetch_tool_schema.py` and pass `--tools tool_schema.json` to `simlab tasks-gen run`.
- **`Prompt length plus max_tokens exceeds the model's context window`** — The trajectory grew past the model's context window. Real failure, not a teardown error. Mitigations: (a) drop oversized rollouts at export time (`--max-tokens`), (b) trim large tool-result JSON blobs in the export (huge multiplier; still TODO), (c) pick a model with more context (Qwen3.5-4B is the only sub-9B Tinker model with >32k).
- **Most assistant turns get zero loss / model never emits tool calls at eval** — Tool-call format mismatch between `export_sft.py` and the model's pretraining. Qwen3.5 expects XML `<function=>/<parameter=>` blocks; Qwen3-4B-Instruct-2507 expects JSON `{"name":...,"arguments":...}`. Pick one consistently across export, train, and eval.

## Next steps

- **Implement the open TODOs** at the top of `run.sh` so the Qwen3.5-4B path runs end-to-end.
- **Trim tool-result blobs in `export_sft.py`** — strip redundant fields from each `observation`. Independent multiplier on top of the model swap; could 3–5× the effective context.
- **Increase data quality** — Run more rollouts, use a stronger expert, or filter by step efficiency (prefer shorter successful trajectories).
- **Scale up** — Fine-tune larger student models (e.g., Qwen3-32B) for better performance.
- **RL fine-tuning** — Use the SFT model as a starting point for on-policy RL, where Tinker samples trajectories and SimLab verifiers provide the reward signal.
