# SFT with Tinker

Fine-tune a model on expert agent trajectories collected from SimLab using Tinker's supervised learning pipeline.

## Quick start

For a fully automated run, use the included script:

```bash
./cookbook/tinker-sft/run.sh
```

Override defaults with flags:

```bash
./cookbook/tinker-sft/run.sh \
  --num-train-tasks 256 --num-eval-tasks 32 --rollout-count 2 \
  --expert-model gpt-5.4-mini \
  --student-model Qwen/Qwen3.5-4B --renderer qwen3 \
  --batch-size 32 --grad-accum 4
```

The script is **resumable** — re-running skips tasks that already have artifacts. See the step-by-step workflow below for details on each stage.

> **Status (2026-04-08): pipeline is mid-migration to Qwen3.5-4B.** The
> previous run trained Qwen3-4B-Instruct-2507 (32k ctx) and discovered that
> 32% of HR rollouts exceed 32k tokens, so most training sequences were
> silently truncated and eval rollouts hit the context window after ~7
> tool calls. The new plan is Qwen3.5-4B (64k ctx) with rollouts filtered
> to ≤64k tokens. See the open TODOs at the top of `run.sh` — until they
> are implemented the pipeline will not run end-to-end on Qwen3.5-4B.

## Prerequisites

Before starting, confirm these are in place:

1. SimLab is installed with `daytona` AND `npc` extras: `simlab --version` (npc is required for rollouts; install via `uv tool install --python 3.13 'simulationlab[daytona,npc] @ git+https://github.com/collinear-ai/simlab.git'`)
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
- What **expert model** and provider should generate the demonstrations? (e.g., `gpt-5.4-mini` via `openai`)
- What **student model** do you want to fine-tune? Default is `Qwen/Qwen3.5-4B` (4B params, 64k context, hybrid reasoning). Tinker's other sub-9B options (Llama-3.x 1B/3B/8B, Qwen3-8B, Qwen3-4B-Instruct-2507) all cap at 32k context.
- Do you have an existing SimLab environment with tasks, or do you need to create one?
- How many train tasks? How many eval tasks? (API caps at 200 per generation request — batch for more)
- How many rollouts per task? (recommend 2; use `--rollout-count 2` or higher — count of 1 reuses a shared sandbox which may be stale)

### 2. Create environment

Always use `--non-interactive` to avoid interactive prompt errors:

```bash
simlab env init <env_name> --template <template> --non-interactive
simlab env up <env_name> --daytona
```

After `env up`, immediately disable Daytona auto-stop (default is 15 min idle, which is much shorter than a typical pipeline run):

```bash
SANDBOX_ID=$(python3 -c "import json; print(json.load(open('environments/<env_name>/daytona-state.json'))['sandbox_id'])")
/Users/me/.local/share/uv/tools/simulationlab/bin/python -c "
from daytona import Daytona
Daytona().get('$SANDBOX_ID').set_autostop_interval(0)
"
```

Wait for this to complete before proceeding.

### 3. Fetch tool schemas

Verifier generation needs real tool schemas to produce meaningful programmatic verifiers. Fetch schemas from the running environment:

```bash
python cookbook/tinker-sft/scripts/fetch_tool_schema.py \
  --env <env_name> --output tool_schema.json --env-dir environments/<env_name>
```

Pass `--tools tool_schema.json` to `simlab tasks-gen run` to include schemas in verifier generation.

### 4. Generate tasks

Generate separate **train** and **eval** task sets. The task-gen API caps at **200 tasks per request**. For larger counts, generate in batches:

```bash
simlab tasks-gen init --template <template> --output-dir ./taskgen
python cookbook/tinker-sft/scripts/patch_taskgen_config.py taskgen/config.toml 200 --model claude-sonnet-4-6
simlab tasks-gen run --config ./taskgen/config.toml
```

`patch_taskgen_config.py` sets `filter = false` (the quality filter is often too strict) and optionally overrides the model.

For 200+ tasks, repeat with separate output dirs and merge the task JSON files into one `tasks/` directory. Filenames must match the internal `meta.task_id` field.

After generation, check verifier health:

```bash
find ./gen-batch1/verifiers -maxdepth 1 -name '*.FAILED.py' | wc -l
find ./gen-batch1/verifiers -maxdepth 1 -name '*.py' ! -name '__init__.py' ! -name 'common.py' | wc -l
```

### 5. Collect expert trajectories

`--task` accepts **one task ID per invocation**. Loop over tasks:

```bash
for task_id in $(ls generated-tasks/tasks/*.json | xargs -I{} basename {} .json); do
  # Skip if already collected
  existing=$(find output -path "*${task_id}*" -name artifacts.json 2>/dev/null | wc -l)
  [[ "$existing" -ge 2 ]] && continue

  simlab tasks run \
    --env <env_name> \
    --task "$task_id" \
    --tasks-dir ./generated-tasks \
    --daytona \
    --rollout-count 2 \
    --max-parallel 5 \
    --agent-model openai/<expert_model> \
    --agent-provider <provider> \
    --agent-api-key "$AGENT_API_KEY" \
    || echo "Warning: some rollouts failed for $task_id"

  # Prevent stale sandbox state from cascading to next task
  find output -name "daytona-state.json" -not -path "*/environments/*" -exec rm -f {} \;
done
```

Each task takes ~50 seconds. Some rollouts may fail due to Daytona sandbox timeouts — this is normal.

### 6. Convert artifacts to Tinker JSONL

Use [`scripts/export_sft.py`](scripts/export_sft.py):

```bash
# Filter to successful rollouts only:
python cookbook/tinker-sft/scripts/export_sft.py output/ training_data.jsonl

# Include all rollouts (no reward filtering):
python cookbook/tinker-sft/scripts/export_sft.py output/ training_data.jsonl 0.0
```

### 7. Run SFT

Use [`scripts/train_sft.py`](scripts/train_sft.py):

```bash
python cookbook/tinker-sft/scripts/train_sft.py \
  --model Qwen/Qwen3.5-4B \
  --renderer qwen3 \
  --data training_data.jsonl \
  --max-length 65536 \
  --save-name my-checkpoint
```

Tinker per-model context windows for sub-9B models (probed empirically):
- `Qwen/Qwen3.5-4B` — **64k** (only sub-9B option above 32k; hybrid reasoning)
- everything else (`Llama-3.2-1B/3B`, `Llama-3.1-8B(-Instruct)`, `Qwen3-4B-Instruct-2507`, `Qwen3-8B`, `Qwen3-8B-Base`) — **32k**

`--max-length` should track the model's context window, not Tinker's previous default of 16384, which silently truncated >50% of HR rollouts.

### 8. Evaluate the fine-tuned model

Use the eval script which wraps Tinker's sampling client in a SimLab `BaseAgent`:

```bash
# Evaluate SFT checkpoint:
bash cookbook/tinker-sft/scripts/run_eval.sh \
  --tasks-dir ./eval-tasks \
  --env <env_name> \
  --checkpoint "tinker://my-checkpoint" \
  --label sft

# Evaluate base model for comparison:
bash cookbook/tinker-sft/scripts/run_eval.sh \
  --tasks-dir ./eval-tasks \
  --env <env_name> \
  --label base
```

The eval uses `scripts/eval_agent.py` (`TinkerEvalAgent`), which handles the sampling -> tool execution -> re-prompt loop. It works by:
1. Building tool specs from the SimLab environment
2. Using `renderer.build_generation_prompt()` and `renderer.parse_response()` for the model
3. Dispatching tool calls to the environment
4. Collecting rubric scores from SimLab's built-in verifier

The `simlab tasks run --agent-import-path eval_agent:TinkerEvalAgent` invocation loads the agent class directly. PYTHONPATH must include the scripts directory.

### 9. Tear down

```bash
simlab env down <env_name> --daytona
```

## Troubleshooting

- **Interactive prompt errors (`OSError: Invalid argument`)** — Always use `--non-interactive` with `simlab env init`.
- **Task generation yields zero tasks** — Use `scripts/patch_taskgen_config.py` to set `filter = false`.
- **Task gen `num_tasks` exceeds 200** — API caps at 200. Generate in batches and merge.
- **`--task` rejects multiple IDs** — `--task` accepts one task ID per invocation.
- **`--rollout-count 1` uses stale sandbox** — Use `--rollout-count 2+` for fresh per-rollout sandboxes.
- **Daytona errors cascade across tasks** — Clean stale state: `find output -name "daytona-state.json" -not -path "*/environments/*" -exec rm -f {} \;`
- **Zero trajectories** — Use `min_reward=0.0` (third arg to `export_sft.py`).
- **`'dict' object has no attribute 'function'`** — Use `scripts/train_sft.py` which handles `ToolCall` deserialization.
- **`NoneType is not iterable` on `tool_calls`** — Do not use `FromConversationFileBuilder`. Use `scripts/train_sft.py`.
- **Verifiers are gibberish / all .FAILED.py** — The verifier LLM had no tool schemas. Fetch schemas with `fetch_tool_schema.py` and pass `--tools tool_schema.json` to `simlab tasks-gen run`.
- **`Prompt length plus max_tokens exceeds the model's context window`** — Real overflow, not a teardown error. The trajectory was longer than the model can hold. With Qwen3.5-4B (64k ctx) this should be rare after `MAX_TRAIN_TOKENS=64000` filtering; if it still happens at eval, the agent should fail the task gracefully (see TODO in `run.sh`).
- **Most assistant messages have zero loss / model never emits tool calls** — Tool-call format mismatch between `export_sft.py` and the model's pretraining. Qwen3.5 expects XML `<function=>/<parameter=>` blocks; Qwen3-4B-Instruct-2507 expects JSON `{"name":...,"arguments":...}`. Pick one and make sure the export script and the eval parser agree.
