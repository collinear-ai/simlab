#!/usr/bin/env bash
#
# End-to-end SimLab + Tinker SFT pipeline.
#
# Creates a SimLab environment, generates train + eval tasks, collects expert
# rollouts, converts trajectories to Tinker format, runs SFT training, and
# evaluates the fine-tuned model against held-out eval tasks.
#
# Resumable: re-running skips tasks that already have rollout artifacts.
# Can be interrupted with Ctrl-C and restarted safely.
#
# Usage:
#   ./run.sh                           # uses defaults (256 train + 32 eval, 2 rollouts)
#   ./run.sh --num-train-tasks 200 --num-eval-tasks 32 --rollout-count 2
#   ./run.sh --student-model Qwen/Qwen3-8B --renderer qwen3
#   ./run.sh --batch-size 32 --grad-accum 4  # training config
#
# Required env vars (set in .env or export manually):
#   SIMLAB_COLLINEAR_API_KEY, DAYTONA_API_KEY, OPENAI_API_KEY, TINKER_API_KEY
#
# ── Open TODOs (this script will not run end-to-end until they are done) ────
#
# 1. scripts/export_sft.py — re-emit trajectories in Qwen3.5 native format:
#    - tool calls: <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>
#    - tool responses: <tool_response>...</tool_response>
#    - assistant turns prefixed with empty <think>\n\n</think>\n\n
#    Add a `--max-tokens N` flag that re-tokenizes each rendered example with
#    the Qwen/Qwen3.5-4B tokenizer and drops any example > N tokens.
#
# 2. scripts/eval_agent.py — match the new format:
#    - parse Qwen3.5 XML tool calls (<function=>/<parameter=>) instead of
#      the inline {"name":...,"arguments":...} JSON form
#    - render assistant prompts with the empty <think> block
#    - catch tinker BadRequestError "Prompt length plus max_tokens exceeds
#      the model's context window" and FAIL the rollout (do not raise);
#      record error="context_window_exceeded" and let the verifier score
#      whatever partial state the agent reached
#
# 3. scripts/run_eval.sh aggregator — currently looks for `rubric_result`
#    in reward.json; the actual schema is {reward, verifier_results: [...]}.
#    Fix so eval_results_<label>.json contains the real reward distribution.
#
# 4. Confirm the correct tinker_cookbook renderer name for Qwen/Qwen3.5-4B
#    (currently RENDERER defaults to "qwen3" — verify before relying on it).
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

TEMPLATE="${TEMPLATE:-hr}"
ENV_NAME="${ENV_NAME:-tinker-sft-env}"
NUM_TRAIN_TASKS="${NUM_TRAIN_TASKS:-256}"
NUM_EVAL_TASKS="${NUM_EVAL_TASKS:-32}"
ROLLOUT_COUNT="${ROLLOUT_COUNT:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
EXPERT_MODEL="${EXPERT_MODEL:-openai/gpt-5.4-mini}"
EXPERT_PROVIDER="${EXPERT_PROVIDER:-openai}"
TASKGEN_MODEL="${TASKGEN_MODEL:-claude-sonnet-4-6}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3.5-4B}"
# TODO: confirm the correct tinker_cookbook renderer name for Qwen3.5-4B
# (Qwen3.5 is a hybrid reasoning model; uses XML <function=>/<parameter=>
# tool-call format and <think> blocks. The qwen3_instruct renderer was for
# Qwen3-4B-Instruct-2507 and is not format-compatible with Qwen3.5.)
RENDERER="${RENDERER:-qwen3}"
# Filter training sequences longer than this many tokens (Qwen3.5-4B ctx = 65536).
# Leave headroom for max_tokens at sample time → 64000 default.
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-64000}"
# Per-example token cap inside Tinker training. Was 16384 (which silently
# truncated >50% of HR rollouts). Bumped to fit Qwen3.5-4B's 64k context.
MAX_LENGTH="${MAX_LENGTH:-65536}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-32}"
MIN_REWARD="${MIN_REWARD:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-2}"
SAVE_NAME="${SAVE_NAME:-simlab-hr-sft}"
WORKDIR="${WORKDIR:-}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-30}"

# ── Parse CLI flags ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case $1 in
    --template)          TEMPLATE="$2";          shift 2 ;;
    --env-name)          ENV_NAME="$2";          shift 2 ;;
    --num-train-tasks)   NUM_TRAIN_TASKS="$2";   shift 2 ;;
    --num-eval-tasks)    NUM_EVAL_TASKS="$2";    shift 2 ;;
    --num-tasks)         NUM_TRAIN_TASKS="$2";   shift 2 ;;  # compat alias
    --rollout-count)     ROLLOUT_COUNT="$2";     shift 2 ;;
    --max-parallel)      MAX_PARALLEL="$2";      shift 2 ;;
    --expert-model)      EXPERT_MODEL="$2";      shift 2 ;;
    --expert-provider)   EXPERT_PROVIDER="$2";   shift 2 ;;
    --taskgen-model)     TASKGEN_MODEL="$2";     shift 2 ;;
    --student-model)     STUDENT_MODEL="$2";     shift 2 ;;
    --renderer)          RENDERER="$2";          shift 2 ;;
    --max-train-tokens)  MAX_TRAIN_TOKENS="$2";  shift 2 ;;
    --max-length)        MAX_LENGTH="$2";        shift 2 ;;
    --learning-rate)     LEARNING_RATE="$2";     shift 2 ;;
    --lora-rank)         LORA_RANK="$2";         shift 2 ;;
    --min-reward)        MIN_REWARD="$2";        shift 2 ;;
    --batch-size)        BATCH_SIZE="$2";        shift 2 ;;
    --grad-accum)        GRAD_ACCUM="$2";        shift 2 ;;
    --epochs)            EPOCHS="$2";            shift 2 ;;
    --save-name)         SAVE_NAME="$2";         shift 2 ;;
    --eval-max-steps)    EVAL_MAX_STEPS="$2";    shift 2 ;;
    --workdir)           WORKDIR="$2";           shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ── Resolve paths ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$SCRIPT_DIR/scripts"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Load .env if present ─────────────────────────────────────────────────────

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi

export SIMLAB_VERIFIER_MODEL="${SIMLAB_VERIFIER_MODEL:-gpt-4o-mini}"
export SIMLAB_VERIFIER_PROVIDER="${SIMLAB_VERIFIER_PROVIDER:-openai}"
export SIMLAB_VERIFIER_API_KEY="${SIMLAB_VERIFIER_API_KEY:-$OPENAI_API_KEY}"

# ── Preflight checks ────────────────────────────────────────────────────────

echo "=== Preflight checks ==="
missing=()
[[ -z "${SIMLAB_COLLINEAR_API_KEY:-}" ]] && missing+=("SIMLAB_COLLINEAR_API_KEY")
[[ -z "${DAYTONA_API_KEY:-}" ]]          && missing+=("DAYTONA_API_KEY")
[[ -z "${OPENAI_API_KEY:-}" ]]           && missing+=("OPENAI_API_KEY")
[[ -z "${TINKER_API_KEY:-}" ]]           && missing+=("TINKER_API_KEY")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: Missing env vars: ${missing[*]}"
  echo "Set them in $REPO_ROOT/.env or export manually."
  exit 1
fi

command -v simlab >/dev/null 2>&1 || { echo "ERROR: simlab not found. Install with: uv tool install --python 3.13 'simulationlab[daytona,npc] @ git+https://github.com/collinear-ai/simlab.git'"; exit 1; }
# Verify npc extra (required for rollouts that spawn NPC chat sessions)
SIMLAB_PY=$(python3 -c "import shutil, subprocess; p = shutil.which('simlab'); print(subprocess.check_output([p, '--python-path'], text=True).strip())" 2>/dev/null) \
  || SIMLAB_PY=$(dirname "$(command -v simlab)")/../bin/python
"$SIMLAB_PY" -c "from simlab.npc_chat.http_server import *" 2>/dev/null || {
  echo "ERROR: simlab[npc] extra not installed. Reinstall with:"
  echo "  uv tool install --force --reinstall --python 3.13 'simulationlab[daytona,npc] @ <path-to-simlab>'"
  exit 1
}
python3 -c "import tinker_cookbook" 2>/dev/null || { echo "ERROR: tinker-cookbook not found. Run: uv pip install tinker-cookbook"; exit 1; }

echo "  Template:         $TEMPLATE"
echo "  TaskGen model:    $TASKGEN_MODEL"
echo "  Expert model:     $EXPERT_MODEL ($EXPERT_PROVIDER)"
echo "  Student model:    $STUDENT_MODEL (renderer: $RENDERER)"
echo "  Train tasks:      $NUM_TRAIN_TASKS"
echo "  Eval tasks:       $NUM_EVAL_TASKS"
echo "  Rollouts/task:    $ROLLOUT_COUNT"
echo "  Batch size:       $BATCH_SIZE (grad_accum: $GRAD_ACCUM)"
echo ""

# ── Working directory ────────────────────────────────────────────────────────

# Default to a stable name so re-running resumes into the same directory.
# Use --workdir or WORKDIR= to override (e.g. for parallel runs).
WORKDIR="${WORKDIR:-$REPO_ROOT/tinker-sft-run}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "=== Working directory: $WORKDIR ==="
echo ""

# ── Step 1: Create environment ───────────────────────────────────────────────

echo "=== Step 1: Creating environment '$ENV_NAME' from template '$TEMPLATE' ==="
simlab env init "$ENV_NAME" --template "$TEMPLATE" --non-interactive
simlab env up "$ENV_NAME" --daytona

# Disable Daytona auto-stop (default 15 min idle) so the env survives long runs.
SANDBOX_ID=$(python3 -c "import json; print(json.load(open('environments/$ENV_NAME/daytona-state.json'))['sandbox_id'])")
"$SIMLAB_PY" -c "
from daytona import Daytona
s = Daytona().get('$SANDBOX_ID')
s.set_autostop_interval(0)
print('  Auto-stop disabled. Sandbox state:', s.state)
"
echo ""

# ── Helper: generate task batch ─────────────────────────────────────────────

generate_tasks() {
  local label="$1"       # "train" or "eval"
  local num_tasks="$2"
  local out_dir="$3"

  echo "=== Generating $num_tasks $label tasks ==="
  mkdir -p "$out_dir/tasks"

  local BATCH_MAX=200
  local remaining=$num_tasks
  local batch_num=0

  while [[ $remaining -gt 0 ]]; do
    local batch_size_gen=$((remaining > BATCH_MAX ? BATCH_MAX : remaining))
    batch_num=$((batch_num + 1))
    local batch_dir="./taskgen-${label}-batch${batch_num}"

    echo "  Batch $batch_num: generating $batch_size_gen tasks..."
    simlab tasks-gen init --template "$TEMPLATE" --output-dir "$batch_dir"
    python3 "$SCRIPTS/patch_taskgen_config.py" "$batch_dir/config.toml" "$batch_size_gen" \
      --model "$TASKGEN_MODEL"

    simlab tasks-gen run --config "$batch_dir/config.toml" --out "./gen-${label}-batch${batch_num}" \
      || echo "  Warning: batch $batch_num had errors (continuing with generated tasks)"

    local batch_task_count=$(find "./gen-${label}-batch${batch_num}/tasks" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    local batch_verifier_count=$(find "./gen-${label}-batch${batch_num}/verifiers" -maxdepth 1 -name '*.py' ! -name '__init__.py' ! -name 'common.py' ! -name '*.FAILED.py' 2>/dev/null | wc -l | tr -d ' ')
    echo "  Batch $batch_num: $batch_task_count tasks, $batch_verifier_count verifiers"
    if [[ "$batch_task_count" -gt 0 && "$batch_verifier_count" -lt "$batch_task_count" ]]; then
      echo "WARNING: batch $batch_num has $batch_task_count task(s) but only $batch_verifier_count usable verifier(s)."
      echo "Some verifiers failed — inspect ./gen-${label}-batch${batch_num}/verifiers for .FAILED.py files."
      echo "Continuing with available verifiers (rollouts for tasks without verifiers will use rubric fallback)."
    fi

    # Merge into out_dir
    if ls "./gen-${label}-batch${batch_num}/tasks/"*.json >/dev/null 2>&1; then
      for f in "./gen-${label}-batch${batch_num}/tasks/"*.json; do
        cp "$f" "$out_dir/tasks/"
      done
    fi
    # Copy verifiers into out_dir for simlab tasks run
    if [[ -d "./gen-${label}-batch${batch_num}/verifiers" ]]; then
      mkdir -p "$out_dir/verifiers"
      cp -n "./gen-${label}-batch${batch_num}/verifiers/"*.py "$out_dir/verifiers/" 2>/dev/null || true
    fi

    remaining=$((remaining - batch_size_gen))
  done

  local task_count=$(ls "$out_dir/tasks/"*.json 2>/dev/null | wc -l | tr -d ' ')
  echo "  Total $label tasks generated: $task_count"

  if [[ "$task_count" -eq 0 ]]; then
    echo "ERROR: No $label tasks generated."
    return 1
  fi
  echo ""
}

# ── Step 2 (tool schema fetch) is intentionally skipped ─────────────────────
#
# When using `simlab tasks-gen run --config`, the hosted API reads tool schemas
# from the config's [[toolset]] definitions automatically. The `--config` and
# `--tools` flags are mutually exclusive — passing `--tools` alongside `--config`
# is an error.  Manual schema fetch via `scripts/fetch_tool_schema.py` is only
# needed if you call `simlab tasks-gen run` without `--config`.

# ── Step 3: Generate train and eval task sets ───────────────────────────────

TRAIN_TASKS_DIR="./train-tasks"
EVAL_TASKS_DIR="./eval-tasks"

generate_tasks "train" "$NUM_TRAIN_TASKS" "$TRAIN_TASKS_DIR"
generate_tasks "eval" "$NUM_EVAL_TASKS" "$EVAL_TASKS_DIR"

TRAIN_COUNT=$(ls "$TRAIN_TASKS_DIR/tasks/"*.json 2>/dev/null | wc -l | tr -d ' ')
EVAL_COUNT=$(ls "$EVAL_TASKS_DIR/tasks/"*.json 2>/dev/null | wc -l | tr -d ' ')
echo "=== Task generation complete: $TRAIN_COUNT train, $EVAL_COUNT eval ==="
echo ""

# ── Step 4: Collect expert rollouts (train set) ────────────────────────────
#
# Runs one task at a time. Rollouts use per-rollout Daytona sandboxes.
# Resumable: skips tasks that already have enough artifacts.

echo "=== Step 4: Collecting expert rollouts ==="

TASK_IDS=$(ls "$TRAIN_TASKS_DIR/tasks/"*.json | xargs -I{} basename {} .json)
echo "  $TRAIN_COUNT task(s), $ROLLOUT_COUNT rollouts each"

count=0
skipped=0
for task_id in $TASK_IDS; do
  count=$((count + 1))

  existing=$(find output -path "*${task_id}*" -name artifacts.json 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$existing" -ge "$ROLLOUT_COUNT" ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  simlab tasks run \
    --env "$ENV_NAME" \
    --task "$task_id" \
    --tasks-dir "$TRAIN_TASKS_DIR" \
    --daytona \
    --rollout-count "$ROLLOUT_COUNT" \
    --max-parallel "$MAX_PARALLEL" \
    --agent-model "$EXPERT_MODEL" \
    --agent-provider "$EXPERT_PROVIDER" \
    --agent-api-key "${EXPERT_API_KEY:-$OPENAI_API_KEY}" \
    || echo "  Warning: some rollouts failed for $task_id (continuing)"

  find output -name "daytona-state.json" -not -path "*/environments/*" -exec rm -f {} \; 2>/dev/null

  if [[ $((count % 25)) -eq 0 ]]; then
    artifacts=$(find output -name artifacts.json 2>/dev/null | wc -l | tr -d ' ')
    echo "  [$count/$TRAIN_COUNT] artifacts=$artifacts skipped=$skipped ($(date +%H:%M:%S))"
  fi
done

ARTIFACT_COUNT=$(find output -name artifacts.json 2>/dev/null | wc -l | tr -d ' ')
echo "  Collected $ARTIFACT_COUNT artifacts ($skipped tasks skipped)"
echo ""

# ── Step 5: Convert to Tinker JSONL ──────────────────────────────────────────
#
# Emits training_data.jsonl in Qwen3.5 native format:
#   - assistant tool_calls serialized as
#       <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>
#   - tool responses wrapped in <tool_response>...</tool_response>
#   - assistant turns prefixed with empty <think>\n\n</think>\n\n block
#     (matches enable_thinking=false in the Qwen3.5 chat template)
# TODO[export]: implement the Qwen3.5 native format in scripts/export_sft.py.
#               The current implementation emits Qwen3-4B-Instruct-2507's
#               JSON-arguments tool-call format which Qwen3.5 was not trained on.

echo "=== Step 5: Converting artifacts to Tinker JSONL (Qwen3.5 format) ==="
python3 "$SCRIPTS/export_sft.py" output/ training_data.jsonl "$MIN_REWARD"

# ── Step 5b: Filter sequences that exceed model context ─────────────────────
#
# Qwen3.5-4B has a 65,536 token context window. Drop any rollout whose
# rendered token length (computed with the Qwen3.5-4B tokenizer) exceeds
# MAX_TRAIN_TOKENS so that the trainer doesn't silently truncate.
# TODO[export]: implement --max-tokens N as a flag to export_sft.py and
#               surface the kept/dropped counts. For now this step is a no-op.

echo "=== Step 5b: Filtering sequences > $MAX_TRAIN_TOKENS tokens ==="
echo "  TODO: not yet implemented; see scripts/export_sft.py"

TRAJ_COUNT=$(wc -l < training_data.jsonl | tr -d ' ')
if [[ "$TRAJ_COUNT" -eq 0 ]]; then
  echo "ERROR: No trajectories converted. Check rollout output."
  simlab env down "$ENV_NAME" --daytona 2>/dev/null || true
  exit 1
fi
echo ""

# ── Step 6: Run Tinker SFT ──────────────────────────────────────────────────

echo "=== Step 6: Running Tinker SFT on $STUDENT_MODEL ==="

TRAIN_ARGS=(
  --model "$STUDENT_MODEL"
  --renderer "$RENDERER"
  --data training_data.jsonl
  --lr "$LEARNING_RATE"
  --max-length "$MAX_LENGTH"
  --lora-rank "$LORA_RANK"
  --epochs "$EPOCHS"
  --save-name "$SAVE_NAME"
)
[[ "$BATCH_SIZE" -gt 0 ]] && TRAIN_ARGS+=(--batch-size "$BATCH_SIZE")
[[ "$GRAD_ACCUM" -gt 1 ]] && TRAIN_ARGS+=(--grad-accum "$GRAD_ACCUM")

python3 "$SCRIPTS/train_sft.py" "${TRAIN_ARGS[@]}"
echo ""

# ── Step 7: Evaluate fine-tuned model ────────────────────────────────────────
#
# TODO[eval]: scripts/eval_agent.py must:
#   1. Parse Qwen3.5 XML tool-calls
#       <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>
#      The current parser only understands the JSON-arguments format.
#   2. Catch BadRequestError "Prompt length plus max_tokens exceeds the
#      model's context window" and end the rollout gracefully (record
#      `error="context_window_exceeded"`, return current state to verifier).
#      The verifier will naturally score the partial trajectory as 0.

echo "=== Step 7: Evaluating fine-tuned model on $EVAL_COUNT eval tasks ==="

# train_sft.py writes the fully-qualified checkpoint path to checkpoint_path.txt
CHECKPOINT=$(cat checkpoint_path.txt 2>/dev/null | tr -d '\n')
if [[ -z "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint_path.txt not found. Did train_sft.py complete successfully?"
  exit 1
fi

export TINKER_BASE_MODEL="$STUDENT_MODEL"
export TINKER_RENDERER="$RENDERER"
export TINKER_CHECKPOINT="$CHECKPOINT"

bash "$SCRIPTS/run_eval.sh" \
  --tasks-dir "$EVAL_TASKS_DIR" \
  --env "$ENV_NAME" \
  --checkpoint "$CHECKPOINT" \
  --base-model "$STUDENT_MODEL" \
  --renderer "$RENDERER" \
  --rollout-count "$ROLLOUT_COUNT" \
  --max-steps "$EVAL_MAX_STEPS" \
  --label "sft"
echo ""

# ── Step 8: Evaluate base model (optional comparison) ───────────────────────

echo "=== Step 8: Evaluating base model for comparison ==="

bash "$SCRIPTS/run_eval.sh" \
  --tasks-dir "$EVAL_TASKS_DIR" \
  --env "$ENV_NAME" \
  --base-model "$STUDENT_MODEL" \
  --renderer "$RENDERER" \
  --rollout-count "$ROLLOUT_COUNT" \
  --max-steps "$EVAL_MAX_STEPS" \
  --label "base"
echo ""

# ── Step 9: Tear down ───────────────────────────────────────────────────────

echo "=== Step 9: Tearing down environment ==="
simlab env down "$ENV_NAME" --daytona || echo "  Warning: teardown had issues (sandbox may need manual cleanup)"
echo ""

echo "=== Done ==="
echo "  Working directory: $WORKDIR"
echo "  Training data:     $WORKDIR/training_data.jsonl ($TRAJ_COUNT trajectories)"
echo "  Rollout artifacts: $WORKDIR/output/ ($ARTIFACT_COUNT artifacts)"
echo "  Train tasks:       $WORKDIR/$TRAIN_TASKS_DIR ($TRAIN_COUNT tasks)"
echo "  Eval tasks:        $WORKDIR/$EVAL_TASKS_DIR ($EVAL_COUNT tasks)"
echo "  Eval results:      $WORKDIR/eval_results_sft.json, $WORKDIR/eval_results_base.json"
