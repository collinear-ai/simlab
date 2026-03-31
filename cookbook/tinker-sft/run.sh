#!/usr/bin/env bash
#
# End-to-end SimLab + Tinker SFT pipeline.
#
# Creates a SimLab environment, generates tasks, collects expert rollouts,
# converts trajectories to Tinker format, and runs SFT training.
#
# Usage:
#   ./run.sh                           # uses defaults
#   ./run.sh --template erp            # pick a different SimLab template
#   ./run.sh --student-model Qwen/Qwen3-8B --renderer qwen3
#
# Required env vars (set in .env or export manually):
#   SIMLAB_COLLINEAR_API_KEY, DAYTONA_API_KEY, OPENAI_API_KEY, TINKER_API_KEY
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

TEMPLATE="${TEMPLATE:-erp}"
ENV_NAME="${ENV_NAME:-tinker-sft-env}"
NUM_TASKS="${NUM_TASKS:-4}"
ROLLOUT_COUNT="${ROLLOUT_COUNT:-4}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
EXPERT_MODEL="${EXPERT_MODEL:-gpt-4o-mini}"
EXPERT_PROVIDER="${EXPERT_PROVIDER:-openai}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
RENDERER="${RENDERER:-qwen3_instruct}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-32}"
MIN_REWARD="${MIN_REWARD:-0.0}"

# ── Parse CLI flags ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case $1 in
    --template)        TEMPLATE="$2";        shift 2 ;;
    --env-name)        ENV_NAME="$2";        shift 2 ;;
    --num-tasks)       NUM_TASKS="$2";       shift 2 ;;
    --rollout-count)   ROLLOUT_COUNT="$2";   shift 2 ;;
    --max-parallel)    MAX_PARALLEL="$2";    shift 2 ;;
    --expert-model)    EXPERT_MODEL="$2";    shift 2 ;;
    --expert-provider) EXPERT_PROVIDER="$2"; shift 2 ;;
    --student-model)   STUDENT_MODEL="$2";   shift 2 ;;
    --renderer)        RENDERER="$2";        shift 2 ;;
    --max-length)      MAX_LENGTH="$2";      shift 2 ;;
    --learning-rate)   LEARNING_RATE="$2";   shift 2 ;;
    --lora-rank)       LORA_RANK="$2";       shift 2 ;;
    --min-reward)      MIN_REWARD="$2";      shift 2 ;;
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

command -v simlab >/dev/null 2>&1 || { echo "ERROR: simlab not found. Run: uv add 'simlab[daytona] @ git+https://github.com/collinear-ai/simlab.git'"; exit 1; }
python3 -c "import tinker_cookbook" 2>/dev/null || { echo "ERROR: tinker-cookbook not found. Run: uv pip install tinker-cookbook"; exit 1; }

echo "  Template:       $TEMPLATE"
echo "  Expert model:   $EXPERT_MODEL ($EXPERT_PROVIDER)"
echo "  Student model:  $STUDENT_MODEL (renderer: $RENDERER)"
echo "  Tasks:          $NUM_TASKS"
echo "  Rollouts/task:  $ROLLOUT_COUNT"
echo ""

# ── Working directory ────────────────────────────────────────────────────────

WORKDIR="$REPO_ROOT/tinker-sft-run-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "=== Working directory: $WORKDIR ==="
echo ""

# ── Step 1: Create environment ───────────────────────────────────────────────

echo "=== Step 1: Creating environment '$ENV_NAME' from template '$TEMPLATE' ==="
simlab env init "$ENV_NAME" --template "$TEMPLATE" --non-interactive
simlab env up "$ENV_NAME" --daytona
echo ""

# ── Step 2: Generate tasks ───────────────────────────────────────────────────

echo "=== Step 2: Generating $NUM_TASKS tasks ==="
simlab tasks-gen init --template "$TEMPLATE" --output-dir ./taskgen
python3 "$SCRIPTS/patch_taskgen_config.py" taskgen/config.toml "$NUM_TASKS"
simlab tasks-gen run --config ./taskgen/config.toml
echo ""

# ── Step 3: Collect expert rollouts ──────────────────────────────────────────

echo "=== Step 3: Collecting expert rollouts ==="

# Extract task IDs from generated task filenames
TASK_IDS=$(ls ./generated-tasks/tasks/*.json 2>/dev/null \
  | xargs -I{} basename {} .json)

if [[ -z "$TASK_IDS" ]]; then
  echo "ERROR: No tasks found. Check task generation output."
  simlab env down "$ENV_NAME" --daytona 2>/dev/null || true
  exit 1
fi

TASK_COUNT=$(echo "$TASK_IDS" | wc -l | tr -d ' ')
echo "  Found $TASK_COUNT task(s). Running $ROLLOUT_COUNT rollouts each..."

for task_id in $TASK_IDS; do
  echo "  Running: $task_id"
  simlab tasks run \
    --env "$ENV_NAME" \
    --task "$task_id" \
    --tasks-dir ./generated-tasks \
    --daytona \
    --rollout-count "$ROLLOUT_COUNT" \
    --max-parallel "$MAX_PARALLEL" \
    --agent-model "$EXPERT_MODEL" \
    --agent-provider "$EXPERT_PROVIDER" \
    --agent-api-key "$OPENAI_API_KEY" \
    || echo "  Warning: some rollouts failed for $task_id (continuing)"
done
echo ""

# ── Step 4: Convert to Tinker JSONL ──────────────────────────────────────────

echo "=== Step 4: Converting artifacts to Tinker JSONL ==="
python3 "$SCRIPTS/export_sft.py" output/ training_data.jsonl "$MIN_REWARD"

TRAJ_COUNT=$(wc -l < training_data.jsonl | tr -d ' ')
if [[ "$TRAJ_COUNT" -eq 0 ]]; then
  echo "ERROR: No trajectories converted. Check rollout output."
  simlab env down "$ENV_NAME" --daytona 2>/dev/null || true
  exit 1
fi
echo ""

# ── Step 5: Run Tinker SFT ──────────────────────────────────────────────────

echo "=== Step 5: Running Tinker SFT on $STUDENT_MODEL ==="
python3 "$SCRIPTS/train_sft.py" \
  --model "$STUDENT_MODEL" \
  --renderer "$RENDERER" \
  --data training_data.jsonl \
  --lr "$LEARNING_RATE" \
  --max-length "$MAX_LENGTH" \
  --lora-rank "$LORA_RANK"
echo ""

# ── Step 6: Tear down ───────────────────────────────────────────────────────

echo "=== Step 6: Tearing down environment ==="
simlab env down "$ENV_NAME" --daytona || echo "  Warning: teardown had issues (sandbox may need manual cleanup)"
echo ""

echo "=== Done ==="
echo "  Working directory: $WORKDIR"
echo "  Training data:     $WORKDIR/training_data.jsonl"
echo "  Rollout artifacts: $WORKDIR/output/"
