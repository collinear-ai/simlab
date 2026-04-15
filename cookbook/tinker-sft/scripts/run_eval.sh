#!/usr/bin/env bash
#
# Evaluate a Tinker model (base or fine-tuned) on SimLab eval tasks.
#
# Runs each task via simlab tasks run --daytona with TinkerEvalAgent,
# which loads the model via Tinker's sampling API. Rubric scoring is
# handled by simlab's built-in rubric judge (requires passthrough verifier
# in task JSON to trigger the rubric judge path).
#
# Usage:
#   # Evaluate SFT checkpoint:
#   ./run_eval.sh --tasks-dir ./eval-tasks --env sft-scale \
#       --checkpoint "tinker://...:train:0/weights/my-checkpoint"
#
#   # Evaluate base model:
#   ./run_eval.sh --tasks-dir ./eval-tasks --env sft-scale
#
#   # Override model/renderer:
#   ./run_eval.sh --tasks-dir ./eval-tasks --env sft-scale \
#       --base-model Qwen/Qwen3-8B --renderer qwen3
#
# Required env vars:
#   TINKER_API_KEY, DAYTONA_API_KEY, SIMLAB_COLLINEAR_API_KEY
#   SIMLAB_VERIFIER_MODEL, SIMLAB_VERIFIER_PROVIDER, SIMLAB_VERIFIER_API_KEY
#
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────

TASKS_DIR=""
ENV_NAME=""
CHECKPOINT=""
BASE_MODEL="${TINKER_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
RENDERER="${TINKER_RENDERER:-qwen3_instruct}"
ROLLOUT_COUNT="${ROLLOUT_COUNT:-2}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LABEL=""

# ── Parse CLI flags ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case $1 in
    --tasks-dir)     TASKS_DIR="$2";     shift 2 ;;
    --env)           ENV_NAME="$2";      shift 2 ;;
    --checkpoint)    CHECKPOINT="$2";    shift 2 ;;
    --base-model)    BASE_MODEL="$2";    shift 2 ;;
    --renderer)      RENDERER="$2";      shift 2 ;;
    --rollout-count) ROLLOUT_COUNT="$2"; shift 2 ;;
    --max-steps)     MAX_STEPS="$2";     shift 2 ;;
    --max-parallel)  MAX_PARALLEL="$2";  shift 2 ;;
    --label)         LABEL="$2";         shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$TASKS_DIR" || -z "$ENV_NAME" ]]; then
  echo "ERROR: --tasks-dir and --env are required"
  exit 1
fi

# ── Resolve paths and label ──────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$LABEL" ]]; then
  if [[ -n "$CHECKPOINT" ]]; then
    LABEL="sft"
  else
    LABEL="base"
  fi
fi

# ── Set Tinker env vars for eval_agent.py ────────────────────────────────────

export TINKER_BASE_MODEL="$BASE_MODEL"
export TINKER_RENDERER="$RENDERER"
if [[ -n "$CHECKPOINT" ]]; then
  export TINKER_CHECKPOINT="$CHECKPOINT"
else
  unset TINKER_CHECKPOINT 2>/dev/null || true
fi

# Make eval_agent.py importable
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ── Preflight ────────────────────────────────────────────────────────────────

echo "=== Eval ($LABEL) ==="
echo "  Model:      $BASE_MODEL"
echo "  Renderer:   $RENDERER"
echo "  Checkpoint: ${CHECKPOINT:-<none, using base model>}"
echo "  Tasks dir:  $TASKS_DIR"
echo "  Env:        $ENV_NAME"
echo "  Rollouts:   $ROLLOUT_COUNT"
echo "  Max steps:  $MAX_STEPS"
echo ""

# Ensure env is up
simlab env up "$ENV_NAME" --daytona 2>&1 | tail -1

# ── Per-label workspace (resumable; isolates eval artifacts from training) ──
#
# We cd into eval_runs/<label>/ before invoking simlab tasks run, so each label
# gets its own ./output/ directory. This:
#   1. keeps eval artifacts separate from training rollouts in the parent output/
#   2. lets us resume by skipping tasks that already have artifacts in this dir
#   3. allows clean re-aggregation per label

PARENT_DIR="$(pwd)"
EVAL_RUN_DIR="$PARENT_DIR/eval_runs/$LABEL"
mkdir -p "$EVAL_RUN_DIR"
# Symlink the env state so simlab can find environments/<env-name>/ relative to CWD
if [[ ! -e "$EVAL_RUN_DIR/environments" ]]; then
  ln -s "$PARENT_DIR/environments" "$EVAL_RUN_DIR/environments"
fi

# Resolve tasks-dir to an absolute path before we cd
ABS_TASKS_DIR="$(cd "$TASKS_DIR" && pwd)"

cd "$EVAL_RUN_DIR"
echo "  Eval workdir: $EVAL_RUN_DIR"
echo ""

# ── Run eval ─────────────────────────────────────────────────────────────────

TASK_IDS=$(ls "$ABS_TASKS_DIR/tasks/"*.json | xargs -I{} basename {} .json)
TASK_COUNT=$(echo "$TASK_IDS" | wc -l | tr -d ' ')
echo "Running $TASK_COUNT tasks (resumable: skips tasks with existing artifacts)..."
echo ""

count=0
skipped=0
for task_id in $TASK_IDS; do
  count=$((count + 1))

  # Resume check: skip if we already have an artifact for this task in this label's dir
  existing=$(find output -path "*${task_id}*" -name artifacts.json 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$existing" -ge 1 ]]; then
    skipped=$((skipped + 1))
    echo "[$count/$TASK_COUNT] ${task_id:0:60}... SKIP (already done)"
    continue
  fi

  # Clean stale per-rollout daytona-state.json from prior runs (env state is symlinked)
  find "$(pwd)" -name "daytona-state.json" -not -path "*/environments/*" -exec rm -f {} \; 2>/dev/null

  echo -n "[$count/$TASK_COUNT] ${task_id:0:60}... "

  if output=$(simlab tasks run \
    --env "$ENV_NAME" \
    --task "$task_id" \
    --tasks-dir "$ABS_TASKS_DIR" \
    --daytona \
    --rollout-count "$ROLLOUT_COUNT" \
    --max-parallel "$MAX_PARALLEL" \
    --max-steps "$MAX_STEPS" \
    --agent-import-path eval_agent:TinkerEvalAgent 2>&1); then
    rc=0
  else
    rc=$?
  fi

  # Extract key metrics from output
  steps=$(echo "$output" | grep -oE "steps=[0-9]+" | head -1 | grep -oE "[0-9]+" || echo "?")
  rubric_score=$(echo "$output" | grep -oE "score=[0-9.]+" | tail -1 | grep -oE "[0-9.]+" || echo "?")
  verdict=$(echo "$output" | grep "Rubric verdict" | head -1 | sed 's/.*Rubric verdict: //' || echo "")

  if [[ $rc -ne 0 ]]; then
    echo "FAILED (exit $rc)"
  elif [[ -n "$verdict" ]]; then
    echo "steps=$steps rubric=$rubric_score ($verdict)"
  else
    echo "steps=$steps (no rubric)"
  fi
done

echo ""
echo "=== Collecting results ($skipped skipped) ==="

# Aggregate reward scores from this label's output dir.
# TODO: the reward.json schema is {reward, verifier_results: [...]}.
# This aggregator reads d["reward"] directly. If the schema changes
# upstream, update the key here.
python3 -c "
import json
from pathlib import Path

pairs = []
for rj in sorted(Path('output').rglob('reward.json')):
    d = json.loads(rj.read_text())
    reward = d.get('reward')
    if reward is None:
        continue
    steps = 0
    arts = rj.parent.parent / 'artifacts.json'
    if arts.exists():
        a = json.loads(arts.read_text())
        steps = a.get('steps_taken', 0)
    pairs.append((float(reward), steps))

rewards = [r for r, _ in pairs]
steps_list = [s for _, s in pairs]

print(f'Label:         $LABEL')
print(f'Scored tasks:  {len(rewards)}')
if rewards:
    print(f'Avg reward:    {sum(rewards)/len(rewards):.3f}')
    print(f'Min/Max:       {min(rewards):.2f} / {max(rewards):.2f}')
    passed = sum(1 for r in rewards if r > 0)
    print(f'Passed:        {passed}/{len(rewards)}')
if steps_list:
    print(f'Avg steps:     {sum(steps_list)/len(steps_list):.1f}')
    with_steps = sum(1 for s in steps_list if s > 0)
    print(f'Used tools:    {with_steps}/{len(steps_list)}')

results = [{'reward': r, 'steps': s} for r, s in pairs]
out_path = Path('$PARENT_DIR') / f'eval_results_$LABEL.json'
out_path.write_text(json.dumps(results, indent=2))
print(f'Saved to {out_path}')
"

echo ""
echo "=== Done ($LABEL) $(date) ==="
